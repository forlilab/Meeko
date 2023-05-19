#!/usr/bin/env python

from os import linesep as os_linesep
import json
import pathlib

pkg_dir = pathlib.Path(__file__).parents[0]
with open(pkg_dir / "data" / "residue_params.json") as f:
    residue_params = json.load(f)
with open(pkg_dir / "data" / "flexres_templates.json") as f:
    flexres_templates = json.load(f)
# the above is controversial, see
# https://stackoverflow.com/questions/6028000/how-to-read-a-static-file-from-inside-a-python-package


def write_pdbqt_line(atomidx, x, y, z, charge, atom_name, res_name, res_num, atom_type, chain=""):
    record_type = "ATOM"
    alt_id = " "
    in_code = ""
    occupancy = 1.0
    temp_factor = 0.0
    line = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}    {:6.3f} {:<2s}\n"
    return line.format(record_type, atomidx, atom_name, alt_id, res_name, chain,
                   res_num, in_code, x, y, z,
                   occupancy, temp_factor, charge, atom_type)


class Receptor:

    flexres_templates = flexres_templates


    def __init__(self, pdbqt_string):
        self.lines = pdbqt_string.split(os_linesep)
        line_idx_by_res, chains = self.get_line_indices_by_residue(self.lines)
        self.line_idx_by_res = line_idx_by_res
        self.chains = chains # to print helpful msg when residue ID does not match


    @classmethod
    def from_pdbqt_filename(cls, pdbqt_filename):
        with open(pdbqt_filename) as f:
            pdbqt_string = f.read()
        return cls(pdbqt_string)


    @staticmethod
    def get_line_indices_by_residue(lines):
        """ return a dict where residues are keys and values are
            lists of line indices
                
            >>> line_idx_by_res = {("A", "LYS", 417): [0, 1, 2, 3, ..., 8]}
        """

        line_idx_by_res = {}
        chains = set()
        for line_index, line in enumerate(lines):
            if line.startswith('ATOM') or line.startswith('HETATM'):
                chain =   line[21]
                resname = line[17:20].strip()
                resnum =  int(line[22:26])
                res_id = (chain, resname, resnum)
                line_idx_by_res.setdefault(res_id, [])
                line_idx_by_res[res_id].append(line_index)
                chains.add(chain)
        return line_idx_by_res, chains


    @staticmethod
    def assign_residue_params(res_list, atom_names_list, residue_params=residue_params):
        if len(res_list) != len(atom_names_list):
            raise ValueError("length of res_list differs from that of atom_names_list")
        atom_params = {}
        resid = 0
        atom_counter = 0
        for (res, atom_names) in zip(res_list, atom_names_list):
            resid += 1
            is_matched = False
            for terminus in ["", "N", "C"]:
                r_id = "%s%s" % (terminus, res) # e.g. "CTYR" for C-term TYR
                ref_names = set(residue_params[r_id]["atom_names"])
                query_names = set(atom_names)
                if ref_names == query_names:
                    is_matched = True
                    break
            if not is_matched:
                raise RuntimeError("did not match atom_names of res (%s, %d): %s" % (res, resid, query_names))
    
            for atom_name in atom_names:
                name_index = residue_params[r_id]["atom_names"].index(atom_name)
                for param in residue_params[r_id].keys():
                    if param == "atom_names":
                        continue
                    if param not in atom_params:
                        atom_params[param] = [None] * atom_counter 
                    value = residue_params[r_id][param][name_index]
                    atom_params[param].append(value)
                atom_counter += 1
    
            # fill None for params missing in residue_params for this residue
            for param in atom_params.keys():
                if param not in residue_params[r_id]:
                    atom_params[param].extend([None] * len(atom_names))
    
        return atom_params 
    

    @staticmethod
    def parse_residue_data_from_pdb(pdb_string):
        data = {
            "res_ids": [],
            "resnames": [],
            "atom_names": [],
            "positions": [],
        }
        resid_set = set()
        success = True
        error_msg = ""
        line_count = 0
        for line in pdb_string.split("\n"):
            line_count += 1
            if line.startswith("HETATM"):
                success = False
                error_msg += "found HETATM record on line %d\n" % line_count
            elif line.startswith("ATOM"):
                resname = line[17:20] 
                resnum = int(line[22:26])
                chain = line[21:22]
                resid = (chain, resnum, resname)
                if resid not in resid_set:
                    resid_set.add(resid)
                    data["res_ids"].append(resid)
                    data["resnames"].append(resname)
                    data["atom_names"].append([])
                    data["positions"].append([])
                atom_name = line[12:16].strip()
                data["atom_names"][-1].append(atom_name)
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                data["positions"][-1].append(xyz)
        return data, success, error_msg


    @staticmethod
    def write_pdbqt_from_residue_data(data, charges, atom_types, skip_types=("H",)):
        nr_res = len(data["resnames"])
        pdbqt_string = ""
        atom_index = 0
        for i in range(nr_res):
            resname = data["resnames"][i]
            chain = data["res_ids"][i][0]
            resnum = data["res_ids"][i][1]
            nr_atoms = len(data["atom_names"])
            for atom_name, xyz in zip(data["atom_names"][i], data["positions"][i]):
                if atom_types[atom_index] not in skip_types:
                    pdbqt_string += write_pdbqt_line(
                        atom_index + 1,
                        xyz[0],
                        xyz[1],
                        xyz[2],
                        charges[atom_index],
                        atom_name,
                        resname,
                        resnum,
                        atom_types[atom_index],
                    )
                atom_index += 1
        return pdbqt_string


    def prepare_flexres_from_template(self, res_id):
        success = True
        error_msg = ""
        output = {"pdbqt": "", "flex_lines": []}
        resname = res_id[1]
        template = self.flexres_templates[resname]
        if res_id not in self.line_idx_by_res:
            success = False
            error_msg += "could not find residue with chain='%s', resname=%s, resnum=%d" % res_id + os_linesep
            error_msg += "chains in this receptor: %s" % ", ".join("'%s'" % c for c in self.chains) + os_linesep
            if " " in self.chains:
                error_msg += "use ' ' (a space character) for empty chain" + os_linesep
            return output, success, error_msg
            
        # collect lines of res_id
        atom_lines_by_name = {}
        for i in self.line_idx_by_res[res_id]:
            line = self.lines[i]
            name = line[12:16].strip()
            if name in ['C', 'N', 'O', 'H', 'H1', 'H2', 'H3', 'OXT']: # skip backbone atoms
                continue
            output["flex_lines"].append(i)
            atom_lines_by_name[name] = line[0:6] + "%5d" + line[11:]

        # check it was a full match
        got_atoms = set(atom_lines_by_name)
        ref_atoms = set()
        for i in range(len(template["is_atom"])):
            if template["is_atom"][i]:
                ref_atoms.add(template["atom_name"][i])
        if got_atoms != ref_atoms: 
            success = False
            error_msg += "mismatch in atom names for residue %s" % str(res_id) + os_linesep
            error_msg += "names found but not in template: %s" % str(got_atoms.difference(ref_atoms)) + os_linesep
            error_msg += "missing names: %s" % str(ref_atoms.difference(got_atoms)) + os_linesep
            return output, success, error_msg

        # create output string
        n_lines = len(template['is_atom'])
        atom_index = 0
        for i in range(n_lines):
            if template['is_atom'][i]:
                atom_index += 1
                name = template['atom_name'][i]
                output["pdbqt"] += atom_lines_by_name[name] % atom_index + "\n"
            else:
                output["pdbqt"] += template['original_line'][i] + "\n" # e.g. BRANCH keywords

        return output, success, error_msg


    def write_pdbqt_string(self, flex_res=()):
        success = True
        error_msg = ""
        
        pdbqt = {"rigid": "",
                 "flex":  ""}
        flex_line_idxs = []
        for res_id in flex_res:
            output, success_, error_msg_ = self.prepare_flexres_from_template(res_id)
            success &= success_
            error_msg += error_msg_
            flex_line_idxs.extend(output["flex_lines"])
            pdbqt["flex"] += "BEGIN_RES %3s %1s%4d" % (res_id) + os_linesep
            pdbqt["flex"] += output["pdbqt"]
            pdbqt["flex"] += "END_RES %3s %1s%4d" % (res_id) + os_linesep

        # rigid
        for i, line in enumerate(self.lines):
            if i not in flex_line_idxs: 
                pdbqt["rigid"] += line + os_linesep
        
        return pdbqt, success, error_msg
