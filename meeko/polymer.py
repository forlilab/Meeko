import pathlib
import json
import logging
import traceback
from importlib.resources import files
from os import linesep as eol
from sys import exc_info
from typing import Union
from typing import Optional
import numpy as np

import rdkit.Chem
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdChemReactions
from rdkit.Chem import rdMolInterchange
from rdkit.Geometry import Point3D

from .molsetup import RDKitMoleculeSetup
from .molsetup import MoleculeSetupEncoder
from .utils.jsonutils import rdkit_mol_from_json
from .utils.rdkitutils import mini_periodic_table
from .utils.rdkitutils import react_and_map
from .utils.rdkitutils import AtomField
from .utils.rdkitutils import build_one_rdkit_mol_per_altloc
from .utils.rdkitutils import _aux_altloc_mol_build
from .utils.rdkitutils import covalent_radius
from .utils.pdbutils import PDBAtomInfo
from .chemtempgen import export_chem_templates_to_json
from .chemtempgen import build_noncovalent_CC
from .chemtempgen import build_linked_CCs

# @alphataubio refactoring [2025/03]
from .residue_chem_templates import ResidueChemTemplates
from .monomer import Monomer, residues_rotamers
from .encoders import *
from .decoders import *

data_path = files("meeko") / "data"
periodic_table = Chem.GetPeriodicTable()

try:
    import prody
except ImportError as _prody_import_error:
    ALLOWED_PRODY_TYPES = None
    AtomGroup = None
    Selection = None
    def prody_to_rdkit(*args):
        raise ImportError(_prody_import_error)
else:
    from .utils.prodyutils import prody_to_rdkit, ALLOWED_PRODY_TYPES
    from prody.atomic.atomgroup import AtomGroup
    from prody.atomic.selection import Selection

logger = logging.getLogger(__name__)
rdkit_logger = logging.getLogger("rdkit")

class Polymer:
    """Represents polymer with its subunits as individual RDKit molecules.

    Used for proteins and nucleic acids. The key class is Monomer,
    which contains, a padded RDKit molecule containing part of the adjacent
    residues to enable chemically meaningful parameterizaion.
    Instances of ResidueTemplate make sure that the input, which may originate
    from a PDB string, matches the RDKit molecule of the template, even if
    hydrogens are missing.

    Attributes
    ----------
    monomers: dict (string -> Monomer) #TODO: figure out exact SciPy standard for dictionary key/value notation
    termini: dict (string (representing residue id) -> string (representing what we want the capping to look like))
    mutate_res_dict: dict (string (representing starting residue id) -> string (representing the desired mutated id))
    res_templates: dict (string -> dict (rdkit_mol and atom_data))
    ambiguous:
    disulfide_bridges:
    suggested_mutations:
    """

    def __init__(
        self,
        raw_input_mols: dict[str, tuple[Chem.Mol, str]],
        bonds: dict[tuple[str, str], tuple[int, int]],
        residue_chem_templates: ResidueChemTemplates,
        mk_prep=None,
        set_template: dict[str, str] = None,
        blunt_ends: list[tuple[str, int]] = None,
    ):
        """
        Parameters
        ----------
        raw_input_mols: dict (string -> (Chem.Mol, string))
            A dictionary of raw input mols where keys are residue IDs in the format <chain>:<resnum> such as "A:42" and
            values are tuples of an RDKit Mols and input resname.
            RDKit Mols will be matched to instances of ResidueTemplate, and may contain none, all, or some of the
            Hydrogens.
        bonds: dict ((string, string) -> (int, int))
        residue_chem_templates: ResidueChemTemplates
            An instance of the ResidueChemTemplates class.
        mk_prep: MoleculePreparation
            An instance of the MoleculePreparation class to parameterize the padded molecules.
        set_template: dict (string -> string)
            A dict mapping residue IDs in the format <chain>:<resnum> such as "A:42" to ResidueTemplate instances.
        blunt_ends: list (tuple (string, int))
            A list of tuples where each tuple is residue IDs and 0-based atom index, e.g.; ("A:42", 0)

        Returns
        -------
        None

        Raises
        ------
        ValueError:
        """

        # TODO simplify SMARTS for adjacent res in padders

        if type(raw_input_mols) != dict:
            msg = f"expected raw_input_mols to be dict, got {type(raw_input_mols)}"
            if type(raw_input_mols) == str:
                msg += eol
                msg += (
                    "consider Polymer.from_pdb_string(pdbstr)" + eol
                )
            raise ValueError(msg)
        self.residue_chem_templates = residue_chem_templates
        residue_templates = residue_chem_templates.residue_templates
        padders = residue_chem_templates.padders
        ambiguous = residue_chem_templates.ambiguous

        if set_template is None:
            set_template = {}
        else:  # make sure all resiude_id in set_template exist
            missing = set(
                [
                    residue_id
                    for residue_id in set_template
                    if residue_id not in raw_input_mols
                ]
            )
            if len(missing):
                raise ValueError(
                    f"Residue IDs in set_template not found: {missing} {raw_input_mols.keys()}"
                )

        # check if input assigned residue name in residue_templates
        err = ""
        supported_resnames = residue_templates.keys() | ambiguous.keys()
        unknown_res_from_input = {res_id: raw_input_mols[res_id][1] 
                                  for res_id in raw_input_mols 
                                  if res_id not in set_template and raw_input_mols[res_id][1] not in supported_resnames
                                  }
        
        if unknown_res_from_input:
            unknown_valid_res_from_input = {k: v for k, v in unknown_res_from_input.items() if v != "UNL"}
            if unknown_valid_res_from_input: 
                err += f"Input residues {unknown_valid_res_from_input} not in residue_templates" + eol
            UNL_from_input = {k: v for k, v in unknown_res_from_input.items() if v == "UNL"}
            if UNL_from_input: 
                err += f"Input residues {UNL_from_input} do not have a concrete definition" + eol
        
        unknown_res_from_assign = {}
        if set_template:
            unknown_res_from_assign = {res_id: resn for res_id, resn in set_template.items() if resn not in supported_resnames}
            unknown_valid_res_from_assign = {k: v for k, v in unknown_res_from_assign.items() if v != "UNL"}
            if unknown_valid_res_from_assign: 
                err += f"Input residues {unknown_valid_res_from_assign} not in residue_templates" + eol
            UNL_from_assign = {k: v for k, v in unknown_res_from_assign.items() if v == "UNL"}
            if UNL_from_assign: 
                err += f"Input residues {UNL_from_assign} do not have a concrete definition" + eol
        
        if err:
            if "UNL" in err: 
                err += "Resdiues that are named UNL can't be parameterized. " + eol
                rec = "1. (to parameterize the residues) Use --set_template to specify valid residue names, " + eol
                rec += "2. (to skip the residues) Use --delete_residues to ignore them. Residues will be deleted from the prepared receptor. "
                raise PolymerCreationError(err, rec)

            print(err)
            print("Trying to resolve unknown residues by building chemical templates... ")

            all_unknown_res = unknown_res_from_input.copy()
            all_unknown_res.update(unknown_res_from_assign)

            bonded_unknown_res = {res_id: all_unknown_res[res_id] for res_id in all_unknown_res 
                                  if any(res_id in respair for respair in bonds)}

            unbound_unknown_res = all_unknown_res.copy()
            for key in bonded_unknown_res:
                unbound_unknown_res.pop(key, None) 

            if unbound_unknown_res: 
                for resname in set(unbound_unknown_res.values()): 
                    try: 
                        cc = build_noncovalent_CC(resname)
                        fetch_template_dict = json.loads(export_chem_templates_to_json([cc]))['residue_templates'][cc.resname]
                        residue_templates.update({resname: ResidueTemplate(
                                                    smiles = fetch_template_dict['smiles'],
                                                    atom_names = fetch_template_dict['atom_name'],
                                                    link_labels = fetch_template_dict['link_labels'])})
                        ambiguous[resname] = [cc.resname]
                    except Exception as e: 
                        print(f"Failed building template from CCD for {resname=}")
                        raise PolymerCreationError(str(e))

            if bonded_unknown_res: 
                failed_build = set()
                try: 
                    for resname in set(bonded_unknown_res.values()): 
                        cc_list = build_linked_CCs(resname)
                        if not cc_list: 
                            failed_build.add(resname)
                        else:
                            for cc in cc_list:
                                fetch_template_dict = json.loads(export_chem_templates_to_json([cc]))['residue_templates'][cc.resname]
                                residue_templates.update({cc.resname: ResidueTemplate(
                                                            smiles = fetch_template_dict['smiles'],
                                                            atom_names = fetch_template_dict['atom_name'],
                                                            link_labels = {int(key): value for key,value in fetch_template_dict['link_labels'].items()})})
                                if resname in ambiguous: 
                                    ambiguous[resname].append(cc.resname)
                                else:
                                    ambiguous[resname] = [cc.resname]
                except Exception as e: 
                    raise PolymerCreationError(str(e))
                            
                if failed_build: 
                    raise PolymerCreationError(f"Template generation failed for unknown residues: {failed_build}, which appear to be linking fragments. " + eol
                                            + "Generation of chemical templates with modified backbones, which involves guessing of linker positions and types, are not currently supported. ", 
                                            "1. (to parameterize the residues) Use --add_templates to pass the additional templates with valid linker_labels, " + eol
                                            + "2. (to skip the residues) Use --delete_residues to ignore them. Residues will be deleted from the prepared receptor. ")

        self.monomers, self.log = self._get_monomers(
            raw_input_mols,
            ambiguous,
            residue_chem_templates,
            set_template,
            bonds,
            blunt_ends,
        )

        _bonds = {}
        for key, bond_list in bonds.items():
            monomer1 = self.monomers[key[0]]
            monomer2 = self.monomers[key[1]]
            if monomer1.rdkit_mol is None or monomer2.rdkit_mol is None:
                continue
            invmap1 = {j: i for i, j in monomer1.mapidx_to_raw.items()}
            invmap2 = {j: i for i, j in monomer2.mapidx_to_raw.items()}
            _bonds[key] = [(invmap1[b[0]], invmap2[b[1]]) for b in bond_list]
        bonds = _bonds

        # padding may seem overkill but we had to run a reaction anyway for h_coord_from_dipep
        padded_mols = self._build_padded_mols(self.monomers, bonds, padders)
        for residue_id, (padded_mol, mapidx_from_pad) in padded_mols.items():
            monomer = self.monomers[residue_id]
            monomer.padded_mol = padded_mol
            monomer.molsetup_mapidx = mapidx_from_pad

        if mk_prep is not None:
            self.parameterize(mk_prep)

        return

    @classmethod
    def from_pdb_string(
        cls,
        pdb_string,
        chem_templates,
        mk_prep,
        set_template=None,
        residues_to_delete=None,
        allow_bad_res=False,
        bonds_to_delete=None,
        blunt_ends=None,
        wanted_altloc=None,
        default_altloc=None
    ):
        """

        Parameters
        ----------
        pdb_string
        chem_templates
        mk_prep
        set_template
        residues_to_delete
        allow_bad_res
        bonds_to_delete
        blunt_ends
        wanted_altloc
        default_altloc

        Returns
        -------

        """

        tmp_raw_input_mols = cls._pdb_to_residue_mols(
            pdb_string,
            wanted_altloc,
            default_altloc,
        )

        # from here on it duplicates self.from_prody(), but extracting
        # this out into a function felt like it sacrificed readibility
        # so I decided to keep the duplication.
        _delete_residues(residues_to_delete, tmp_raw_input_mols)
        raw_input_mols = {}
        res_needed_altloc = []
        res_missed_altloc = []
        unparsed_res = []

        charmm_to_amber_histidine = {"HSD": "HID", "HSE": "HIE", "HSP": "HIP"}

        for res_id, stuff in tmp_raw_input_mols.items():
            mol, resname, missed_altloc, needed_altloc = stuff

            # Convert CHARMM histidine names before assignment
            if resname in charmm_to_amber_histidine:
                resname = charmm_to_amber_histidine[resname]

            if mol is None and missed_altloc:
                res_missed_altloc.append(res_id)
            elif mol is None and needed_altloc:
                res_needed_altloc.append(res_id)
            elif mol is None:
                unparsed_res.append(res_id)
            else:
                raw_input_mols[res_id] = (mol, resname)
        bonds = find_inter_mols_bonds(raw_input_mols)
        if bonds_to_delete is not None:
            for res1, res2 in bonds_to_delete:
                popped = ()
                if (res1, res2) in bonds:
                    popped = bonds.pop((res1, res2))
                elif (res2, res1) in bonds:
                    popped = bonds.pop((res2, res1))
                if len(popped) >= 2:
                    msg = (
                        "can't delete bonds for residue pairs that have more"
                        " than one bond between them"
                    )
                    raise NotImplementedError(msg)
        polymer = cls(
            raw_input_mols,
            bonds,
            chem_templates,
            mk_prep,
            set_template,
            blunt_ends,
        )

        unmatched_res = polymer.get_ignored_monomers()
        handle_parsing_situations(
            unmatched_res,
            unparsed_res,
            allow_bad_res,
            res_missed_altloc,
            res_needed_altloc,
        )

        return polymer


    @classmethod
    def from_prody(
        cls,
        prody_obj: Union[Selection, AtomGroup],
        chem_templates,
        mk_prep,
        set_template=None,
        residues_to_delete=None,
        allow_bad_res=False,
        bonds_to_delete=None,
        blunt_ends=None,
        wanted_altloc: Optional[dict]=None,
        default_altloc: Optional[str]=None,
    ):
        """

        Parameters
        ----------
        prody_obj
        chem_templates
        mk_prep
        set_template
        residues_to_delete
        allow_bad_res
        bonds_to_delete
        blunt_ends
        wanted_altloc
        default_altloc

        Returns
        -------

        """

        tmp_raw_input_mols = cls._prody_to_residue_mols(
            prody_obj,
            wanted_altloc,
            default_altloc,
        )

        # from here on it duplicates self.from_pdb_string(), but extracting
        # this out into a function felt like it sacrificed readibility
        # so I decided to keep the duplication.
        _delete_residues(residues_to_delete, tmp_raw_input_mols)
        raw_input_mols = {}
        res_needed_altloc = []
        res_missed_altloc = []
        unparsed_res = []
        for res_id, stuff in tmp_raw_input_mols.items():
            mol, resname, missed_altloc, needed_altloc = stuff
            if mol is None and missed_altloc:
                res_missed_altloc.append(res_id)
            elif mol is None and needed_altloc:
                res_needed_altloc.append(res_id)
            elif mol is None:
                unparsed_res.append(res_id)
            else:
                raw_input_mols[res_id] = (mol, resname)

        bonds = find_inter_mols_bonds(raw_input_mols)
        if bonds_to_delete is not None:
            for res1, res2 in bonds_to_delete:
                popped = ()
                if (res1, res2) in bonds:
                    popped = bonds.pop((res1, res2))
                elif (res2, res1) in bonds:
                    popped = bonds.pop((res2, res1))
                if len(popped) >= 2:
                    msg = (
                        "can't delete bonds for residue pairs that have more"
                        " than one bond between them"
                    )
                    raise NotImplementedError(msg)
        polymer = cls(
            raw_input_mols,
            bonds,
            chem_templates,
            mk_prep,
            set_template,
            blunt_ends,
        )
        unmatched_res = polymer.get_ignored_monomers()
        handle_parsing_situations(
            unmatched_res,
            unparsed_res,
            allow_bad_res,
            res_missed_altloc,
            res_needed_altloc,
        )

        return polymer

    @classmethod
    def from_json(cls, json_string):
        return json.loads(
            json_string,
            object_hook=polymer_json_decoder,
        )

    def to_json(self):
        return json.dumps(self, cls=PolymerEncoder)

    def parameterize(self, mk_prep):
        """

        Parameters
        ----------
        mk_prep

        Returns
        -------

        """

        for residue_id in self.get_valid_monomers():
            self.monomers[residue_id].parameterize(mk_prep, residue_id)

    @staticmethod
    def _build_rdkit_mol(raw_mol, template, mapping, nr_missing_H):
        """

        Parameters
        ----------
        raw_mol
        template
        mapping
        nr_missing_H

        Returns
        -------

        """
        rdkit_mol = Chem.Mol(template.mol)  # making a copy
        conf = Chem.Conformer(rdkit_mol.GetNumAtoms())
        input_conf = raw_mol.GetConformer()
        for i, j in mapping.items():
            conf.SetAtomPosition(i, input_conf.GetAtomPosition(j))

        rdkit_mol.AddConformer(conf, assignId=True)

        if nr_missing_H:  # add positions to Hs missing in raw_mol
            if rdkit_mol.GetNumAtoms() != len(mapping) + nr_missing_H:
                raise RuntimeError(
                    f"nr of atoms ({rdkit_mol.GetNumAtoms()}) != "
                    f"{len(mapping)=} + {nr_missing_H=}"
                )
            idxs = [i for i in range(rdkit_mol.GetNumAtoms()) if i not in mapping]
            update_H_positions(rdkit_mol, idxs)

        return rdkit_mol

    @staticmethod
    def _get_best_missing_Hs(results):
        """

        Parameters
        ----------
        results

        Returns
        -------

        """
        min_missing_H = 999999
        best_idxs = []
        fail_log = []
        for i, result in enumerate(results):
            fail_log.append([])
            if result["heavy"]["missing"] > 0:
                fail_log[-1].append("heavy missing")
            if result["heavy"]["excess"] > 0:
                fail_log[-1].append("heavy excess")
            if len(result["H"]["excess"]) > 0:
                fail_log[-1].append("H excess")
            if len(result["bonds"]["excess"]) > 0:
                fail_log[-1].append("bonds excess")
            if len(result["bonds"]["missing"]) > 0:
                fail_log[-1].append(f"bonds missing at {result['bonds']['missing']}")
            if len(fail_log[-1]):
                continue
            if result["H"]["missing"] < min_missing_H:
                best_idxs = []
                min_missing_H = result["H"]["missing"]
            if result["H"]["missing"] == min_missing_H:
                best_idxs.append(i)
        return best_idxs, fail_log

    @classmethod
    def _get_monomers(
        cls,
        raw_input_mols,
        ambiguous,
        residue_chem_templates,
        set_template,
        bonds,
        blunt_ends,
    ):
        """

        Parameters
        ----------
        raw_input_mols
        ambiguous
        residue_chem_templates
        set_template
        bonds
        blunt_ends

        Returns
        -------

        """

        residue_templates = residue_chem_templates.residue_templates
        monomers = {}
        log = {
            "chosen_by_fewest_missing_H": {},
            "chosen_by_default": {},
            "no_match": [],
            "no_mol": [],
            "msg": "",
        }
        for residue_key, (raw_mol, input_resname) in raw_input_mols.items():
            if raw_mol is None:
                monomers[residue_key] = Monomer(
                    None, None, None, input_resname, None
                )
                log["no_mol"].append(residue_key)
                logger.warning(f"molecule for {residue_key=} is None")
                continue

            raw_mol_has_H = sum([a.GetAtomicNum() == 1 for a in raw_mol.GetAtoms()]) > 0
            excess_H_ok = False
            if set_template is not None and residue_key in set_template:
                excess_H_ok = True  # e.g. allow set LYN (NH2) from LYS (NH3+)
                template_key = set_template[residue_key]  # e.g. HID, NALA
                if template_key not in residue_templates: 
                    if template_key in ambiguous: 
                        raise RuntimeError(f"Can't assign an ambiguous tamplate_key ({template_key}) to residue ({residue_key}). ")
                    raise RuntimeError(f"Assigned tamplate_key ({template_key}) for residue ({residue_key}) is not in residue_templates. ")
                template = residue_templates[template_key]
                candidate_template_keys = [set_template[residue_key]]
                candidate_templates = [template]

            elif input_resname not in ambiguous:
                template_key = input_resname
                template = residue_templates[template_key]
                candidate_template_keys = [template_key]
                candidate_templates = [template]
            elif len(ambiguous[input_resname]) == 1:
                template_key = ambiguous[input_resname][0]
                template = residue_templates[template_key]
                candidate_template_keys = [template_key]
                candidate_templates = [template]
            else:
                candidate_template_keys = []
                candidate_templates = []
                for key in ambiguous[input_resname]:
                    template = residue_templates[key]
                    candidate_templates.append(template)
                    candidate_template_keys.append(key)

            # gather raw_mol atoms that have bonds or blunt ends
            if blunt_ends is None:
                blunt_ends = []
            raw_atoms_with_bonds = []
            for (r1, r2), bond_list in bonds.items():
                for i, j in bond_list:
                    if r1 == residue_key:
                        raw_atoms_with_bonds.append(i)
                    if r2 == residue_key:
                        raw_atoms_with_bonds.append(j)

            all_stats = {
                "heavy_missing": [],
                "heavy_excess": [],
                "H_excess": [],
                "H_missing": [],
                "bonded_atoms_missing": [],
                "bonded_atoms_excess": [],
            }
            mappings = []
            for index, template in enumerate(candidate_templates):

                # match intra-residue graph
                match_stats, mapping = template.match(raw_mol)
                mappings.append(mapping)

                # match inter-residue bonds
                atoms_with_bonds = set()
                from_raw = {value: key for (key, value) in mapping.items()}
                for raw_index in raw_atoms_with_bonds:
                    if raw_index in from_raw:  # bonds can occur on atoms the template does not have
                        atom_index = from_raw[raw_index]
                        atoms_with_bonds.add(atom_index)
                # we treat blunt ends like bonds
                for res_id, atom_idx in blunt_ends:
                    if res_id == residue_key:
                        atoms_with_bonds.add(from_raw[atom_idx])
                expected = set(template.link_labels)
                bonded_atoms_found = atoms_with_bonds.intersection(expected)
                bonded_atoms_missing = expected.difference(atoms_with_bonds)
                bonded_atoms_excess = atoms_with_bonds.difference(expected)

                all_stats["heavy_missing"].append(match_stats["heavy"]["missing"])
                all_stats["heavy_excess"].append(match_stats["heavy"]["excess"])
                all_stats["H_excess"].append(match_stats["H"]["excess"])
                all_stats["H_missing"].append(match_stats["H"]["missing"])
                all_stats["bonded_atoms_missing"].append(bonded_atoms_missing)
                all_stats["bonded_atoms_excess"].append(bonded_atoms_excess)

            passed = []

            embedded_indices = [index for index, template in enumerate(candidate_templates) if len(template.link_labels) >= 2]
            # 1st round
            for i in embedded_indices:
                if (
                    all_stats["heavy_missing"][i]
                    or all_stats["heavy_excess"][i]
                    or all_stats["H_excess"][i]
                    or all_stats["bonded_atoms_missing"][i]
                    or len(all_stats["bonded_atoms_excess"][i])
                ):
                    continue
                passed.append(i)

            # 2nd round
            if len(passed) == 0: 
                for i in embedded_indices:
                    auto_blunt = set()
                    for j, padder_label in candidate_templates[i].link_labels.items():
                        if residue_chem_templates.padders[padder_label].auto_blunt:
                            auto_blunt.add(j)
                    if (
                        all_stats["heavy_missing"][i]
                        or all_stats["heavy_excess"][i]
                        or (not set(all_stats["H_excess"][i]) <= set(candidate_templates[i].link_labels) and not excess_H_ok)
                        or not all_stats["bonded_atoms_missing"][i] <= auto_blunt
                        or len(all_stats["bonded_atoms_excess"][i])
                    ):
                        continue
                    passed.append(i)

            # 3rd round
            if len(passed) == 0: 
                for i in range(len(candidate_templates)):
                    if (
                        all_stats["heavy_missing"][i]
                        or all_stats["heavy_excess"][i]
                        or (all_stats["H_excess"][i] and not excess_H_ok)
                        or len(all_stats["bonded_atoms_missing"][i])
                        or len(all_stats["bonded_atoms_excess"][i])
                    ):
                        continue
                    passed.append(i)

            if len(passed) == 0:
                template_key = None
                template = None
                mapping = None
                m = f"No template matched for {residue_key=}" + eol
                m += f"tried {len(candidate_templates)} templates for {residue_key=}"
                m += f"{excess_H_ok=}"
                m += eol
                for i in range(len(all_stats["H_excess"])):
                    heavy_miss = all_stats["heavy_missing"][i]
                    heavy_excess = all_stats["heavy_excess"][i]
                    H_excess = all_stats["H_excess"][i]
                    bond_miss = all_stats["bonded_atoms_missing"][i]
                    bond_excess = all_stats["bonded_atoms_excess"][i]
                    tkey = candidate_template_keys[i]
                    m += (
                        f"{tkey:10} {heavy_miss=} {heavy_excess=} {H_excess=} {bond_miss=} {bond_excess=}"
                        + eol
                    )
                logger.warning(m)
            elif len(passed) == 1 or not raw_mol_has_H:
                index = passed[0]
                template_key = candidate_template_keys[index]
                template = candidate_templates[index]
                mapping = mappings[index]
                H_miss = all_stats["H_missing"][index]
            else:
                min_missing_H = 999999
                for i, index in enumerate(passed):
                    H_missed = all_stats["H_missing"][index]
                    if H_missed < min_missing_H:
                        best_idxs = []
                        min_missing_H = H_missed
                    if H_missed == min_missing_H:
                        best_idxs.append(index)

                if len(best_idxs) > 1:
                    tied = " ".join(candidate_template_keys[i] for i in best_idxs)
                    m = f"for {residue_key=}, {len(passed)} have passed: "
                    tkeys = [candidate_template_keys[i] for i in passed]
                    m += f"{tkeys} and tied for fewest missing H: {tied} "
                    raise RuntimeError(m)
                elif len(best_idxs) == 0:
                    raise RuntimeError("unexpected situation")
                else:
                    index = best_idxs[0]
                    template_key = candidate_template_keys[index]
                    template = residue_templates[template_key]
                    mapping = mappings[index]
                    H_miss = all_stats["H_missing"][index]
                    log["chosen_by_fewest_missing_H"][residue_key] = template_key
            if template is None:
                rdkit_mol = None
                atom_names = None
                mapping = None
            else:
                rdkit_mol = cls._build_rdkit_mol(
                    raw_mol,
                    template,
                    mapping,
                    H_miss,
                )
                atom_names = template.atom_names
            monomers[residue_key] = Monomer(
                raw_mol,
                rdkit_mol,
                mapping,
                input_resname,
                template_key,
                atom_names,
            )
            monomers[residue_key].template = template
            if template is not None and template.link_labels is not None:
                mapping_inv = monomers[
                    residue_key
                ].mapidx_from_raw  # {j: i for (i, j) in mapping.items()}
                # TODO check here mapping_inv unnused
                link_labels = {i: label for i, label in template.link_labels.items()}
                monomers[residue_key].link_labels = link_labels

        return monomers, log

    @staticmethod
    def _build_padded_mols(monomers, bonds, padders):
        """

        Parameters
        ----------
        monomers
        bonds
        padders

        Returns
        -------

        """
        padded_mols = {}
        bond_use_count = {key: 0 for key in bonds}
        for (
            residue_id,
            monomer,
        ) in monomers.items():
            if monomer.rdkit_mol is None:
                continue
            padded_mol = monomer.rdkit_mol
            mapidx_pad = {
                atom.GetIdx(): atom.GetIdx() for atom in padded_mol.GetAtoms()
            }
            for atom_index, link_label in monomer.link_labels.items():
                adjacent_rid = None
                adjacent_mol = None
                adjacent_atom_index = None
                for (r1_id, r2_id), bond_list in bonds.items():
                    # TODO the second and subsequent bonds between a pair of
                    # residues will not update the padding atoms with the
                    # positions of the adjacent residues. This is OK, the same
                    # happens for blunt residues, because the adjacent residue
                    # is missing.
                    i1, i2 = bond_list[0]
                    if r1_id == residue_id and i1 == atom_index:
                        adjacent_rid = r2_id
                        adjacent_atom_index = i2
                        break
                    elif r2_id == residue_id and i2 == atom_index:
                        adjacent_rid = r1_id
                        adjacent_atom_index = i1
                        break
                
                if adjacent_rid is not None:
                    adjacent_mol = monomers[adjacent_rid].rdkit_mol
                    bond_use_count[(r1_id, r2_id)] += 1
                
                padded_mol, mapidx = padders[link_label](
                    padded_mol, adjacent_mol, atom_index, adjacent_atom_index
                )

                tmp = {}
                for i, j in enumerate(mapidx):
                    if j is None:
                        continue  # new padding atom
                    if j not in mapidx_pad:
                        continue  # padding atom from previous iteration for another link_label
                    tmp[i] = mapidx_pad[j]
                mapidx_pad = tmp

            # update position of hydrogens bonded to link atoms
            inv = {j: i for (i, j) in mapidx_pad.items()}
            padded_idxs_to_update = []
            no_pad_idxs_to_update = []
            for atom_index in monomer.link_labels:
                heavy_atom = monomer.rdkit_mol.GetAtomWithIdx(atom_index)
                for neighbor in heavy_atom.GetNeighbors():
                    if neighbor.GetAtomicNum() != 1:
                        continue
                    if neighbor.GetIdx() in monomer.mapidx_to_raw:
                        # index of H exists in mapidx_to_raw, which means that
                        # the raw_input_mol had the hydrogen. Thus, we do not
                        # want to update its coordiantes.
                        continue
                    no_pad_idxs_to_update.append(neighbor.GetIdx())
                    padded_idxs_to_update.append(inv[neighbor.GetIdx()])
            update_H_positions(padded_mol, padded_idxs_to_update)
            source = padded_mol.GetConformer()
            destination = monomer.rdkit_mol.GetConformer()
            for i, j in zip(no_pad_idxs_to_update, padded_idxs_to_update):
                destination.SetAtomPosition(i, source.GetAtomPosition(j))
                # can invert chirality in 3D positions

            padded_mols[residue_id] = (padded_mol, mapidx_pad)
                

        # verify that all bonds resulted in padding
        err_msg = ""
        for key, count in bond_use_count.items():
            if count != 2:
                err_msg += (
                    f"expected two paddings for {key} {bonds[key]}, padded {count}"
                    + eol
                )
        if len(err_msg):
            raise RuntimeError(err_msg)
        return padded_mols

    def flexibilize_sidechain(self, residue_id, mk_prep):
        """

        Parameters
        ----------
        residue_id
        mk_prep

        Returns
        -------

        """
        monomer = self.monomers[residue_id]
        inv = {j: i for i, j in monomer.molsetup_mapidx.items()}
        link_atoms = [inv[i] for i in monomer.template.link_labels]
        if len(link_atoms) == 0:
            raise RuntimeError(
                "can't define a sidechain without bonds to other residues"
            )
        # TODO: rewrite this to work better with new MoleculeSetups
        graph = {atom.index: atom.graph for atom in monomer.molsetup.atoms}
        for i in range(len(link_atoms) - 1):
            start_node = link_atoms[i]
            end_nodes = [k for (j, k) in enumerate(link_atoms) if j != i]
            backbone_paths = find_graph_paths(graph, start_node, end_nodes)
            for path in backbone_paths:
                for x in range(len(path) - 1):
                    idx1 = min(path[x], path[x + 1])
                    idx2 = max(path[x], path[x + 1])
                    monomer.molsetup.bond_info[(idx1, idx2)].rotatable = False
        monomer.is_movable = True

        mk_prep.calc_flex(
            monomer.molsetup,
            root_atom_index=link_atoms[0],
        )

        molsetup = monomer.molsetup
        is_rigid_atom = [False for _ in molsetup.atoms]
        graph = molsetup.flexibility_model["rigid_body_graph"]
        root_body_idx = molsetup.flexibility_model["root"]
        conn = molsetup.flexibility_model["rigid_body_connectivity"]
        rigid_index_by_atom = molsetup.flexibility_model["rigid_index_by_atom"]
        # from the root, use only the atom that is bonded to the only rotatable bond
        for other_body_idx in graph[root_body_idx]:
            root_link_atom_idx = conn[(root_body_idx, other_body_idx)][0]
            for atom_idx, body_idx in rigid_index_by_atom.items():
                if body_idx != root_body_idx or atom_idx == root_link_atom_idx:
                    monomer.is_flexres_atom[atom_idx] = True
        return

    @staticmethod
    def _pdb_to_residue_mols(
        pdb_string,
        wanted_altloc: Optional[dict[str, str]]=None,
        default_altloc: Optional[str]=None,
    ):
        """

        Parameters
        ----------
        pdb_string

        Returns
        -------

        """
        blocks_by_residue = {}
        reskey_to_resname = {}
        reskey = None
        buffered_reskey = None
        buffered_resname = None
        # residues in non-consecutive lines due to TER or another res
        interrupted_residues = set()
        pdb_block = []

        def _add_if_new(to_dict, key, value, repeat_log):
            if key in to_dict:
                repeat_log.add(key)
            else:
                to_dict[key] = value
            return

        for line in pdb_string.splitlines(True):
            if line.startswith("TER") and reskey is not None:
                _add_if_new(blocks_by_residue, reskey, pdb_block, interrupted_residues)
                blocks_by_residue[reskey] = pdb_block
                pdb_block = []
                reskey = None
                buffered_reskey = None
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atomname = line[12:16].strip()
                altloc = line[16:17].strip()
                resname = line[17:20].strip()
                chainid = line[21:22].strip()
                resnum = int(line[22:26].strip())
                icode = line[26:27].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                element = line[76:78].strip() or atomname[0] # charmm workaround
                reskey = f"{chainid}:{resnum}{icode}"  # e.g. ":42", "A:42B"
                reskey_to_resname.setdefault(reskey, set())
                reskey_to_resname[reskey].add(resname)
                atom = AtomField(
                    atomname, altloc, resname, chainid,
                    resnum, icode, x, y, z, element,
                )

                if reskey == buffered_reskey:  # this line continues existing residue
                    pdb_block.append(atom)
                else:
                    if buffered_reskey is not None:
                        _add_if_new(
                            blocks_by_residue,
                            buffered_reskey,
                            pdb_block,
                            interrupted_residues,
                        )
                    buffered_reskey = reskey
                    pdb_block = [atom]

        if pdb_block:  # there was not a TER line
            _add_if_new(blocks_by_residue, reskey, pdb_block, interrupted_residues)

        if interrupted_residues:
            msg = f"interrupted residues in PDB: {interrupted_residues}"
            raise ValueError(msg)

        # verify that each identifier (e.g. "A:17" has a single resname
        violations = {k: v for k, v in reskey_to_resname.items() if len(v) != 1}
        if len(violations):
            msg = "each residue key must have exactly 1 resname" + eol
            msg += f"but got {violations=}"
            raise ValueError(msg)

        if wanted_altloc is None:
            wanted_altloc = {}
        raw_input_mols = {}
        for reskey, atom_field_list in blocks_by_residue.items():
            requested_altloc = wanted_altloc.get(reskey, None)
            pdbmol, _, missed_altloc, needed_altloc = _aux_altloc_mol_build(
                atom_field_list,
                requested_altloc,
                default_altloc,
            )
            resname = list(reskey_to_resname[reskey])[0]  # verified length 1
            raw_input_mols[reskey] = (pdbmol, resname, missed_altloc, needed_altloc)

        return raw_input_mols


    @staticmethod
    def _prody_to_residue_mols(
            prody_obj: ALLOWED_PRODY_TYPES,
            wanted_altloc_dict: Optional[dict] = None,
            default_altloc: Optional[str] = None,
        ) -> dict:
        """

        Parameters
        ----------
        prody_obj

        Returns
        -------

        """

        if wanted_altloc_dict is None:
            wanted_altloc_dict = {}
        raw_input_mols = {}
        reskey_to_resname = {}
        # generate macromolecule hierarchy iterator
        hierarchy = prody_obj.getHierView()
        # iterate chains
        for chain in hierarchy.iterChains():
            # iterate residues
            for res in chain.iterResidues():
                # gather residue info
                chain_id = str(res.getChid()).strip()
                res_name = str(res.getResname()).strip()
                res_num = int(res.getResnum())
                icode = str(res.getIcode()).strip()
                reskey = f"{chain_id}:{res_num}{icode}"
                reskey_to_resname.setdefault(reskey, set())
                reskey_to_resname[reskey].add(res_name)
                requested_altloc = wanted_altloc_dict.get(reskey, None)
                # we are not sanitizing because protonated LYS don't have the
                # formal charge set on the N and Chem.SanitizeMol raises error
                # Chem.SanitizeMol(prody_mol)
                prody_mol, missed_altloc, needed_altloc = prody_to_rdkit(
                    res,
                    sanitize=False,
                    requested_altloc=requested_altloc,
                    default_altloc=default_altloc,
                )
                raw_input_mols[reskey] = (prody_mol, res_name,
                                          missed_altloc, needed_altloc)
        return raw_input_mols



    def to_pdb(self, new_positions: Optional[dict]=None):
        """
        Parameters
        ----------
        new_positions: dict (str -> dict (int -> (float, float, float)))
                             |            |      |
                    residue_id            |      |
                                 atom_index      |
                                                 new_position
        Returns
        _______
        pdb_string: str
        """    

        if new_positions is None:
            new_positions = {}
        valid_monomers = self.get_valid_monomers()

        # check that residue IDs passed in new_positions are valid
        unknown_res_ids = set()
        for res_id in new_positions:
            if res_id not in valid_monomers:
                unknown_res_ids.add(res_id)
        if unknown_res_ids:
            msg = f"Residue IDs not in valid monomers: {unknown_res_ids}"
            raise ValueError(msg)

        pdbout = ""
        atom_count = 0
        pdb_line = "{:6s}{:5d} {:^4s} {:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}                       {:2s} "
        pdb_line += eol
        for res_id in self.get_valid_monomers():
            rdkit_mol = self.monomers[res_id].rdkit_mol
            if res_id in new_positions:
                positions = get_updated_positions(
                    self.monomers[res_id],
                    new_positions[res_id],
                )
            else:
                positions = rdkit_mol.GetConformer().GetPositions()

            chain, resnum = res_id.split(":")
            if resnum[-1].isalpha():
                icode = resnum[-1]
                resnum = resnum[:-1]
            else:
                icode = ""
            resnum = int(resnum)

            for i, atom in enumerate(rdkit_mol.GetAtoms()):
                atom_count += 1
                props = atom.GetPropsAsDict()
                atom_name = self.monomers[res_id].atom_names[i]
                x, y, z = positions[i]
                element = mini_periodic_table[atom.GetAtomicNum()]
                pdbout += pdb_line.format(
                    "ATOM",
                    atom_count,
                    atom_name,
                    self.monomers[res_id].input_resname,
                    chain,
                    resnum,
                    icode,
                    x,
                    y,
                    z,
                    element,
                )
        return pdbout

    def export_static_atom_params(self):
        """

        Returns
        -------
        atom_params: dict
        coords: list
        """
        atom_params = {}
        counter_atoms = 0
        coords = []
        dedicated_attribute = (
            "charge",
            "atom_type",
        )  # molsetup has a dedicated attribute
        for res_id in self.get_valid_monomers():
            molsetup = self.monomers[res_id].molsetup
            wanted_atom_indices = []
            for atom in molsetup.atoms:
                if not atom.is_ignore and not self.monomers[res_id].is_flexres_atom[atom.index]:
                    wanted_atom_indices.append(atom.index)
                    coords.append(molsetup.get_coord(atom.index))
            for key, values in molsetup.atom_params.items():
                atom_params.setdefault(key, [None] * counter_atoms)  # add new "column"
                for i in wanted_atom_indices:
                    atom_params[key].append(values[i])
            # This was reworked to specifically address the new MoleculeSetup structure. Needs re-thinking
            charge_dict = {atom.index: atom.charge for atom in molsetup.atoms}
            atom_type_dict = {atom.index: atom.atom_type for atom in molsetup.atoms}
            for key in dedicated_attribute:
                atom_params.setdefault(key, [None] * counter_atoms)  # add new "column"
                if key == "charge":
                    values_dict = charge_dict
                else:
                    values_dict = atom_type_dict
                for i in wanted_atom_indices:
                    atom_params[key].append(values_dict[i])
            counter_atoms += len(wanted_atom_indices)
            added_keys = set(molsetup.atom_params).union(dedicated_attribute)
            for key in set(atom_params).difference(
                added_keys
            ):  # <key> missing in current molsetup
                atom_params[key].extend(
                    [None] * len(wanted_atom_indices)
                )  # fill in incomplete "row"
        if hasattr(self, "param_rename"):  # e.g. "gasteiger" -> "q"
            for key, new_key in self.param_rename.items():
                atom_params[new_key] = atom_params.pop(key)
        return atom_params, coords

    # region Filtering Residues
    def get_ignored_monomers(self):
        return {k: v for k, v in self.monomers.items() if v.rdkit_mol is None}

    def get_valid_monomers(self):
        return {k: v for k, v in self.monomers.items() if v.rdkit_mol is not None}

    # endregion


def add_rotamers_to_polymer_molsetups(rotamer_states_list, polymer):
    """

    Parameters
    ----------
    rotamer_states_list
    polymer

    Returns
    -------

    """
    rotamer_res_disambiguate = {}
    for (
        primary_res,
        specific_res_list,
    ) in polymer.residue_chem_templates.ambiguous.items():
        for specific_res in specific_res_list:
            rotamer_res_disambiguate[specific_res] = primary_res

    no_resname_to_resname = {}
    for res_with_resname in polymer.monomers:
        chain, resname, resnum = res_with_resname.split(":")
        no_resname_key = f"{chain}:{resnum}"
        if no_resname_key in no_resname_to_resname:
            errmsg = "both %s and %s would be keyed by %s" % (
                res_with_resname,
                no_resname_to_resname[no_resname_key],
                no_resname_key,
            )
            raise RuntimeError(errmsg)
        no_resname_to_resname[no_resname_key] = res_with_resname

    state_indices_list = []
    for state_index, state_dict in enumerate(rotamer_states_list):
        print(f"adding rotamer state {state_index + 1}")
        state_indices = {}
        for res_no_resname, angles in state_dict.items():
            res_with_resname = no_resname_to_resname[res_no_resname]
            if polymer.monomers[res_with_resname].molsetup is None:
                raise RuntimeError(
                    "no molsetup for %s, can't add rotamers" % (res_with_resname)
                )
            # next block is inefficient for large rotamer_states_list
            # refactored polymers could help by having the following
            # data readily available
            molsetup = polymer.monomers[res_with_resname].molsetup
            name_to_molsetup_idx = {}
            for atom in molsetup.atoms:
                atom_name = atom.pdbinfo.name
                name_to_molsetup_idx[atom_name] = atom.index

            resname = res_with_resname.split(":")[1]
            resname = rotamer_res_disambiguate.get(resname, resname)

            atom_names = residues_rotamers[resname]
            if len(atom_names) != len(angles):
                raise RuntimeError(
                    f"expected {len(atom_names)} angles for {resname}, got {len(angles)}"
                )

            atom_idxs = []
            for names in atom_names:
                tmp = [name_to_molsetup_idx[name] for name in names]
                atom_idxs.append(tmp)

            state_indices[res_with_resname] = len(molsetup.rotamers)
            molsetup.add_rotamer(atom_idxs, np.radians(angles))

        state_indices_list.append(state_indices)

    return state_indices_list
