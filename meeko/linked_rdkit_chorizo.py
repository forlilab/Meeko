import pathlib
import json
from os import linesep as os_linesep
from typing import Union
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdChemReactions
from rdkit.Chem import rdMolInterchange
from rdkit.Chem.AllChem import EmbedMolecule, AssignBondOrdersFromTemplate
import prody
from prody.atomic.atomgroup import AtomGroup
from prody.atomic.selection import Selection

from .writer import PDBQTWriterLegacy
from .molsetup import MoleculeSetup
from .molsetup import MoleculeSetupEncoder
from .utils.rdkitutils import mini_periodic_table
from .utils.rdkitutils import react_and_map
from .utils.prodyutils import prody_to_rdkit, ALLOWED_PRODY_TYPES
from .utils.pdbutils import PDBAtomInfo

import numpy as np


def find_inter_mols_bonds(mols):
    covalent_radius = {  # from wikipedia
        1: 0.31, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 12: 1.41, 14: 1.11,
        15: 1.07, 16: 1.05, 17: 1.02, 19: 2.03, 20: 1.76, 24: 1.39, 26: 1.32, 30: 1.22,
        34: 1.20, 35: 1.20, 53: 1.39,
    }
    allowance = 1.2  # vina uses 1.1 but covalent radii are shorter here
    max_possible_covalent_radius = 2 * allowance * max([r for k, r in covalent_radius.items()])
    cubes_min = []
    cubes_max = []
    for mol in mols:
        positions = mol.GetConformer().GetPositions()
        cubes_min.append(np.min(positions, axis=0))
        cubes_max.append(np.max(positions, axis=0))
    tmp = np.array([0, 0, 1, 1])
    pairs_to_consider = []
    for i in range(len(mols)):
        for j in range(i + 1, len(mols)):
            do_consider = True
            for d in range(3):
                x = (cubes_min[i][d], cubes_max[i][d], cubes_min[j][d], cubes_max[j][d])
                idx = np.argsort(x)
                has_overlap = tmp[idx][0] != tmp[idx][1]
                close_enough = abs(x[idx[1]] - x[idx[2]]) < max_possible_covalent_radius
                do_consider &= (close_enough or has_overlap)
            if do_consider:
                pairs_to_consider.append((i, j))
    bonds = []
    for i, j in pairs_to_consider:
        p1 = mols[i].GetConformer().GetPositions()
        p2 = mols[j].GetConformer().GetPositions()
        for a1 in mols[i].GetAtoms():
            for a2 in mols[j].GetAtoms():
                vec = p1[a1.GetIdx()] - p2[a2.GetIdx()]
                distsqr = (np.dot(vec, vec))
                cov_dist = covalent_radius[a1.GetAtomicNum()] + covalent_radius[a2.GetAtomicNum()]
                if distsqr < (allowance * cov_dist) ** 2:
                    bonds.append((i, j, a1.GetIdx(), a2.GetIdx()))

    return len(pairs_to_consider), bonds


def mapping_by_mcs(mol, ref):
    mcs_result = rdFMCS.FindMCS([mol, ref], bondCompare=rdFMCS.BondCompare.CompareAny)
    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)

    mol_idxs = mol.GetSubstructMatch(mcs_mol)
    ref_idxs = ref.GetSubstructMatch(mcs_mol)

    atom_map = {i: j for (i, j) in zip(mol_idxs, ref_idxs)}
    return atom_map


def _snap_to_int(value, tolerance=0.12):
    for inc in [-1, 0, 1]:
        if abs(value - int(value) - inc) <= tolerance:
            return int(value) + inc
    return None


def divide_int_gracefully(integer, weights, allow_equal_weights_to_differ=False):
    for weight in weights:
        if type(weight) not in [int, float] or weight < 0:
            raise ValueError("weights must be numeric and non-negative")
    if type(integer) != int:
        raise ValueError("integer must be integer")
    inv_total_weight = 1.0 / sum(weights)
    shares = [w * inv_total_weight for w in weights]  # normalize
    result = [_snap_to_int(integer * s, tolerance=0.5) for s in shares]
    surplus = integer - sum(result)
    if surplus == 0:
        return result
    data = [(i, w) for (i, w) in enumerate(weights)]
    data = sorted(data, key=lambda x: x[1], reverse=True)
    idxs = [i for (i, _) in data]
    if allow_equal_weights_to_differ:
        groups = [1 for _ in weights]
    else:
        groups = []
        last_weight = None
        for i in idxs:
            if weights[i] == last_weight:
                groups[-1] += 1
            else:
                groups.append(1)
            last_weight = weights[i]

    # iterate over all possible combinations of groups
    # this is potentially very slow
    nr_groups = len(groups)
    for j in range(1, 2 ** nr_groups):
        n_changes = 0
        combo = []
        for grpidx in range(nr_groups):
            is_changed = bool(j & 2 ** grpidx)
            combo.append(is_changed)
            n_changes += is_changed * groups[grpidx]
        if n_changes == abs(surplus):
            break

    # add or subtract 1 to distribute surplus
    increment = surplus / abs(surplus)
    index = 0
    for i, is_changed in enumerate(combo):
        if is_changed:
            for j in range(groups[i]):
                result[idxs[index]] += increment
                index += 1

    return result


def rectify_charges(q_list, net_charge=None, decimals=3):
    """make them 3 decimals and sum to an integer"""

    fstr = "%%.%df" % decimals
    charges_dec = [float(fstr % q) for q in q_list]

    if net_charge is None:
        net_charge = _snap_to_int(sum(charges_dec), tolerance=0.15)
        if net_charge is None:
            msg = "net charge could not be predicted from input q_list. (residual is beyond tolerance) "
            msg = "Please set the net_charge argument directly"
            raise RuntimeError(msg)
    elif type(net_charge) != int:
        raise TypeError("net charge must be an integer")

    surplus = net_charge - sum(charges_dec)
    surplus_int = _snap_to_int(10 ** decimals * surplus)

    if surplus_int == 0:
        return charges_dec

    weights = [abs(q) for q in q_list]
    surplus_int_splits = divide_int_gracefully(surplus_int, weights)
    for i, increment in enumerate(surplus_int_splits):
        charges_dec[i] += 10 ** -decimals * increment

    return charges_dec

def update_H_positions(mol, indices_to_update):
    """re-calculate positions of some existing hydrogens

    no guarantees that chirality can be preserved

    Parameters
    ----------
    mol: RDKitmol
        molecule with hydrogens
    indices_to_update: list
        indices of hydrogens for which positions will be re-calculated
    """

    conf = mol.GetConformer()
    tmpmol = Chem.RWMol(mol)
    to_del = {}
    to_add_h = []
    for h_index in indices_to_update:
        atom = tmpmol.GetAtomWithIdx(h_index)
        if atom.GetAtomicNum() != 1:
            raise RuntimeError("only H positions can be updated")
        heavy_neighbors = []
        for neigh_atom in atom.GetNeighbors():
            if neigh_atom.GetAtomicNum() != 1:
                heavy_neighbors.append(neigh_atom)
        if len(heavy_neighbors) != 1:
            raise RuntimeError(f"hydrogens must have 1 non-H neighbor, got {len(heavy_neighbors)}")
        to_add_h.append(heavy_neighbors[0])
        to_del[h_index] = heavy_neighbors[0]
    for i in sorted(to_del, reverse=True):
        tmpmol.RemoveAtom(i)
        to_del[i].SetNumExplicitHs(to_del[i].GetNumExplicitHs() + 1)
    to_add_h = list(
        set([atom.GetIdx() for atom in to_add_h]))  # atom.GetIdx() returns new index after deleting Hs
    tmpmol = tmpmol.GetMol()
    tmpmol.UpdatePropertyCache()
    # for atom in tmpmol.GetAtoms():
    #    print(atom.GetAtomicNum(), atom.GetNumImplicitHs(), atom.GetNumExplicitHs())
    Chem.SanitizeMol(tmpmol)
    tmpmol = Chem.AddHs(tmpmol, onlyOnAtoms=to_add_h, addCoords=True)
    # tmpmol = Chem.AddHs(tmpmol, addCoords=True)
    # print(tmpmol.GetNumAtoms())
    tmpconf = tmpmol.GetConformer()
    # print(tmpconf.GetPositions())
    used_h = set()  # heavy atom may have multiple H that were missing, keep track of Hs that were visited
    for h_index, parent in to_del.items():
        for atom in tmpmol.GetAtomWithIdx(parent.GetIdx()).GetNeighbors():
            # print(atom.GetAtomicNum(), atom.GetIdx(), len(mapping), tmpmol.GetNumAtoms())
            has_new_position = atom.GetIdx() >= mol.GetNumAtoms() - len(to_del)
            if atom.GetAtomicNum() == 1 and has_new_position:
                if atom.GetIdx() not in used_h:
                    # print(h_index, tuple(tmpconf.GetAtomPosition(atom.GetIdx())))
                    conf.SetAtomPosition(h_index, tmpconf.GetAtomPosition(atom.GetIdx()))
                    used_h.add(atom.GetIdx())
                    break  # h_index coords copied, don't look into further H bound to parent
                    # no guarantees about preserving chirality, which we don't need

    if len(used_h) != len(to_del):
        raise RuntimeError(f"Updated {len(used_h)} H positions but deleted {len(to_del)}")

    return



class ResidueChemTemplates:

    """Holds template data required to initialize LinkedRDKitChorizo

    Attributes
    ----------
    residue_templates: dict (string -> ResidueTemplate)
        keys are the ID of an instance of ResidueTemplate
    padders: dict
        instances of ResiduePadder keyed by a link_label (a string)
        link_labels establish the relationship between ResidueTemplates
        and ResiduePadders, determining which padder is to be used to
        pad each atom of an instance of ChorizoResidue that needs padding.
    ambiguous: dict
        mapping between input residue names (e.g. the three-letter residue
        name from PDB files) and IDs (strings) of ResidueTemplates
    """

    def __init__(self, residue_templates, padders, ambiguous):
        self._check_missing_padders(residue_templates, padders)
        self._check_ambiguous_reskeys(residue_templates, ambiguous)
        self.residue_templates = residue_templates
        self.padders = padders
        self.ambiguous = ambiguous


    @classmethod
    def from_dict(cls, alldata):
        """
        constructs ResidueTemplates and ResiduePadders from a dictionary
        with raw data such as that in data/residue_chem_templates.json
        This is pretty much a JSON deserializer that takes a dictionary
        as input to allow users to modify the input dict in Python
        """

        # alldata = json.loads(json_string)
        ambiguous = alldata["ambiguous"]
        residue_templates = {}
        padders = {}
        for key, data in alldata["residue_templates"].items():
            link_labels = None
            if "link_labels" in data:
                link_labels = {int(key): value for key, value in data["link_labels"].items()}
            # print(key)
            res_template = ResidueTemplate(data["smiles"], link_labels, data.get("atom_name", None))
            residue_templates[key] = res_template
        for link_label, data in alldata["padders"].items():
            rxn_smarts = data["rxn_smarts"]
            adjacent_res_smarts = data.get("adjacent_res_smarts", None)
            padders[link_label] = ResiduePadder(rxn_smarts, adjacent_res_smarts)
        return cls(residue_templates, padders, ambiguous)

    @staticmethod
    def _check_missing_padders(residues, padders):

        # can't guarantee full coverage because the topology that is passed
        # to the chorizo may contain bonds between residues that are not
        # anticipated to be bonded, for example, protein N-term bonded to
        # nucleic acid 5 prime.

        # collect labels from residues
        link_labels_in_residues = set()
        for reskey, res_template in residues.items():
            for _, link_label in res_template.link_labels.items():
                link_labels_in_residues.add(link_label)

        # and check we have padders for all of them
        link_labels_in_padders = set([label for label in padders])
        # for link_label in padders:
        #    for (link_labels) in padder.link_labels:
        #        print(link_key, link_labels)
        #        for (label, _) in link_labels: # link_labels is a list of pairs
        #            link_labels_in_padders.add(label)

        missing = link_labels_in_residues.difference(link_labels_in_padders)
        if missing:
            raise RuntimeError(f"missing padders for {missing}")

        return

    @staticmethod
    def _check_ambiguous_reskeys(residue_templates, ambiguous):
        missing = {}
        for input_resname, reskeys in ambiguous.items():
            for reskey in reskeys:
                if reskey not in residue_templates:
                    missing.setdefault(input_resname, set())
                    missing[input_resname].add(reskey)
        if len(missing):
            raise ValueError(f"missing residue templates for ambiguous: {missing}")
        return


class LinkedRDKitChorizo:
    """Represents polymer with its subunits as individual RDKit molecules.

    Used for proteins and nucleic acids. The key class is ChorizoResidue,
    which contains, a padded RDKit molecule containing part of the adjacent
    residues to enable chemically meaningful parameterizaion.
    Instances of ResidueTemplate make sure that the input, which may originate
    from a PDB string, matches the RDKit molecule of the template, even if
    hydrogens are missing.

    Attributes
    ----------
    residues: dict (string -> ChorizoResidue) #TODO: figure out exact SciPy standard for dictionary key/value notation
    """

    @classmethod
    def from_pdb_string(cls, pdb_string, chem_templates, mk_prep,
                        set_template=None, residues_to_delete=None, allow_bad_res=False):

        raw_input_mols = cls._pdb_to_residue_mols(pdb_string)
        chorizo = cls(raw_input_mols, chem_templates, mk_prep,
                      set_template, residues_to_delete, allow_bad_res)
        return chorizo

    @classmethod
    def from_prody(
        cls,
        prody_obj: Union[Selection, AtomGroup],
        chem_templates,
        mk_prep,
        set_template=None,
        residues_to_delete=None,
        allow_bad_res=False,
    ):
        raw_input_mols = cls._prody_to_residue_mols(prody_obj)
        chorizo = cls(
            raw_input_mols,
            chem_templates,
            mk_prep,
            set_template,
            residues_to_delete,
            allow_bad_res,
        )
        return chorizo

    def __init__(self, raw_input_mols, residue_chem_templates, mk_prep=None,
                 set_template=None, residues_to_delete=None,
                 allow_bad_res=False):
        """
        Parameters
        ----------
        raw_input_mols: dict (string -> RDKit Mol)
            keys are residue IDs <chain>:<resnum> such as "A:42"
            values are RDKit Mols that will be matched to instances of ResidueTemplate,
            and may contain none, all, or some of the hydrogens.
        residue_chem_templates: ResidueChemTemplates
            one instance of it
        mk_prep: MoleculePreparation
            to parameterize the padded molecules
        set_template: dict (string -> string)
            keys are residue IDs <chain>:<resnum> such as "A:42"
            values identify ResidueTemplate instances
        residues_to_delete: list (string)
            list of residue IDs (e.g.; "A:42") to mark as ignored
        allow_bad_res: bool
            mark unmatched residues as ignored instead of raising error
        """

        # TODO allow_bad_res and residues to delete should exist only in wrapper class methods
        # such as cls.from_pdb_string and future cls.from_openmm, cls.from_prody, etc

        # TODO move bonds calculation outta here to from_pdb_string and take bonds as arg here

        # TODO integrate bonds with matches to distinguish CYX vs CYX-
        # and simplify SMARTS for adjacent res in padders

        self.residue_chem_templates = residue_chem_templates
        residue_templates = residue_chem_templates.residue_templates
        padders = residue_chem_templates.padders
        ambiguous = residue_chem_templates.ambiguous

        # make sure all resiude_id in set_template exist
        if set_template is not None:
            missing = set([residue_id for residue_id in set_template if residue_id not in raw_input_mols])
            if len(missing):
                raise ValueError(f"Residue IDs in set_template not found: {missing}")

        self.residues, self.log = self._get_residues(raw_input_mols, ambiguous, residue_templates, set_template)

        print(self.log)  # TODO integrate with mk_prepare_receptor.py (former suggested_mutations)

        if residues_to_delete is None:
            residues_to_delete = ()  # self._delete_residues expects an iterator
        self._delete_residues(residues_to_delete, self.residues)  # sets ChorizoResidue.user_deleted = True

        # currently limied to max 1 bonde beteen each pair of residues
        # TODO use raw mol indices rather than rdkit_mol indices to allow external topology
        # TODO move _get_bonds call to cls.from_pdb_string and pass bonds as argument here
        bonds = self._get_bonds(self.residues)

        # padding may seem overkill but we had to run a reaction anyway for h_coord_from_dipep
        padded_mols = self._build_padded_mols(self.residues, bonds, padders)
        for residue_id, (padded_mol, mapidx_from_pad) in padded_mols.items():
            residue = self.residues[residue_id]
            residue.padded_mol = padded_mol
            residue.molsetup_mapidx = mapidx_from_pad

        if mk_prep is not None:
            self.parameterize(mk_prep)

        return


    def parameterize(self, mk_prep):

        for residue_id in self.get_valid_residues():
            residue = self.residues[residue_id]
            molsetups = mk_prep(residue.padded_mol)
            if len(molsetups) != 1:
                raise NotImplementedError(f"need 1 molsetup but got {len(molsetups)}")
            molsetup = molsetups[0]
            self.residues[residue_id].molsetup = molsetup
            self.residues[residue_id].is_flexres_atom = [False for _ in molsetup.atom_ignore]

            # set ignore to True for atoms that are padding
            for index in range(len(molsetup.atom_ignore)):
                if index not in residue.molsetup_mapidx:
                    molsetup.atom_ignore[index] = True

            # rectify charges to sum to integer (because of padding)
            if mk_prep.charge_model == "zero":
                net_charge = 0
            else:
                rdkit_mol = self.residues[residue_id].rdkit_mol
                net_charge = sum([atom.GetFormalCharge() for atom in rdkit_mol.GetAtoms()])
            not_ignored_idxs = []
            charges = []
            for i, q in molsetup.charge.items():  # charge is ordered dict
                if i in residue.molsetup_mapidx:  # TODO offsite not in mapidx
                    charges.append(q)
                    not_ignored_idxs.append(i)
            charges = rectify_charges(charges, net_charge, decimals=3)
            chain, resnum = residue_id.split(":")
            resname = self.residues[residue_id].input_resname
            if self.residues[residue_id].atom_names is None:
                atom_names = ["" for _ in not_ignored_idxs]
            else:
                atom_names = self.residues[residue_id].atom_names
            for i, j in enumerate(not_ignored_idxs):
                molsetup.charge[j] = charges[i]
                atom_name = atom_names[residue.molsetup_mapidx[j]]
                molsetup.pdbinfo[j] = PDBAtomInfo(atom_name, resname, int(resnum), chain)
        return

    @staticmethod
    def _build_rdkit_mol(raw_mol, template, mapping, nr_missing_H):

        rdkit_mol = Chem.Mol(template.mol)  # making a copy
        conf = Chem.Conformer(rdkit_mol.GetNumAtoms())
        input_conf = raw_mol.GetConformer()
        for i, j in mapping.items():
            conf.SetAtomPosition(i, input_conf.GetAtomPosition(j))

        rdkit_mol.AddConformer(conf, assignId=True)

        if nr_missing_H:  # add positions to Hs missing in raw_mol
            if rdkit_mol.GetNumAtoms() != len(mapping) + nr_missing_H:
                raise RuntimeError(
                        f"nr of atoms ({rdkit_mol.GetNumAtoms()}) != "\
                        f"{len(mapping)=} + {nr_missing_H=}")
            idxs = [i for i in range(rdkit_mol.GetNumAtoms()) if i not in mapping]
            update_H_positions(rdkit_mol, idxs)

        return rdkit_mol


    @staticmethod
    def _find_least_missing_Hs(raw_input_mol, residue_templates):
        min_missing_Hs = float("+inf")
        best = []
        for index, template in enumerate(residue_templates):
            match_stats, mapping = template.match(raw_input_mol)
            if match_stats["heavy"]["missing"] or match_stats["heavy"]["excess"]:
                continue
            if match_stats["H"]["excess"]:
                continue
            if match_stats["H"]["missing"] == min_missing_Hs:
                best.append({"index": index, "match_stats": match_stats, "mapping": mapping})
            elif match_stats["H"]["missing"] < min_missing_Hs:
                best = [{"index": index, "match_stats": match_stats, "mapping": mapping}]
                min_missing_Hs = match_stats["H"]["missing"]
        return best

    @classmethod
    def _get_residues(cls, raw_input_mols, ambiguous, residue_templates, set_template):

        residues = {}
        log = {
            "chosen_by_fewest_missing_H": {},
            "chosen_by_default": {},
            "no_match": [],
            "no_mol": [],
        }
        for residue_key, (raw_mol, input_resname) in raw_input_mols.items():
            if raw_mol is None:
                residues[residue_key] = ChorizoResidue(None, None, None, input_resname, None)
                log["no_mol"].append(residue_key)
                continue
            raw_mol_has_H = sum([a.GetAtomicNum() == 1 for a in raw_mol.GetAtoms()]) > 0
            tolerate_excess_H = False
            if set_template is not None and residue_key in set_template:
                template_key = set_template[residue_key]  # often resname or resname-like, e.g. HID, NALA
                template = residue_templates[template_key]
                match_stats, mapping = template.match(raw_mol)
                tolerate_excess_H = True  # allows setting LYN from protonated (LYS+) input
            elif raw_mol_has_H and input_resname in ambiguous and len(ambiguous[input_resname]) > 1:
                candidate_template_keys = ambiguous[input_resname]
                candidate_templates = [residue_templates[key] for key in candidate_template_keys]
                best_matches = cls._find_least_missing_Hs(raw_mol, candidate_templates)
                if len(best_matches) > 1:
                    raise RuntimeError(
                        "{len(output)} templates have fewest missing Hs to {residue_key} please change templates or input to avoid ties")
                elif len(best_matches) == 0:
                    template_key = None
                    template = None
                else:
                    match_stats = best_matches[0]["match_stats"]
                    mapping = best_matches[0]["mapping"]
                    index = best_matches[0]["index"]
                    template_key = candidate_template_keys[index]
                    template = residue_templates[template_key]
                    log["chosen_by_fewest_missing_H"][residue_key] = template_key
            elif input_resname in ambiguous:  # use default (first) template in ambiguous
                template_key = ambiguous[input_resname][0]
                template = residue_templates[template_key]
                match_stats, mapping = template.match(raw_mol)
                log["chosen_by_default"][residue_key] = template_key
            else:
                template_key = input_resname
                template = residue_templates[template_key]
                match_stats, mapping = template.match(raw_mol)

            if template_key is None or match_stats["heavy"]["missing"] or match_stats["heavy"]["excess"]:
                residues[residue_key] = ChorizoResidue(raw_mol, None, None, input_resname, template_key)
                log["no_match"].append(residue_key)
                if residue_key in log["chosen_by_default"]:
                    log["chosen_by_default"].pop(residue_key)
            else:
                rdkit_mol = cls._build_rdkit_mol(raw_mol, template, mapping, match_stats["H"]["missing"])
                residues[residue_key] = ChorizoResidue(raw_mol, rdkit_mol, mapping, input_resname, template_key, template.atom_names)
                if template.link_labels is not None:
                    mapping_inv = residues[residue_key].mapidx_from_raw  # {j: i for (i, j) in mapping.items()}
                    link_labels = {i: label for i, label in template.link_labels.items()}
                    residues[residue_key].link_labels = link_labels

                # TODO link atoms connections add to chorizoResidue
        return residues, log

    @staticmethod
    def _get_bonds(residues):
        mols = [residue.raw_rdkit_mol for key, residue in residues.items() if residue.rdkit_mol is not None]
        keys = [key for key, residue in residues.items() if residue.rdkit_mol is not None]
        # t0 = time()
        ### print("Starting find inter mols bonds", f"{len(mols)=}")
        ### print(f"maximum nr of pairs to consider: {len(mols) * (len(mols)-1)}")
        nr_pairs, bonds_ = find_inter_mols_bonds(mols)
        bonds = {}
        for (i, j, k, l) in bonds_:
            bonds[(keys[i], keys[j])] = (  # k, l)
                residues[keys[i]].mapidx_from_raw[k],
                residues[keys[j]].mapidx_from_raw[l],
            )
        assert len(bonds) == len(bonds_), "can add only 1 bond between each pair of residues, for now"
        # t = time()
        # print(f"took {t-t0:.4f} seconds")
        return bonds

    @staticmethod
    def _build_padded_mols(residues, bonds, padders):

        padded_mols = {}
        bond_use_count = {key: 0 for key in bonds}
        for residue_id, residue, in residues.items():
            # print(residue_id, residue.rdkit_mol is not None)
            if residue.rdkit_mol is None:
                continue
            padded_mol = residue.rdkit_mol
            mapidx_pad = {atom.GetIdx(): atom.GetIdx() for atom in padded_mol.GetAtoms()}
            for atom_index, link_label in residue.link_labels.items():
                adjacent_mol = None
                adjacent_atom_index = None
                for (r1_id, r2_id), (i1, i2) in bonds.items():
                    if r1_id == residue_id and i1 == atom_index:
                        adjacent_mol = residues[r2_id].rdkit_mol
                        adjacent_atom_index = i2
                        bond_use_count[(r1_id, r2_id)] += 1
                        break
                    elif r2_id == residue_id and i2 == atom_index:
                        adjacent_mol = residues[r1_id].rdkit_mol
                        adjacent_atom_index = i1
                        bond_use_count[(r1_id, r2_id)] += 1
                        break

                # print(residue_id, atom_index, link_label, adjacent_mol is not None, adjacent_atom_index)
                padded_mol, mapidx = padders[link_label](padded_mol, adjacent_mol, atom_index, adjacent_atom_index)

                tmp = {}
                for i, j in enumerate(mapidx):
                    if j is None:
                        continue  # new padding atom
                    if j not in mapidx_pad:
                        continue  # padding atom from previous iteration for another link_label
                    tmp[i] = mapidx_pad[j]
                mapidx_pad = tmp
                # print(f"{mapidx_pad=}")

            # update position of hydrogens bonded to link atoms
            inv = {j: i for (i, j) in mapidx_pad.items()}
            padded_idxs_to_update = []
            no_pad_idxs_to_update = []
            for atom_index in residue.link_labels:
                heavy_atom = residue.rdkit_mol.GetAtomWithIdx(atom_index)
                for neighbor in heavy_atom.GetNeighbors():
                    if neighbor.GetAtomicNum() != 1:
                        continue
                    no_pad_idxs_to_update.append(neighbor.GetIdx())
                    padded_idxs_to_update.append(inv[neighbor.GetIdx()])
            update_H_positions(padded_mol, padded_idxs_to_update)
            source = padded_mol.GetConformer()
            destination = residue.rdkit_mol.GetConformer()
            for i, j in zip(no_pad_idxs_to_update, padded_idxs_to_update):
                destination.SetAtomPosition(i, source.GetAtomPosition(j))
                # can invert chirality in 3D positions

            padded_mols[residue_id] = (padded_mol, mapidx_pad)

        # verify that all bonds resulted in padding
        err_msg = ""
        for key, count in bond_use_count.items():
            if count != 2:
                err_msg += "expected two paddings for {key} {bonds[key]}, padded {count}" + os_linesep
        if len(err_msg):
            raise RuntimeError(err_msg)
        return padded_mols

    @staticmethod
    def _delete_residues(query_res, residues):
        missing = set()
        for res in query_res:
            if res not in residues:
                missing.add(res)
            elif residues[res]:  # is not None: # expecting None if templates didn't match
                residues[res].user_deleted = True
        if len(missing) > 0:
            msg = "can't find the following residues to delete: " + " ".join(missing)
            raise ValueError(msg)

    def flexibilize_protein_sidechain(self, res, mk_prep, cut_at_calpha=False):
        molsetup, mapidx, is_flexres_atom = self.res_to_molsetup(res, mk_prep,
                                                                 is_protein_sidechain=True,
                                                                 cut_at_calpha=cut_at_calpha)
        self.residues[res].molsetup = molsetup
        self.residues[res].molsetup_mapidx = mapidx
        self.residues[res].is_flexres_atom = is_flexres_atom
        self.residues[res].is_movable = True
        return

    @staticmethod
    def print_residues_by_resname(removed_residues):
        by_resname = dict()
        for res_id in removed_residues:
            chain, resn, resi = res_id.split(":")
            by_resname.setdefault(resn, [])
            by_resname[resn].append(f"{chain}:{resi}")
        string = ""
        for resname, removed_res in by_resname.items():
            string += f"Resname: {resname}:" + pathlib.os.linesep
            string += " ".join(removed_res) + pathlib.os.linesep
        return string

    @staticmethod
    def _pdb_to_residue_mols(pdb_string):
        blocks_by_residue = {}
        reskey_to_resname = {}
        reskey = None
        buffered_reskey = None
        buffered_resname = None
        interrupted_residues = set()  # e.g. non-consecutive residue lines due to interruption by TER or another res
        pdb_block = ""

        def _add_if_new(to_dict, key, value, repeat_log):
            if key in to_dict:
                repeat_log.add(key)
            else:
                to_dict[key] = value
            return

        for line in pdb_string.splitlines(True):
            if line.startswith('TER') and reskey is not None:
                _add_if_new(blocks_by_residue, reskey, pdb_block, interrupted_residues)
                blocks_by_residue[reskey] = pdb_block
                pdb_block = ""
                reskey = None
                buffered_reskey = None
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # Generating dictionary key
                resname = line[17:20].strip()
                resid = int(line[22:26].strip())
                chainid = line[21].strip()
                icode = line[26:27].strip()
                reskey = f"{chainid}:{resid}{icode}"  # e.g. "A:42", ":42", "A:42B", ":42B"
                reskey_to_resname.setdefault(reskey, set())
                reskey_to_resname[reskey].add(resname)

                if reskey == buffered_reskey:  # this line continues existing residue
                    pdb_block += line
                else:
                    if buffered_reskey is not None:
                        _add_if_new(blocks_by_residue, buffered_reskey, pdb_block, interrupted_residues)
                    buffered_reskey = reskey
                    pdb_block = line

        if len(pdb_block):  # there was not a TER line
            _add_if_new(blocks_by_residue, reskey, pdb_block, interrupted_residues)

        # verify that each identifier (e.g. "A:17" has a single resname
        violations = {k: v for k, v in reskey_to_resname.items() if len(v) != 1}
        if len(violations):
            msg = "each residue key must have exactly 1 resname" + os_linesep
            msg += f"but got {violations=}"
            raise ValueError(msg)

        # create rdkit molecules from PDB strings for each residue
        raw_input_mols = {}
        for reskey, pdb_block in blocks_by_residue.items():
            pdbmol = Chem.MolFromPDBBlock(pdb_block, removeHs=False)  # TODO RDKit ignores AltLoc ?
            resname = list(reskey_to_resname[reskey])[0]  # already verified length of set is 1
            raw_input_mols[reskey] = (pdbmol, resname)

        return raw_input_mols

    @staticmethod
    def _prody_to_residue_mols(prody_obj: ALLOWED_PRODY_TYPES) -> dict:
        raw_input_mols = {}
        reskey_to_resname = {}
        # generate macromolecule hierarchy iterator
        hierarchy = prody_obj.getHierView()
        # iterate chains
        for chain in hierarchy.iterChains():
            # iterate residues
            for res in chain.iterResidues():
                # gather residue info
                chain_id = str(res.getChid())
                res_name = str(res.getResname())
                res_num = int(res.getResnum())
                res_index = int(res.getResnum())
                icode = str(res.getIcode())
                reskey = f"{chain_id}:{res_num}{icode}"
                reskey_to_resname.setdefault(reskey, set())
                reskey_to_resname[reskey].add(res_name)
                mol_name = f"{chain_id}:{res_index}:{res_name}:{res_num}:{icode}"
                # we are not sanitizing because protonated LYS don't have the
                # formal charge set on the N and Chem.SanitizeMol raises error
                prody_mol = prody_to_rdkit(res, name=mol_name, sanitize=False)
                raw_input_mols[reskey] = (prody_mol, res_name)
        return raw_input_mols

    def to_pdb(self, use_modified_coords=False, modified_coords_index=0):
        pdbout = ""
        atom_count = 0
        icode = ""
        pdb_line = "{:6s}{:5d} {:^4s} {:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}                       {:2s} "
        pdb_line += pathlib.os.linesep
        for res_id in self.residues:
            if self.residues[res_id].user_deleted or self.residues[res_id].rdkit_mol is None:
                continue
            resmol = self.residues[res_id].rdkit_mol
            if use_modified_coords and self.residues[res_id].molsetup is not None:
                molsetup = self.residues[res_id].molsetup
                if len(molsetup.modified_atom_positions) <= modified_coords_index:
                    errmsg = "Requesting pose %d but only got %d in molsetup of %s" % (
                        modified_coords_index, len(molsetup.modified_atom_positions), res_id)
                    raise RuntimeError(errmsg)
                p = molsetup.modified_atom_positions[modified_coords_index]
                modified_positions = molsetup.get_conformer_with_modified_positions(p).GetPositions()
                positions = {}
                for i, j in self.residues[res_id].molsetup_mapidx.items():
                    positions[j] = modified_positions[i]
            else:
                positions = {i: xyz for (i, xyz) in enumerate(resmol.GetConformer().GetPositions())}

            chain, resname, resnum = res_id.split(":")
            resnum = int(resnum)

            for (i, atom) in enumerate(resmol.GetAtoms()):
                atom_count += 1
                props = atom.GetPropsAsDict()
                atom_name = props.get("atom_name", "")
                x, y, z = positions[i]
                element = mini_periodic_table[atom.GetAtomicNum()]
                pdbout += pdb_line.format("ATOM", atom_count, atom_name, resname, chain, resnum, icode, x, y, z,
                                          element)
        return pdbout

    def export_static_atom_params(self):
        atom_params = {}
        counter_atoms = 0
        coords = []
        dedicated_attribute = ("charge", "atom_type")  # molsetup has a dedicated attribute
        for res_id in self.get_valid_residues():
            molsetup = self.residues[res_id].molsetup
            wanted_atom_indices = []
            for i, ignore in molsetup.atom_ignore.items():
                if not ignore and not self.residues[res_id].is_flexres_atom[i]:
                    wanted_atom_indices.append(i)
            for key, values in molsetup.atom_params.items():
                atom_params.setdefault(key, [None] * counter_atoms)  # add new "column"
                for i in wanted_atom_indices:
                    atom_params[key].append(values[i])
            for key in dedicated_attribute:
                atom_params.setdefault(key, [None] * counter_atoms)  # add new "column"
                values_dict = getattr(molsetup, key)
                for i in wanted_atom_indices:
                    atom_params[key].append(values_dict[i])
            counter_atoms += len(wanted_atom_indices)
            added_keys = set(molsetup.atom_params).union(dedicated_attribute)
            for key in set(atom_params).difference(added_keys):  # <key> missing in current molsetup
                atom_params[key].extend([None] * len(wanted_atom_indices))  # fill in incomplete "row"
            coords.append(molsetup.coord[i])
        if hasattr(self, "param_rename"):  # e.g. "gasteiger" -> "q"
            for key, new_key in self.param_rename.items():
                atom_params[new_key] = atom_params.pop(key)
        return atom_params, coords

    # The following functions return filtered dictionaries of residues based on the value of residue flags.
    def get_user_deleted_residues(self):
        return {k: v for k, v in self.residues.items() if v.user_deleted}

    def get_non_user_deleted_residues(self):
        return {k: v for k, v in self.residues.items() if not v.user_deleted}

    def get_ignored_residues(self):
        return {k: v for k, v in self.residues.items() if v.rdkit_mol is None}

    def get_not_ignored_residues(self):
        return {k: v for k, v in self.residues.items() if not v.rdkit_mol is not None}

    # TODO: rename this
    def get_valid_residues(self):
        return {k: v for k, v in self.residues.items() if v.is_valid_residue()}


residues_rotamers = {"SER": [("C", "CA", "CB", "OG")],
                     "THR": [("C", "CA", "CB", "CG2")],
                     "CYS": [("C", "CA", "CB", "SG")],
                     "VAL": [("C", "CA", "CB", "CG1")],
                     "HIS": [("C", "CA", "CB", "CG"),
                             ("CA", "CB", "CG", "CD2")],
                     "ASN": [("C", "CA", "CB", "CG"),
                             ("CA", "CB", "CG", "ND2")],
                     "ASP": [("C", "CA", "CB", "CG"),
                             ("CA", "CB", "CG", "OD1")],
                     "ILE": [("C", "CA", "CB", "CG2"),
                             ("CA", "CB", "CG2", "CD1")],
                     "LEU": [("C", "CA", "CB", "CG"),
                             ("CA", "CB", "CG", "CD1")],
                     "PHE": [("C", "CA", "CB", "CG"),
                             ("CA", "CB", "CG", "CD2")],
                     "TYR": [("C", "CA", "CB", "CG"),
                             ("CA", "CB", "CG", "CD2")],
                     "TRP": [("C", "CA", "CB", "CG"),
                             ("CA", "CB", "CG", "CD2")],
                     "GLU": [("C", "CA", "CB", "CG"),
                             ("CA", "CB", "CG", "CD"),
                             ("CB", "CG", "CD", "OE1")],
                     "GLN": [("C", "CA", "CB", "CG"),
                             ("CA", "CB", "CG", "CD"),
                             ("CB", "CG", "CD", "OE1")],
                     "MET": [("C", "CA", "CB", "CG"),
                             ("CA", "CB", "CG", "SD"),
                             ("CB", "CG", "SD", "CE")],
                     "ARG": [("C", "CA", "CB", "CG"),
                             ("CA", "CB", "CG", "CD"),
                             ("CB", "CG", "CD", "NE"),
                             ("CG", "CD", "NE", "CZ")],
                     "LYS": [("C", "CA", "CB", "CG"),
                             ("CA", "CB", "CG", "CD"),
                             ("CB", "CG", "CD", "CE"),
                             ("CG", "CD", "CE", "NZ")]}


def add_rotamers_to_chorizo_molsetups(rotamer_states_list, chorizo):
    rotamer_res_disambiguate = {}
    for primary_res, specific_res_list in chorizo.residue_chem_templates.ambiguous.items():
        for specific_res in specific_res_list:
            rotamer_res_disambiguate[specific_res] = primary_res

    no_resname_to_resname = {}
    for res_with_resname in chorizo.residues:
        chain, resname, resnum = res_with_resname.split(":")
        no_resname_key = f"{chain}:{resnum}"
        if no_resname_key in no_resname_to_resname:
            errmsg = "both %s and %s would be keyed by %s" % (
                res_with_resname, no_resname_to_resname[no_resname_key], no_resname_key)
            raise RuntimeError(errmsg)
        no_resname_to_resname[no_resname_key] = res_with_resname

    state_indices_list = []
    for state_index, state_dict in enumerate(rotamer_states_list):
        print(f"adding rotamer state {state_index + 1}")
        state_indices = {}
        for res_no_resname, angles in state_dict.items():
            res_with_resname = no_resname_to_resname[res_no_resname]
            if chorizo.residues[res_with_resname].molsetup is None:
                raise RuntimeError("no molsetup for %s, can't add rotamers" % (res_with_resname))
            # next block is inefficient for large rotamer_states_list
            # refactored chorizos could help by having the following
            # data readily available
            molsetup = chorizo.residues[res_with_resname].molsetup
            name_to_molsetup_idx = {}
            for atom_index, pdbinfo in molsetup.pdbinfo.items():
                atom_name = pdbinfo.name
                name_to_molsetup_idx[atom_name] = atom_index

            resname = res_with_resname.split(":")[1]
            resname = rotamer_res_disambiguate.get(resname, resname)

            atom_names = residues_rotamers[resname]
            if len(atom_names) != len(angles):
                raise RuntimeError(
                    f"expected {len(atom_names)} angles for {resname}, got {len(angles)}")

            atom_idxs = []
            for names in atom_names:
                tmp = [name_to_molsetup_idx[name] for name in names]
                atom_idxs.append(tmp)

            state_indices[res_with_resname] = len(molsetup.rotamers)
            molsetup.add_rotamer(atom_idxs, np.radians(angles))

        state_indices_list.append(state_indices)

    return state_indices_list


class ChorizoResidue:
    """Individual subunit of a polymer represented by LinkedRDKitChorizo.

    Attributes
    ----------
    raw_rdkit_mol: RDKit Mol
        defines element and connectivity within a residue. Bond orders and
        formal charges may be incorrect, and hydrogens may be missing.
        This molecule may originate from a PDB string and it defines also
        the positions of the atoms.
    rdkit_mol: RDKit Mol
        Copy of the molecule from a ResidueTemplate, with positions from
        raw_rdkit_mol. All hydrogens are real atoms except for those
        at connections with adjacent residues.
    mapidx_to_raw: dict (int -> int)
        indices of atom in rdkit_mol to raw_rdkit_mol
    input_resname: str
        usually a three-letter code from a PDB
    template_key: str
        identifies instance of ResidueTemplate in ResidueChemTemplates
    atom_names: list (str)
        names of the atoms in the same order as rdkit_mol
    padded_mol: RDKit Mol
        molecule padded with ResiduePadder
    molsetup_mapidx: dict (int -> int)
        key: index of atom in padded_mol
        value: index of atom in rdkit_mol
    """

    def __init__(self, raw_input_mol, rdkit_mol, mapidx_to_raw, input_resname=None, template_key=None,
                 atom_names=None): #  link_labels=None,

        self.raw_rdkit_mol = raw_input_mol
        self.rdkit_mol = rdkit_mol
        self.mapidx_to_raw = mapidx_to_raw
        self.residue_template_key = template_key  # same as pdb_resname except NALA, etc
        self.input_resname = input_resname  # even if using openmm topology, there are residues
        self.atom_names = atom_names  # assumes same order and length as atoms in rdkit_mol, used in e.g. rotamers

        # TODO convert link indices/labels in template to rdkit_mol indices herein
        #self.link_labels = {}

        if mapidx_to_raw is not None:
            self.mapidx_from_raw = {j: i for (i, j) in mapidx_to_raw.items()}
            assert len(self.mapidx_from_raw) == len(self.mapidx_to_raw)
        else:
            self.mapidx_from_raw = None

        ### self.residue_id = residue_id
        ### self.previous_id = previous_id
        ### self.next_id = next_id

        self.padded_mol = None
        self.molsetup = None
        self.molsetup_mapidx = None
        self.is_flexres_atom = None  # Check about these data types/Do we want the default to be None or empty

        # flags
        ### self.ignore_residue = False NOTE using rdkit_mol=None instead
        self.is_movable = False
        self.user_deleted = False

    def set_atom_names(self, atom_names_list):
        if self.rdkit_mol is None:
            raise RuntimeError("can't set atom_names if rdkit_mol is not set yet")
        if len(atom_names_list) != self.rdkit_mol.GetNumAtoms():
            raise ValueError(f"{len(atom_names_list)=} differs from {self.rdkit_mol.GetNumAtoms()=}")
        name_types = set([type(name) for name in atom_names_list])
        if name_types != {str}:
            raise ValueError(f"atom names must be str but {name_types=}")
        self.atom_names = atom_names_list
        return

    def to_json(self):
        return json.dumps(self, cls=ChorizoResidueEncoder)

    @classmethod
    def from_json(cls, json_string):
        residue = json.loads(json_string, object_hook=cls.chorizo_residue_json_decoder) 
        return residue

    def is_valid_residue(self):
        """
        Returns true if the residue is not marked as deleted by a user and has not been marked as a residue to
        ignore
        """
        return self.rdkit_mol is not None and not self.user_deleted


    @staticmethod
    def chorizo_residue_json_decoder(obj):
        """
        Takes an object and attempts to decode it into a chorizo residue object.

        Parameters
        ----------
        obj: Object
            This can be any object, but it should be a dictionary from deserializing a JSON of a chorizo residue object.

        Returns
        -------
        If the input is a dictionary corresponding to a molecule setup, will return a Chorizo Residue with data
        populated from the dictionary. Otherwise, returns the input object.

        """
        # if the input object is not a dict, we know that it will not be parsable and is unlikely to be usable or
        # safe data, so we should ignore it.
        if type(obj) is not dict:
            return obj
        # check that all the keys we expect are in the object dictionary as a safety measure
        expected_residue_keys = {"residue_id", "pdb_text", "previous_id", "next_id",
                                 "rdkit_mol",
                                 "molsetup", "molsetup_mapidx", "is_flexres_atom", "ignore_residue", "is_movable",
                                 "user_deleted", "additional_connections"}
        if set(obj.keys()) != expected_residue_keys:
            return obj
        # creates a chorizo residue and sets all the expected fields
        residue = ChorizoResidue(obj["residue_id"], obj["pdb_text"], obj["previous_id"], obj["next_id"])
        residue.molsetup = MoleculeSetup.molsetup_json_decoder(obj["molsetup"])
        residue.molsetup_mapidx = obj["molsetup_mapidx"]
        residue.is_flexres_atom = obj["is_flexres_atom"]
        residue.ignore_residue = obj["ignore_residue"]
        residue.user_deleted = obj["user_deleted"]
        residue.additional_connections = [ResidueAdditionalConnection(*v) for k, v in obj["additional_connections"]]
        rdkit_mols = rdMolInterchange.JSONToMols(obj["rdkit_mol"])
        if len(rdkit_mols) != 1:
            raise ValueError(f"Expected 1 rdkit mol from json string but got {len(rdkit_mols)}")
        residue.rdkit_mol = rdkit_mols[0]
        return residue

class ResiduePadder:
    """
    Data and methods to pad rdkit molecules of chorizo residues with parts of adjacent residues.

    """

    # Replacing ResidueConnection by ResiduePadding
    # Why have two ResiduePadding instances per connection between two-residues?
    #  - three-way merge: if three carbons joined in cyclopropare, we can still pad
    #  - defines padding in the reaction for blunt residues
    #  - all bonds will be defined in the input topology after a future refactor

    # reaction should not delete atoms, not even Hs
    # reaction should create bonds at non-real Hs (implicit or explicit rdktt H)

    def __init__(self, rxn_smarts, adjacent_res_smarts=None):  # , link_labels=None):
        """
        Parameters
        ----------
        rxn_smarts: string
            Reaction SMARTS to pad a link atom of a ChorizoResidue molecule.
            Product atoms that are not mapped in the reactants will have
            their coordinates set from an adjacent residue molecule, given
            that adjacent_res_smarts is provided and the atom labels match
            the unmapped product atoms of rxn_smarts.
        adjacent_res_smarts: string
            SMARTS pattern to identify atoms in molecule of adjacent residue
            and copy their positions to padding atoms. The SMARTS atom labels
            must match those of the product atoms of rxn_smarts that are
            unmapped in the reagents.
        """

        rxn = rdChemReactions.ReactionFromSmarts(rxn_smarts)
        if rxn.GetNumReactantTemplates() != 1:
            raise ValueError(f"expected 1 reactants, got {rxn.GetNumReactantTemplates()} in {rxn_smarts}")
        if rxn.GetNumProductTemplates() != 1:
            raise ValueError(f"expected 1 product, got {rxn.GetNumProductTemplates()} in {rxn_smarts}")
        self.rxn = rxn
        if adjacent_res_smarts is None:
            self.adjacent_smartsmol = None
            self.adjacent_smartsmol_mapidx = None
        else:
            adjacent_smartsmol = Chem.MolFromSmarts(adjacent_res_smarts)
            assert adjacent_smartsmol is not None
            is_ok, padding_ids, adjacent_ids = self._check_adj_smarts(rxn, adjacent_smartsmol)
            if not is_ok:
                msg = f"SMARTS labels in adjacent_smartsmol ({adjacent_ids}) differ from unmapped product labels in reaction ({padding_ids})" + os_linesep
                msg += f"{rxn_smarts=}" + os_linesep
                msg += f"{adjacent_res_smarts=}" + os_linesep
            self.adjacent_smartsmol = adjacent_smartsmol
            self.adjacent_smartsmol_mapidx = {}
            for atom in self.adjacent_smartsmol.GetAtoms():
                if atom.HasProp("molAtomMapNumber"):
                    j = atom.GetIntProp("molAtomMapNumber")
                    self.adjacent_smartsmol_mapidx[j] = atom.GetIdx()

    def __call__(self, target_mol,
                 adjacent_mol=None,
                 target_required_atom_index=None,
                 adjacent_required_atom_index=None,
                 ):
        # add Hs only to padding atoms
        # copy coordinates if adjacent res has Hs bound to heavy atoms
        # labels have been checked upstream

        products, idxmap = react_and_map((target_mol,), self.rxn)
        if target_required_atom_index is not None:
            passing_products = []
            for product, atomidx in zip(products, idxmap["atom_idx"]):
                if target_required_atom_index in atomidx[0]:  # 1 product template
                    passing_products.append(product)
            if len(passing_products) > 1:
                raise RuntimeError("more than 1 padding product has target_required_atom_index")
            products = passing_products
        if len(products) > 1 and target_required_atom_index is None:
            raise RuntimeError("more than 1 padding product, consider using target_required_atom_index")
        if len(products) == 0:
            raise RuntimeError("zero products from padding reaction")

        padded_mol = products[0][0]

        # is_pad_atom = [reactant_idx == 0 for reactant_idx in idxmap["reactant_idx"][0][0]]
        # mapidx = {i: j for (i, j) in enumerate(idxmap["atom_idx"][0][0]) if j is not None}
        # return padded_mol, is_pad_atom, mapidx

        mapidx = idxmap["atom_idx"][0][0]
        padding_heavy_atoms = []
        for i, j in enumerate(idxmap["atom_idx"][0][0]):
            # j is index in reactants, if j is None then it's a padding atom
            if j is None and padded_mol.GetAtomWithIdx(i).GetAtomicNum() != 1:
                padding_heavy_atoms.append(i)

        if adjacent_mol is None:
            padded_mol.UpdatePropertyCache()  # avoids getNumImplicitHs() called without preceding call to calcImplicitValence()
            Chem.SanitizeMol(padded_mol)  # just in case
            padded_h = Chem.AddHs(padded_mol, onlyOnAtoms=padding_heavy_atoms)
            mapidx += [None] * (padded_h.GetNumAtoms() - padded_mol.GetNumAtoms())
        elif self.adjacent_smartsmol is None:
            raise RuntimeError("had to be initialized with adjacent_res_smarts to support adjacent_mol")
        else:
            hits = adjacent_mol.GetSubstructMatches(self.adjacent_smartsmol)
            if adjacent_required_atom_index is not None:
                hits = [hit for hit in hits if adjacent_required_atom_index in hit]
            if len(hits) != 1:
                raise RuntimeError(f"length of hits must be 1, but it's {len(hits)}") # TODO use required_atom_idx from bonds
            hit = hits[0]
            adjacent_coords = adjacent_mol.GetConformer().GetPositions()
            for atom in self.adjacent_smartsmol.GetAtoms():
                if not atom.HasProp("molAtomMapNumber"):
                    continue
                j = atom.GetIntProp("molAtomMapNumber")
                k = idxmap["new_atom_label"][0][0].index(j)
                l = self.adjacent_smartsmol_mapidx[j]
                padded_mol.GetConformer().SetAtomPosition(k, adjacent_coords[hit[l]])
            padded_mol.UpdatePropertyCache()  # avoids getNumImplicitHs() called without preceding call to calcImplicitValence()
            Chem.SanitizeMol(padded_mol)  # got crooked Hs without this
            padded_h = Chem.AddHs(padded_mol, onlyOnAtoms=padding_heavy_atoms, addCoords=True)
        return padded_h, mapidx

    @staticmethod
    def _check_adj_smarts(rxn, adjacent_smartsmol):
        def get_molAtomMapNumbers(mol):
            numbers = set()
            for atom in mol.GetAtoms():
                if atom.HasProp("molAtomMapNumber"):
                    numbers.add(atom.GetIntProp("molAtomMapNumber"))
            return numbers

        reactant_ids = get_molAtomMapNumbers(rxn.GetReactantTemplate(0))
        product_ids = get_molAtomMapNumbers(rxn.GetProductTemplate(0))
        adjacent_ids = get_molAtomMapNumbers(adjacent_smartsmol)
        padding_ids = product_ids.difference(reactant_ids)
        is_ok = padding_ids == adjacent_ids
        return is_ok, padding_ids, adjacent_ids

    @classmethod
    def from_json(cls, string):
        d = json.loads(string)
        return cls(**d)

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__)


class ResidueTemplate:
    """
    Data and methods to pad rdkit molecules of chorizo residues with parts of adjacent residues.

    Attributes
    ----------
    mol: RDKit Mol
        molecule with the exact atoms that constitute the system.
        All Hs are explicit, but atoms bonded to adjacent residues miss an H.
    link_labels: dict (int -> string)
        Keys are indices of atoms that need padding
        Values are strings to identify instances of ResiduePadder
    atom_names: list (string)
        list of atom names, matching order of atoms in rdkit mol
    """


    def __init__(self, smiles, link_labels=None, atom_names=None):
        ps = Chem.SmilesParserParams()
        ps.removeHs = False
        mol = Chem.MolFromSmiles(smiles, ps)
        self.check(mol, link_labels, atom_names)
        self.mol = mol
        self.link_labels = link_labels
        self.atom_names = atom_names
        return

    def check(self, mol, link_labels, atom_names):
        have_implicit_hs = set()
        for atom in mol.GetAtoms():
            if atom.GetTotalNumHs() > 0:
                have_implicit_hs.add(atom.GetIdx())
        if link_labels is not None and set(link_labels) != have_implicit_hs:
            raise ValueError(f"expected any atom with non-real Hs ({have_implicit_hs}) to be in {link_labels=}")
        if atom_names is None:
            return
        #data_lengths = set([len(values) for (_, values) in data.items()])
        #if len(data_lengths) != 1:
        #    raise ValueError(f"each array in data must have the same length, but got {data_lengths=}")
        #data_length = data_lengths.pop()
        if len(atom_names) != mol.GetNumAtoms():
            raise ValueError(f"{len(atom_names)=} differs from {mol.GetNumAtoms()=}")
        return

    def match(self, input_mol):
        mapping = mapping_by_mcs(self.mol, input_mol)
        mapping_inv = {value: key for (key, value) in mapping.items()}
        if len(mapping_inv) != len(mapping):
            raise RuntimeError(f"bug in atom indices, repeated value different keys? {mapping=}")
        # atoms "missing" exist in self.mol but not in input_mol
        # "excess" atoms exist in input_mol but not in self.mol
        result = {
            "H": {"found": 0, "missing": 0, "excess": 0},
            "heavy": {"found": 0, "missing": 0, "excess": 0},
        }
        for atom in self.mol.GetAtoms():
            element = "H" if atom.GetAtomicNum() == 1 else "heavy"
            key = "found" if atom.GetIdx() in mapping else "missing"
            result[element][key] += 1
        for atom in input_mol.GetAtoms():
            element = "H" if atom.GetAtomicNum() == 1 else "heavy"
            if atom.GetIdx() not in mapping_inv:
                result[element]["excess"] += 1
        return result, mapping


class ChorizoResidueEncoder(json.JSONEncoder):
    """
    JSON Encoder class for Chorizo Residue objects.
    """

    molecule_setup_encoder = MoleculeSetupEncoder()

    def default(self, obj):
        """
        Overrides the default JSON encoder for data structures for Chorizo Residue objects.

        Parameters
        ----------
        obj: object
            Can take any object as input, but will only create the Chorizo Residue JSON format for Molsetup objects.
            For all other objects will return the default json encoding.

        Returns
        -------
        A JSON serializable object that represents the Chorizo Residue class or the default JSONEncoder output for an
        object.
        """
        if isinstance(obj, ChorizoResidue):
            return {
                "residue_id": obj.residue_id,
                "pdb_text": obj.pdb_text,
                "previous_id": obj.previous_id,
                "next_id": obj.next_id,
                "rdkit_mol": rdMolInterchange.MolToJSON(obj.rdkit_mol),
                "molsetup": self.molecule_setup_encoder.default(obj.molsetup),
                "molsetup_mapidx": obj.molsetup_mapidx,
                "is_flexres_atom": obj.is_flexres_atom,
                "ignore_residue": obj.ignore_residue,
                "is_movable": obj.is_movable,
                "user_deleted": obj.user_deleted,
                "additional_connections": [var.__dict__ for var in obj.additional_connections]
            }
        return json.JSONEncoder.default(self, obj)
