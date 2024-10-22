import pathlib
import json
import logging
from os import linesep as os_linesep
from typing import Union
from typing import Optional

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
from .utils.pdbutils import PDBAtomInfo

import numpy as np

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

residues_rotamers = {
    "SER": [("C", "CA", "CB", "OG")],
    "THR": [("C", "CA", "CB", "CG2")],
    "CYS": [("C", "CA", "CB", "SG")],
    "VAL": [("C", "CA", "CB", "CG1")],
    "HIS": [("C", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD2")],
    "ASN": [("C", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND2")],
    "ASP": [("C", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "ILE": [("C", "CA", "CB", "CG2"), ("CA", "CB", "CG2", "CD1")],
    "LEU": [("C", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "PHE": [("C", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD2")],
    "TYR": [("C", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD2")],
    "TRP": [("C", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD2")],
    "GLU": [
        ("C", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "OE1"),
    ],
    "GLN": [
        ("C", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "OE1"),
    ],
    "MET": [
        ("C", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "SD"),
        ("CB", "CG", "SD", "CE"),
    ],
    "ARG": [
        ("C", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "NE"),
        ("CG", "CD", "NE", "CZ"),
    ],
    "LYS": [
        ("C", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "CE"),
        ("CG", "CD", "CE", "NZ"),
    ],
}


def find_graph_paths(graph, start_node, end_nodes, current_path=(), paths_found=()):
    """
    Recursively finds all paths between start and end nodes.

    Parameters
    ----------
    graph
    start_node
    end_nodes
    current_path
    paths_found

    Returns
    -------

    """
    current_path = current_path + (start_node,)
    paths_found = list(paths_found)
    for node in graph[start_node]:
        if node in current_path:
            continue
        if node in end_nodes:
            paths_found.append(list(current_path) + [node])
        more_paths = find_graph_paths(graph, node, end_nodes, current_path)
        paths_found.extend(more_paths)
    return paths_found


def find_inter_mols_bonds(mols_dict):
    """

    Parameters
    ----------
    mols_dict:

    Returns
    -------

    """
    covalent_radius = {  # from wikipedia
        1: 0.31,
        5: 0.84,
        6: 0.76,
        7: 0.71,
        8: 0.66,
        9: 0.57,
        12: 0.00,  # hack to avoid bonds with metals
        14: 1.11,
        15: 1.07,
        16: 1.05,
        17: 1.02,
        # 19: 2.03,
        20: 0.00,
        # 24: 1.39,
        25: 0.00,  # hack to avoid bonds with metals
        26: 0.00,
        30: 0.00,  # hack to avoid bonds with metals
        # 34: 1.20,
        35: 1.20,
        53: 1.39,
    }
    allowance = 1.2  # vina uses 1.1 but covalent radii are shorter here
    max_possible_covalent_radius = (
        2 * allowance * max([r for k, r in covalent_radius.items()])
    )
    cubes_min = []
    cubes_max = []
    for key, (mol, _) in mols_dict.items():
        positions = mol.GetConformer().GetPositions()
        cubes_min.append(np.min(positions, axis=0))
        cubes_max.append(np.max(positions, axis=0))
    tmp = np.array([0, 0, 1, 1])
    pairs_to_consider = []
    keys = list(mols_dict)
    for i in range(len(mols_dict)):
        for j in range(i + 1, len(mols_dict)):
            do_consider = True
            for d in range(3):
                x = (cubes_min[i][d], cubes_max[i][d], cubes_min[j][d], cubes_max[j][d])
                idx = np.argsort(x)
                has_overlap = tmp[idx][0] != tmp[idx][1]
                close_enough = abs(x[idx[1]] - x[idx[2]]) < max_possible_covalent_radius
                do_consider &= close_enough or has_overlap
            if do_consider:
                pairs_to_consider.append((i, j))

    bonds = {}  # key is pair mol indices, valuei is list of pairs of atom indices
    for i, j in pairs_to_consider:
        p1 = mols_dict[keys[i]][0].GetConformer().GetPositions()
        p2 = mols_dict[keys[j]][0].GetConformer().GetPositions()
        for a1 in mols_dict[keys[i]][0].GetAtoms():
            for a2 in mols_dict[keys[j]][0].GetAtoms():
                vec = p1[a1.GetIdx()] - p2[a2.GetIdx()]
                distsqr = np.dot(vec, vec)

                # check if atom has implemented covalent radius
                for atom in [a1, a2]:
                    if atom.GetAtomicNum() not in covalent_radius:
                        raise RuntimeError(f"Element {periodic_table.GetElementSymbol(atom.GetAtomicNum())} doesn't have an implemented covalent radius, which was required for the perception of intermolecular bonds. ")
                    
                cov_dist = (
                    covalent_radius[a1.GetAtomicNum()]
                    + covalent_radius[a2.GetAtomicNum()]
                )
                if distsqr < (allowance * cov_dist) ** 2:
                    key = (keys[i], keys[j])
                    value = (a1.GetIdx(), a2.GetIdx())
                    bonds.setdefault(key, [])
                    bonds[key].append(value)
    return bonds


def mapping_by_mcs(mol, ref):
    """

    Parameters
    ----------
    mol
    ref

    Returns
    -------

    """
    mcs_result = rdFMCS.FindMCS([mol, ref], bondCompare=rdFMCS.BondCompare.CompareAny)
    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)

    mol_idxs = mol.GetSubstructMatch(mcs_mol)
    ref_idxs = ref.GetSubstructMatch(mcs_mol)

    atom_map = {i: j for (i, j) in zip(mol_idxs, ref_idxs)}
    return atom_map


def _snap_to_int(value, tolerance=0.12):
    """

    Parameters
    ----------
    value
    tolerance

    Returns
    -------

    """
    for inc in [-1, 0, 1]:
        if abs(value - int(value) - inc) <= tolerance:
            return int(value) + inc
    return None


def divide_int_gracefully(integer, weights, allow_equal_weights_to_differ=False):
    """

    Parameters
    ----------
    integer
    weights
    allow_equal_weights_to_differ

    Returns
    -------

    """
    for weight in weights:
        if type(weight) not in [int, float] or weight < 0:
            raise ValueError("weights must be numeric and non-negative")
    if type(integer) is not int:
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
    for j in range(1, 2**nr_groups):
        n_changes = 0
        combo = []
        for grpidx in range(nr_groups):
            is_changed = bool(j & 2**grpidx)
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


def rectify_charges(q_list, net_charge=None, decimals=3) -> list[float]:
    """
    Makes charges 3 decimals in length and ensures they sum to an integer

    Parameters
    ----------
    q_list
    net_charge
    decimals

    Returns
    -------
    charges_dec: list[float]

    """

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
    surplus_int = _snap_to_int(10**decimals * surplus)

    if surplus_int == 0:
        return charges_dec

    weights = [abs(q) for q in q_list]
    surplus_int_splits = divide_int_gracefully(surplus_int, weights)
    for i, increment in enumerate(surplus_int_splits):
        charges_dec[i] += 10**-decimals * increment

    return charges_dec


def get_updated_positions(residue, new_positions: dict): 
    """
    Returns full set of positions for the rdkit_mol in residue given a partial
    set of new_positions. Hydrogens not specified in new_positions will
    have their position reset by RDKit if they are one or two bonds away
    from an atom in new_positions.

    Parameters
    ----------
    residue: ChorizoResidue
        molecule associated with new positions
    new_positions: dict (int -> (float, float, float))
                         |      |
                atom_index      |
                                new_position
    """

    h_to_update = set()
    mol = Chem.Mol(residue.rdkit_mol)  # avoids side effects
    conformer = mol.GetConformer()

    for n1 in (mol.GetAtomWithIdx(idx) for idx in new_positions):
        for n2 in n1.GetNeighbors():
            if n2.GetAtomicNum() == 1:  # 1 bond away
                h_to_update.add(n2.GetIdx())
            else:
                if n2.GetIdx() not in new_positions:  # 2 bonds away
                    h_to_update.update(set(n2h.GetIdx() for n2h in n2.GetNeighbors() if n2h.GetAtomicNum() == 1))

    # hydrogens in new_positions shall not be updated by RDKit
    h_to_update -= set(new_positions)

    for index in new_positions:
        x, y, z = new_positions[index]
        p = Point3D(float(x), float(y), float(z))
        conformer.SetAtomPosition(index, p)
    if h_to_update:
        update_H_positions(mol, list(h_to_update))
    return mol.GetConformer().GetPositions()


def update_H_positions(mol: Chem.Mol, indices_to_update: list[int]) -> None:
    """
    Re-calculates the position of some hydrogens already existing in the mol. Does not guarantee that chirality can be
    preserved.

    Parameters
    ----------
    mol: Chem.Mol
        RDKit Mol object with hydrogens
    indices_to_update: list[int]
        Hydrogen indices to update

    Returns
    -------
    None

    Raises
    ------
    RuntimeError:
        If a provided index in indices_to_update is not a Hydrogen, if a Hydrogen only has Hydrogen neighbors, or if the
        number of visited Hydrogens does not match the number of Hydrogens marked to be deleted.
    """

    # Gets the conformer and a readable and writable version of the Mol using RDKit
    conf = mol.GetConformer()
    tmpmol = Chem.RWMol(mol)
    # Sets up data structures to manage Hydrogens to delete and add
    to_del = {}
    to_add_h = []
    # Loops through input indices_to_update, checks index validity, adds data to the addition and deletion data structs
    for h_index in indices_to_update:
        # Checks that the atom at this index is a Hydrogen
        atom = tmpmol.GetAtomWithIdx(h_index)
        if atom.GetAtomicNum() != 1:
            raise RuntimeError("only H positions can be updated")
        # Ensures that all Hydrogens have at least 1 non-Hydrogen neighbor
        heavy_neighbors = []
        for neigh_atom in atom.GetNeighbors():
            if neigh_atom.GetAtomicNum() != 1:
                heavy_neighbors.append(neigh_atom)
        if len(heavy_neighbors) != 1:
            raise RuntimeError(
                f"hydrogens must have 1 non-H neighbor, got {len(heavy_neighbors)}"
            )
        # Adds the first neighbor to the addition and deletion data structures.
        to_add_h.append(heavy_neighbors[0])
        to_del[h_index] = heavy_neighbors[0]
    # Loops through the delete list and deletes the
    for i in sorted(to_del, reverse=True):
        tmpmol.RemoveAtom(i)
        to_del[i].SetNumExplicitHs(to_del[i].GetNumExplicitHs() + 1)
    to_add_h = list(set([atom.GetIdx() for atom in to_add_h]))
    tmpmol = tmpmol.GetMol()
    tmpmol.UpdatePropertyCache()
    Chem.SanitizeMol(tmpmol)
    tmpmol = Chem.AddHs(tmpmol, onlyOnAtoms=to_add_h, addCoords=True)
    tmpconf = tmpmol.GetConformer()
    used_h = (
        set()
    )  # heavy atom may have multiple H that were missing, keep track of Hs that were visited
    for h_index, parent in to_del.items():
        for atom in tmpmol.GetAtomWithIdx(parent.GetIdx()).GetNeighbors():
            has_new_position = atom.GetIdx() >= mol.GetNumAtoms() - len(to_del)
            if atom.GetAtomicNum() == 1 and has_new_position:
                if atom.GetIdx() not in used_h:
                    # print(h_index, tuple(tmpconf.GetAtomPosition(atom.GetIdx())))
                    conf.SetAtomPosition(
                        h_index, tmpconf.GetAtomPosition(atom.GetIdx())
                    )
                    used_h.add(atom.GetIdx())
                    break  # h_index coords copied, don't look into further H bound to parent
                    # no guarantees about preserving chirality, which we don't need

    if len(used_h) != len(to_del):
        raise RuntimeError(
            f"Updated {len(used_h)} H positions but deleted {len(to_del)}"
        )

    return

def _delete_residues(res_to_delete, raw_input_mols):
    """

    Parameters
    ----------
    res_to_delete: list (str) or None
        residue IDs to delete in format <chain>:<resnum><icode>
    raw_input_mols: dict (str -> RDKit mol)
        keys are residue IDs

    Returns
    -------
    None
    (modifies raw_input_mols in-place)

    """
    if res_to_delete is None:
        return
    missing = set()
    for res in res_to_delete:
        if res not in raw_input_mols:
            missing.add(res)
        raw_input_mols.pop(res, None)
    if len(missing) > 0:
        msg = "can't find the following residues to delete: " + " ".join(missing)
        raise ValueError(msg)
    return

class ChorizoCreationError(RuntimeError):
    pass

def handle_parsing_situations(
    unmatched_res,
    unparsed_res,
    allow_bad_res,
    res_missed_altloc,
    res_needed_altloc,
    ):

    err = ""
    if unparsed_res:
        msg = f"- Parsing failed for: {unparsed_res}."
        if not allow_bad_res:
            err += msg + os_linesep
        else: 
            msg += " Ignored due to allow_bad_res."
            logger.warning(msg)

    if unmatched_res:
        msg = f"- Template matching failed for: {list(unmatched_res)}"
        if not allow_bad_res:
            err += msg + os_linesep
        else:
            msg += " Ignored due to allow_bad_res."
            logger.warning(msg)

    if err:
        err += "These residues can be ignored with option allow_bad_res." + os_linesep

    if res_needed_altloc: 
        msg = f"- Residues with alternate location: {res_needed_altloc}" + os_linesep
        msg += "Either specify an altloc for each with option wanted_altloc" + os_linesep
        msg += "or a general default altloc with option default_altloc."
        err += msg

    if res_missed_altloc:
        msg = f"- Requested altlocs not found for: {res_missed_altloc}." + os_linesep
        err += msg

    if err:
        raise ChorizoCreationError(err)
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
                link_labels = {
                    int(key): value for key, value in data["link_labels"].items()
                }
            # print(key)
            res_template = ResidueTemplate(
                data["smiles"], link_labels, data.get("atom_name", None)
            )
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
                msg += os_linesep
                msg += (
                    "consider LinkedRDKitChorizo.from_pdb_string(pdbstr)" + os_linesep
                )
            raise ValueError(msg)
        self.residue_chem_templates = residue_chem_templates
        residue_templates = residue_chem_templates.residue_templates
        padders = residue_chem_templates.padders
        ambiguous = residue_chem_templates.ambiguous

        # make sure all resiude_id in set_template exist
        if set_template is not None:
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

        # currently allowing only one bond per residue pair
        if any([len(v) > 1 for k, v in bonds.items()]):
            msg = "got more than one bond between some residue pairs:"
            for key, value in bonds.items():
                if len(value) > 1:
                    msg += f" {key}"
            raise ValueError(msg)
        bonds = {k: v[0] for k, v in bonds.items()}

        self.residues, self.log = self._get_residues(
            raw_input_mols,
            ambiguous,
            residue_templates,
            set_template,
            bonds,
            blunt_ends,
        )

        _bonds = {}
        for key, bond in bonds.items():
            res1 = self.residues[key[0]]
            res2 = self.residues[key[1]]
            if res1.rdkit_mol is None or res2.rdkit_mol is None:
                continue
            invmap1 = {j: i for i, j in res1.mapidx_to_raw.items()}
            invmap2 = {j: i for i, j in res2.mapidx_to_raw.items()}
            _bonds[key] = (invmap1[bond[0]], invmap2[bond[1]])
        bonds = _bonds

        # padding may seem overkill but we had to run a reaction anyway for h_coord_from_dipep
        padded_mols = self._build_padded_mols(self.residues, bonds, padders)
        for residue_id, (padded_mol, mapidx_from_pad) in padded_mols.items():
            residue = self.residues[residue_id]
            residue.padded_mol = padded_mol
            residue.molsetup_mapidx = mapidx_from_pad

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
                if (res1, res2) in bonds:
                    bonds.pop((res1, res2))
                elif (res2, res1) in bonds:
                    bonds.pop((res2, res1))
        chorizo = cls(
            raw_input_mols,
            bonds,
            chem_templates,
            mk_prep,
            set_template,
            blunt_ends,
        )

        unmatched_res = chorizo.get_ignored_residues()
        handle_parsing_situations(
            unmatched_res,
            unparsed_res,
            allow_bad_res,
            res_missed_altloc,
            res_needed_altloc,
        )

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
                if (res1, res2) in bonds:
                    bonds.pop((res1, res2))
                elif (res2, res1) in bonds:
                    bonds.pop((res2, res1))
        chorizo = cls(
            raw_input_mols,
            bonds,
            chem_templates,
            mk_prep,
            set_template,
            blunt_ends,
        )
        unmatched_res = chorizo.get_ignored_residues()
        handle_parsing_situations(
            unmatched_res,
            unparsed_res,
            allow_bad_res,
            res_missed_altloc,
            res_needed_altloc,
        )

        return chorizo

    @classmethod
    def from_json(cls, json_string):
        return json.loads(
            json_string,
            object_hook=linked_rdkit_chorizo_json_decoder,
        )

    def to_json(self):
        return json.dumps(self, cls=LinkedRDKitChorizoEncoder)

    def parameterize(self, mk_prep):
        """

        Parameters
        ----------
        mk_prep

        Returns
        -------

        """

        for residue_id in self.get_valid_residues():
            residue = self.residues[residue_id]
            molsetups = mk_prep(residue.padded_mol)
            if len(molsetups) != 1:
                raise NotImplementedError(f"need 1 molsetup but got {len(molsetups)}")
            molsetup = molsetups[0]
            self.residues[residue_id].molsetup = molsetup
            self.residues[residue_id].is_flexres_atom = [
                False for _ in molsetup.atoms
            ]

            # set ignore to True for atoms that are padding
            for atom in molsetup.atoms:
                if atom.index not in residue.molsetup_mapidx:
                    atom.is_ignore = True

            # recalculate flexibility tree after setting ignored atoms
            mk_prep.calc_flex(molsetup)

            # rectify charges to sum to integer (because of padding)
            if mk_prep.charge_model == "zero":
                net_charge = 0
            else:
                rdkit_mol = self.residues[residue_id].rdkit_mol
                net_charge = sum(
                    [atom.GetFormalCharge() for atom in rdkit_mol.GetAtoms()]
                )
            not_ignored_idxs = []
            charges = []
            for atom in molsetup.atoms:
                if atom.index in residue.molsetup_mapidx: # TODO offsite not in mapidx
                    charges.append(atom.charge)
                    not_ignored_idxs.append(atom.index)
            charges = rectify_charges(charges, net_charge, decimals=3)
            chain, resnum = residue_id.split(":")
            resname = self.residues[residue_id].input_resname
            if self.residues[residue_id].atom_names is None:
                atom_names = ["" for _ in not_ignored_idxs]
            else:
                atom_names = self.residues[residue_id].atom_names
            for i, j in enumerate(not_ignored_idxs):
                molsetup.atoms[j].charge = charges[i]
                atom_name = atom_names[residue.molsetup_mapidx[j]]
                if resnum[-1].isalpha():
                    icode = resnum[-1]
                    resnum = resnum[:-1]
                else:
                    icode = ""
                molsetup.atoms[j].pdbinfo = PDBAtomInfo(
                    atom_name, resname, int(resnum), icode, chain
                )
        return

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
    def _run_matching(raw_input_mol, residue_templates, bonds, residue_key, blunt_ends):
        """

        Parameters
        ----------
        raw_input_mol
        residue_templates
        bonds
        residue_key
        blunt_ends

        Returns
        -------

        """
        if blunt_ends is None:
            blunt_ends = []
        raw_atoms_with_bonds = []
        for (r1, r2), (i, j) in bonds.items():
            if r1 == residue_key:
                raw_atoms_with_bonds.append(i)
            if r2 == residue_key:
                raw_atoms_with_bonds.append(j)
        results = []
        for index, template in enumerate(residue_templates):

            # match intra-residue graph
            match_stats, mapping = template.match(raw_input_mol)
            from_raw = {value: key for (key, value) in mapping.items()}

            # match inter-residue bonds
            gotten = set()
            for raw_index in raw_atoms_with_bonds:
                index = from_raw[raw_index]
                gotten.add(index)

            # blunt ends are treated like fake bonds
            for res_id, atom_idx in blunt_ends:
                if res_id == residue_key:
                    index = from_raw[atom_idx]
                    gotten.add(index)
            if blunt_ends is not None:
                print(f"{gotten=}")
            expected = set(template.link_labels)
            result = {
                "found": gotten.intersection(expected),
                "missing": expected.difference(gotten),
                "excess": gotten.difference(expected),
            }
            match_stats["bonds"] = result
            match_stats["mapping"] = mapping

            results.append(match_stats)
        return results

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
            if result["H"]["excess"] > 0:
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
    def _get_residues(
        cls,
        raw_input_mols,
        ambiguous,
        residue_templates,
        set_template,
        bonds,
        blunt_ends,
    ):
        """

        Parameters
        ----------
        raw_input_mols
        ambiguous
        residue_templates
        set_template
        bonds
        blunt_ends

        Returns
        -------

        """
        residues = {}
        log = {
            "chosen_by_fewest_missing_H": {},
            "chosen_by_default": {},
            "no_match": [],
            "no_mol": [],
            "msg": "",
        }
        for residue_key, (raw_mol, input_resname) in raw_input_mols.items():
            if raw_mol is None:
                residues[residue_key] = ChorizoResidue(
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
            for (r1, r2), (i, j) in bonds.items():
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
            for i in range(len(all_stats["H_excess"])):
                if (
                    all_stats["heavy_missing"][i]
                    or all_stats["heavy_excess"][i]
                    or (all_stats["H_excess"][i] and not excess_H_ok)
                    #or len(all_stats["bonded_atoms_missing"][i])
                    or len(all_stats["bonded_atoms_excess"][i])
                ):
                    continue
                passed.append(i)

            if len(passed) == 0:
                template_key = None
                template = None
                mapping = None
                m = f"No template matched for {residue_key=}" + os_linesep
                m += f"tried {len(candidate_templates)} templates for {residue_key=}"
                m += f"{excess_H_ok=}"
                m += os_linesep
                for i in range(len(all_stats["H_excess"])):
                    heavy_miss = all_stats["heavy_missing"][i]
                    heavy_excess = all_stats["heavy_excess"][i]
                    H_excess = all_stats["H_excess"][i]
                    bond_miss = all_stats["bonded_atoms_missing"][i]
                    bond_excess = all_stats["bonded_atoms_excess"][i]
                    tkey = candidate_template_keys[i]
                    m += (
                        f"{tkey:10} {heavy_miss=} {heavy_excess=} {H_excess=} {bond_miss=} {bond_excess=}"
                        + os_linesep
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
                    m = f"for {residue_key=}, {len(passed)} have passed" + os_linesep
                    tkeys = [candidate_template_keys[i] for i in passed]
                    m += f"these are: {tkeys}" + os_linesep
                    m += "and were evaluated for the number of missing H"
                    m += "however there was a tie between"  # TODO
                    logger.error(m)
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
            residues[residue_key] = ChorizoResidue(
                raw_mol,
                rdkit_mol,
                mapping,
                input_resname,
                template_key,
                atom_names,
            )
            residues[residue_key].template = template
            if template is not None and template.link_labels is not None:
                mapping_inv = residues[
                    residue_key
                ].mapidx_from_raw  # {j: i for (i, j) in mapping.items()}
                # TODO check here mapping_inv unnused
                link_labels = {i: label for i, label in template.link_labels.items()}
                residues[residue_key].link_labels = link_labels

        return residues, log

    @staticmethod
    def _build_padded_mols(residues, bonds, padders):
        """

        Parameters
        ----------
        residues
        bonds
        padders

        Returns
        -------

        """
        padded_mols = {}
        bond_use_count = {key: 0 for key in bonds}
        for (
            residue_id,
            residue,
        ) in residues.items():
            if residue.rdkit_mol is None:
                continue
            padded_mol = residue.rdkit_mol
            mapidx_pad = {
                atom.GetIdx(): atom.GetIdx() for atom in padded_mol.GetAtoms()
            }
            for atom_index, link_label in residue.link_labels.items():
                adjacent_rid  = None
                adjacent_mol = None
                adjacent_atom_index = None
                for (r1_id, r2_id), (i1, i2) in bonds.items():
                    if r1_id == residue_id and i1 == atom_index:
                        adjacent_rid = r2_id
                        adjacent_atom_index = i2
                        break
                    elif r2_id == residue_id and i2 == atom_index:
                        adjacent_rid = r1_id
                        adjacent_atom_index = i1
                        break
                
                if adjacent_rid is not None:
                    adjacent_mol = residues[adjacent_rid].rdkit_mol
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
            for atom_index in residue.link_labels:
                heavy_atom = residue.rdkit_mol.GetAtomWithIdx(atom_index)
                for neighbor in heavy_atom.GetNeighbors():
                    if neighbor.GetAtomicNum() != 1:
                        continue
                    if neighbor.GetIdx() in residue.mapidx_to_raw:
                        # index of H exists in mapidx_to_raw, which means that
                        # the raw_input_mol had the hydrogen. Thus, we do not
                        # want to update its coordiantes.
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
                err_msg += (
                    f"expected two paddings for {key} {bonds[key]}, padded {count}"
                    + os_linesep
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
        residue = self.residues[residue_id]
        inv = {j: i for i, j in residue.molsetup_mapidx.items()}
        link_atoms = [inv[i] for i in residue.template.link_labels]
        if len(link_atoms) == 0:
            raise RuntimeError(
                "can't define a sidechain without bonds to other residues"
            )
        # TODO: rewrite this to work better with new MoleculeSetups
        graph = {atom.index: atom.graph for atom in residue.molsetup.atoms}
        for i in range(len(link_atoms) - 1):
            start_node = link_atoms[i]
            end_nodes = [k for (j, k) in enumerate(link_atoms) if j != i]
            backbone_paths = find_graph_paths(graph, start_node, end_nodes)
            for path in backbone_paths:
                for x in range(len(path) - 1):
                    idx1 = min(path[x], path[x + 1])
                    idx2 = max(path[x], path[x + 1])
                    residue.molsetup.bond_info[(idx1, idx2)].rotatable = False
        residue.is_movable = True

        mk_prep.calc_flex(
            residue.molsetup,
            root_atom_index=link_atoms[0],
        )

        molsetup = residue.molsetup
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
                    residue.is_flexres_atom[atom_idx] = True
        return

    @staticmethod
    def print_residues_by_resname(removed_residues):
        """

        Parameters
        ----------
        removed_residues

        Returns
        -------

        """
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
                element = line[76:78].strip()
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
            msg = "each residue key must have exactly 1 resname" + os_linesep
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
        valid_residues = self.get_valid_residues()

        # check that residue IDs passed in new_positions are valid
        unknown_res_ids = set()
        for res_id in new_positions:
            if res_id not in valid_residues:
                unknown_res_ids.add(res_id)
        if unknown_res_ids:
            msg = f"Residue IDs not in valid residues: {unknown_res_ids}"
            raise ValueError(msg)

        pdbout = ""
        atom_count = 0
        pdb_line = "{:6s}{:5d} {:^4s} {:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}                       {:2s} "
        pdb_line += pathlib.os.linesep
        for res_id in self.get_valid_residues():
            rdkit_mol = self.residues[res_id].rdkit_mol
            if res_id in new_positions:
                positions = get_updated_positions(
                    self.residues[res_id],
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
                atom_name = self.residues[res_id].atom_names[i]
                x, y, z = positions[i]
                element = mini_periodic_table[atom.GetAtomicNum()]
                pdbout += pdb_line.format(
                    "ATOM",
                    atom_count,
                    atom_name,
                    self.residues[res_id].input_resname,
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
        for res_id in self.get_valid_residues():
            molsetup = self.residues[res_id].molsetup
            wanted_atom_indices = []
            for atom in molsetup.atoms:
                if not atom.is_ignore and not self.residues[res_id].is_flexres_atom[atom.index]:
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

    # The following functions return filtered dictionaries of residues based on the value of residue flags.
    # region Filtering Residues
    def get_ignored_residues(self):
        return {k: v for k, v in self.residues.items() if v.rdkit_mol is None}

    def get_valid_residues(self):
        return {k: v for k, v in self.residues.items() if v.rdkit_mol is not None}

    # endregion

    def to_json(self):
        pass


def add_rotamers_to_chorizo_molsetups(rotamer_states_list, chorizo):
    """

    Parameters
    ----------
    rotamer_states_list
    chorizo

    Returns
    -------

    """
    rotamer_res_disambiguate = {}
    for (
        primary_res,
        specific_res_list,
    ) in chorizo.residue_chem_templates.ambiguous.items():
        for specific_res in specific_res_list:
            rotamer_res_disambiguate[specific_res] = primary_res

    no_resname_to_resname = {}
    for res_with_resname in chorizo.residues:
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
            if chorizo.residues[res_with_resname].molsetup is None:
                raise RuntimeError(
                    "no molsetup for %s, can't add rotamers" % (res_with_resname)
                )
            # next block is inefficient for large rotamer_states_list
            # refactored chorizos could help by having the following
            # data readily available
            molsetup = chorizo.residues[res_with_resname].molsetup
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
    molsetup: RDKitMoleculeSetup
        An RDKitMoleculeSetup associated with this residue
    molsetup_mapidx: dict (int -> int)
        key: index of atom in padded_mol
        value: index of atom in rdkit_mol
    template: ResidueTemplate
        provides access to link_labels in the template
    """

    def __init__(
        self,
        raw_input_mol,
        rdkit_mol,
        mapidx_to_raw,
        input_resname=None,
        template_key=None,
        atom_names=None,
    ):

        self.raw_rdkit_mol = raw_input_mol
        self.rdkit_mol = rdkit_mol
        self.mapidx_to_raw = mapidx_to_raw
        self.residue_template_key = template_key  # same as pdb_resname except NALA, etc
        self.input_resname = input_resname  # exists even in openmm topology
        self.atom_names = (
            atom_names  # same order as atoms in rdkit_mol, used in rotamers
        )

        # TODO convert link indices/labels in template to rdkit_mol indices herein
        # self.link_labels = {}
        self.template = None

        if mapidx_to_raw is not None:
            self.mapidx_from_raw = {j: i for (i, j) in mapidx_to_raw.items()}
            if len(self.mapidx_from_raw) != len(self.mapidx_to_raw):
                raise RuntimeError(f"index mapping not invertable {mapidx_to_raw=}")
        else:
            self.mapidx_from_raw = None

        self.padded_mol = None
        self.molsetup = None
        self.molsetup_mapidx = None
        self.is_flexres_atom = None  # Check about these data types/Do we want the default to be None or empty
        self.is_movable = False

    def set_atom_names(self, atom_names_list):
        """

        Parameters
        ----------
        atom_names_list

        Returns
        -------

        """
        if self.rdkit_mol is None:
            raise RuntimeError("can't set atom_names if rdkit_mol is not set yet")
        if len(atom_names_list) != self.rdkit_mol.GetNumAtoms():
            raise ValueError(
                f"{len(atom_names_list)=} differs from {self.rdkit_mol.GetNumAtoms()=}"
            )
        name_types = set([type(name) for name in atom_names_list])
        if name_types != {str}:
            raise ValueError(f"atom names must be str but {name_types=}")
        self.atom_names = atom_names_list
        return

    def to_json(self):
        """

        Returns
        -------

        """
        return json.dumps(self, cls=ChorizoResidueEncoder)

    @classmethod
    def from_json(cls, json_string):
        """

        Parameters
        ----------
        json_string

        Returns
        -------

        """
        residue = json.loads(json_string, object_hook=cls.chorizo_residue_json_decoder)
        return residue

class NoAtomMapWarning(logging.Filter):
    def filter(self, record):
        fields = record.getMessage().split()
        a = " ".join(fields[1:4]) == "product atom-mapping number"
        b = " ".join(fields[5:]) == "not found in reactants."
        is_atom_map_warning = a and b
        return not is_atom_map_warning

class ResiduePadder:
    """
    A class for padding RDKit molecules of residues with parts from adjacent residues.

    Attributes
    ----------
    rxn : rdChemReactions.ChemicalReaction
        Reaction SMARTS of a single-reactant, single-product reaction for padding.
    adjacent_smartsmol : Chem.Mol
        SMARTS molecule with mapping numbers to copy atom positions from part of adjacent residue.
    adjacent_smartsmol_mapidx : list
        Mapping for atoms in adjacent_smartsmol, from mapping numbers to atom indicies. 
    """

    # Replacing ResidueConnection by ResiduePadding
    # Why have two ResiduePadding instances per connection between two-residues?
    #  - three-way merge: if three carbons joined in cyclopropare, we can still pad
    #  - defines padding in the reaction for blunt residues
    #  - all bonds will be defined in the input topology after a future refactor

    # reaction should not delete atoms, not even Hs
    # reaction should create bonds at non-real Hs (implicit or explicit rdktt H)

    def __init__(self, rxn_smarts: str, adjacent_res_smarts: str = None): 
        """
        Initialize the ResiduePadder with reaction SMARTS and optional adjacent residue SMARTS.

        Parameters
        ----------
        rxn_smarts: str
            Reaction SMARTS to pad a link atom of a ChorizoResidue molecule.
            Product atoms that are not mapped in the reactants will have
            their coordinates set from an adjacent residue molecule, given
            that adjacent_res_smarts is provided and the atom labels match
            the unmapped product atoms of rxn_smarts.
        adjacent_res_smarts: str
            SMARTS pattern to identify atoms in molecule of adjacent residue
            and copy their positions to padding atoms. The SMARTS atom labels
            must match those of the product atoms of rxn_smarts that are
            unmapped in the reagents.
        """

        # Ensure rxn_smarts has single reactant and single product
        self.rxn = self._validate_rxn_smarts(rxn_smarts)

        # Fill in adjacent_smartsmol_mapidx
        if adjacent_res_smarts is None:
            self.adjacent_smartsmol = None
            self.adjacent_smartsmol_mapidx = None
            return

        # Ensure adjacent_res_smarts is None or a valid SMARTS        
        self.adjacent_smartsmol = self._initialize_adj_smartsmol(adjacent_res_smarts)

        # Ensure the mapping numbers are the same in adjacent_smartsmol and rxn_smarts's product
        self._check_adj_smarts(self.rxn, self.adjacent_smartsmol)

        self.adjacent_smartsmol_mapidx = {
            atom.GetIntProp("molAtomMapNumber"): atom.GetIdx()
            for atom in self.adjacent_smartsmol.GetAtoms()
            if atom.HasProp("molAtomMapNumber")
        }
        return
    
    @staticmethod
    def _validate_rxn_smarts(rxn_smarts: str) -> rdChemReactions.ChemicalReaction:
        """Validate rxn_smarts and return rxn"""
        rxn = rdChemReactions.ReactionFromSmarts(rxn_smarts)
        if rxn.GetNumReactantTemplates() != 1:
            raise ValueError(f"Expected 1 reactant, got {rxn.GetNumReactantTemplates()}.")
        if rxn.GetNumProductTemplates() != 1:
            raise ValueError(f"Expected 1 product, got {rxn.GetNumProductTemplates()}.")
        return rxn
    
    @staticmethod
    def _initialize_adj_smartsmol(adjacent_res_smarts: str) -> Chem.Mol:
        """Validate adjacent_res_smarts and return adjacent_smartsmol"""
        adjacent_smartsmol = Chem.MolFromSmarts(adjacent_res_smarts)
        if adjacent_smartsmol is None:
            raise RuntimeError("Invalid SMARTS pattern in adjacent_res_smarts")
        return adjacent_smartsmol
    
    @staticmethod
    def _check_adj_smarts(rxn: rdChemReactions.ChemicalReaction, adjacent_smartsmol: Chem.Mol):
        """
        Ensure the atom mapping numbers are the same in adjacent_smartsmol and rxn_smarts's product
        """

        # Assumes single reactant, single product
        reactant_ids = get_molAtomMapNumbers(rxn.GetReactantTemplate(0))
        product_ids = get_molAtomMapNumbers(rxn.GetProductTemplate(0))
        adjacent_ids = get_molAtomMapNumbers(adjacent_smartsmol)
        padding_ids = product_ids.difference(reactant_ids)
        is_ok = padding_ids == adjacent_ids

        if not is_ok:
            raise ValueError(f"SMARTS labels in adjacent_smartsmol ({adjacent_ids}) differ from \
                             unmapped product labels in reaction ({padding_ids})")

    def __call__(self, target_mol: Chem.Mol, adjacent_mol = None, 
                 target_required_atom_index = None, adjacent_required_atom_index = None):
        # add Hs only to padding atoms
        # copy coordinates if adjacent res has Hs bound to heavy atoms
        # labels have been checked upstream

        # Ensure target_mol contains self.rxn's reactant
        rxn = self.rxn
        if not self._check_target_mol(target_mol):
            print(f"target_mol ({Chem.MolToSmiles(target_mol)}) is not fully compliant with the template rxn ({rdChemReactions.ReactionToSmarts(self.rxn)})...")
            # Assumes single reactant and single product
            reactant_smartsmol = rxn.GetReactantTemplate(0)
            reactant_ids = get_molAtomMapNumbers(reactant_smartsmol)

            # Generate fallback options for reactants
            fallback_reactant_smartsmol = Chem.MolFromSmarts(rdFMCS.FindMCS([reactant_smartsmol, target_mol]).smartsString)
            if fallback_reactant_smartsmol is None:
                raise RuntimeError(f"There is no common substructure between target_mol and the expected reactant. ")

            # Add mapping number to fallback reactants and filter the fallback options
            # To be accepted, the fallback reactant needs to at least have a match with target_mol
            # containing target_mol's atom with target_required_atom_index
            fallback_reactants = [
                reactant_mol for reactant_mol in apply_atom_mappings(fallback_reactant_smartsmol, reactant_smartsmol)
                if any(target_required_atom_index in match for match in target_mol.GetSubstructMatches(reactant_mol))
            ]
            if len(fallback_reactants) == 0:
                raise RuntimeError(f"The maximum common substructure between target_mol and the expected reactant does not contain the expected linker atom with target_required_atom_index.")
            
            # Take any fallback reactant; actually, they're the same reactant mols having different mapping numbers
            fallback_reactant = fallback_reactants[0]
            
            # Modify rxn smarts and update rxn
            fallback_reactant_ids = get_molAtomMapNumbers(fallback_reactant)
            skipping_ids = reactant_ids.difference(fallback_reactant_ids)
            fallback_product = remove_atoms_with_mapping(rxn.GetProductTemplate(0), skipping_ids)
            fallback_rxnsmarts = f"{Chem.MolToSmarts(fallback_reactant)}>>{Chem.MolToSmarts(fallback_product)}"
            rxn = rdChemReactions.ReactionFromSmarts(fallback_rxnsmarts)
            print(f"Switched from Template rxn ({rdChemReactions.ReactionToSmarts(self.rxn)}) to Fallback rxn ({fallback_rxnsmarts})")
        
        # Get adjacent_mol's reacting part that contains adjacent_required_atom_index
        if adjacent_mol is not None:

            # Ensure adjacent_mol contains expected_adjacent_smartsmol, and 
            # there's exactly one match that includes atom with adjacent_required_atom_index
            if self._check_adjacent_mol(self.adjacent_smartsmol, adjacent_mol, adjacent_required_atom_index):
                adjacent_smartsmol = self.adjacent_smartsmol
            
            # Remove unmapped atoms from Template adjacent mol SMARTS as the fallback option;
            # The unmapped atoms aren't needed for positions anyways
            else:
                print(f"adjacent_mol ({Chem.MolToSmiles(adjacent_mol)}) is not fully compliant with the template adjacent_smarts ({Chem.MolToSmarts(self.adjacent_smartsmol)})...")
                adjacent_smartsmol = remove_unmapped_atoms_from_mol(self.adjacent_smartsmol)

                # Evaluate adjacent mol against the fallback adjacent mol SMARTS
                if self._check_adjacent_mol(adjacent_smartsmol, adjacent_mol, adjacent_required_atom_index):
                     print(f"Switched from Template adjacent mol ({Chem.MolToSmarts(self.adjacent_smartsmol)}) to Fallback adjacent mol ({Chem.MolToSmarts(adjacent_smartsmol)})")
                else:
                    raise RuntimeError(f"adjacent_mol doesn't contain the mapped atoms in adjacent_smartsmol.") 
            
            # Update hit and adjacent_smartsmol_mapidx 
            hit = adjacent_mol.GetSubstructMatches(adjacent_smartsmol)[0]
            adjacent_smartsmol_mapidx = {
                atom.GetIntProp("molAtomMapNumber"): atom.GetIdx()
                for atom in adjacent_smartsmol.GetAtoms() if atom.HasProp("molAtomMapNumber")
                }

        # suppress rdkit warning about product atom map not found in reactants
        # e.g. in "[C:1]>>[C:1][O:2]" label :2 is missing in reactants
        filtr = NoAtomMapWarning()
        rdkit_logger.addFilter(filtr)
        
        # Get padded mol and index map from the rxn
        outcomes = react_and_map((target_mol,), rxn)
        rdkit_logger.removeFilter(filtr)

        # Filter outcomes by target_required_atom_index
        if target_required_atom_index is not None:
            outcomes = [
                (product, index_map)
                for (product, index_map) in outcomes 
                if target_required_atom_index in index_map["atom_idx"] 
            ]

        # Ensure single outcome
        if len(outcomes) == 0:
            raise RuntimeError(f"The padding reaction of target_mol has no outcome that contains the atom with target_required_atom_index")
        elif len(outcomes) > 1:
            raise RuntimeError(f"The padding reaction of target_mol has multiple outcomes that contain the atom with target_required_atom_index")
        padded_mol, idxmap = outcomes[0]

        padding_heavy_atoms = [
            i for i, j in enumerate(idxmap["atom_idx"])
            if j is None and padded_mol.GetAtomWithIdx(i).GetAtomicNum() != 1
        ]
        mapidx = idxmap["atom_idx"]

        # Add Hs to padded_mol and update mapidx
        if adjacent_mol is None:
            padded_mol.UpdatePropertyCache()  # avoids getNumImplicitHs() called without preceding call to calcImplicitValence()
            Chem.SanitizeMol(padded_mol)  # just in case
            padded_h = Chem.AddHs(padded_mol, onlyOnAtoms=padding_heavy_atoms)
            mapidx += [None] * (padded_h.GetNumAtoms() - padded_mol.GetNumAtoms())
        else:
            # Get coordinates of existing atoms
            adjacent_coords = adjacent_mol.GetConformer().GetPositions()
            for atom in adjacent_smartsmol.GetAtoms():
                if not atom.HasProp("molAtomMapNumber"):
                    continue
                j = atom.GetIntProp("molAtomMapNumber")
                k = idxmap["new_atom_label"].index(j)
                l = adjacent_smartsmol_mapidx[j]
                padded_mol.GetConformer().SetAtomPosition(k, adjacent_coords[hit[l]])
            padded_mol.UpdatePropertyCache()  # avoids getNumImplicitHs() called without preceding call to calcImplicitValence()
            Chem.SanitizeMol(padded_mol)  # got crooked Hs without this
            padded_h = Chem.AddHs(
                padded_mol, onlyOnAtoms=padding_heavy_atoms, addCoords=True
            )

        return padded_h, mapidx
    
    @staticmethod
    def _check_adjacent_mol(expected_adjacent_smartsmol: Chem.Mol, adjacent_mol: Chem.Mol, adjacent_required_atom_index: str):
        """
        Ensure adjacent_mol contains expected_adjacent_smartsmol, and 
        there's exactly one match that includes atom with adjacent_required_atom_index
        """
        if expected_adjacent_smartsmol is None:
            raise RuntimeError("adjacent_res_smarts must be initialized to support adjacent_mol.")

        hits = adjacent_mol.GetSubstructMatches(expected_adjacent_smartsmol)
        if adjacent_required_atom_index is not None:
            hits = [hit for hit in hits if adjacent_required_atom_index in hit]
            if len(hits) > 1:
                raise RuntimeError(f"adjacent_mol has multiple matches for adjacent_smartsmol.")  
            elif len(hits) == 0:
                return False
        return True

    def _check_target_mol(self, target_mol: Chem.Mol):
        """Ensure target_mol contains self.rxn's reactant"""
        # Assumes single reactant
        if target_mol.GetSubstructMatches(self.rxn.GetReactantTemplate(0)):
            return True
        else:
            return False

    @classmethod
    def from_json(cls, string):
        d = json.loads(string)
        return cls(**d)
    
    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__)

# Utility Functions

def get_molAtomMapNumbers(mol: Chem.Mol) -> set[int]:
    """Return the set of mapping numbers in a molecule."""
    return {atom.GetIntProp("molAtomMapNumber") for atom in mol.GetAtoms() if atom.HasProp("molAtomMapNumber")}

def remove_unmapped_atoms_from_mol(mol: Chem.Mol) -> Chem.Mol:
    """Remove atoms without mapping numbers from a molecule."""
    atoms_to_remove = [
        atom.GetIdx() for atom in mol.GetAtoms() 
        if not atom.HasProp("molAtomMapNumber")
        ]

    if len(atoms_to_remove) > 0:
        mol = Chem.RWMol(mol)
        for idx in sorted(atoms_to_remove, reverse=True):
            mol.RemoveAtom(idx)
        mol = mol.GetMol()

    return mol

def apply_atom_mappings(mcs_mol: Chem.Mol, original_mol: Chem.Mol) -> list[Chem.Mol]:
    """
    Apply atom mappings from the original molecule to the MCS molecule by substructure match.
    Be prepared for multiple matches, return a list for further evaluation
    """

    # Assumes original_mol contains mcs_mol
    matches = original_mol.GetSubstructMatches(mcs_mol)
    mapped_mcs_molecules = []

    for match in matches:
        rw_mcs_mol = Chem.RWMol(mcs_mol)
        
        for i, mcs_atom in enumerate(rw_mcs_mol.GetAtoms()):
            original_atom_idx = match[i]
            original_atom = original_mol.GetAtomWithIdx(original_atom_idx)
            
            if original_atom.HasProp("molAtomMapNumber"):
                mcs_atom.SetProp("molAtomMapNumber", original_atom.GetProp("molAtomMapNumber"))

        mapped_mcs_molecules.append(rw_mcs_mol.GetMol())
    
    return mapped_mcs_molecules

def remove_atoms_with_mapping(product: Chem.Mol, mapping_numbers: set) -> Chem.Mol:
    """Remove atoms with specific atom mapping numbers from a molecule."""
    editable_product = Chem.RWMol(product)

    atoms_to_remove = []
    for atom in editable_product.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            map_num = int(atom.GetProp("molAtomMapNumber"))
            if map_num in mapping_numbers:
                atoms_to_remove.append(atom.GetIdx())
    for idx in sorted(atoms_to_remove, reverse=True):
        editable_product.RemoveAtom(idx)
    
    return editable_product.GetMol()


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
            raise ValueError(
                f"expected any atom with non-real Hs ({have_implicit_hs}) to be in {link_labels=}"
            )
        if atom_names is None:
            return
        # data_lengths = set([len(values) for (_, values) in data.items()])
        # if len(data_lengths) != 1:
        #    raise ValueError(f"each array in data must have the same length, but got {data_lengths=}")
        # data_length = data_lengths.pop()
        if len(atom_names) != mol.GetNumAtoms():
            raise ValueError(f"{len(atom_names)=} differs from {mol.GetNumAtoms()=}")
        return

    def match(self, input_mol):
        mapping = mapping_by_mcs(self.mol, input_mol)
        mapping_inv = {value: key for (key, value) in mapping.items()}
        if len(mapping_inv) != len(mapping):
            raise RuntimeError(
                f"bug in atom indices, repeated value different keys? {mapping=}"
            )
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

def rdkit_or_none_to_json(rdkit_mol):
    if rdkit_mol is None:
        return None
    return rdMolInterchange.MolToJSON(rdkit_mol)


# region JSON Encoders
class ChorizoResidueEncoder(json.JSONEncoder):
    """
    JSON Encoder class for Chorizo Residue objects.
    """

    molecule_setup_encoder = MoleculeSetupEncoder()

    def default(self, obj):
        """
        Overrides the default JSON encoder for data structures for ChorizoResidue objects.

        Parameters
        ----------
        obj: object
            Can take any object as input, but will only create the ChorizoResidue JSON format for ChorizoResidue objects.
            For all other objects will return the default json encoding.

        Returns
        -------
        A JSON serializable object that represents the ChorizoResidue class or the default JSONEncoder output for an
        object.
        """
        if isinstance(obj, ChorizoResidue):
            if obj.molsetup is None:
                molsetup_json = None
            else:
                molsetup_json = self.molecule_setup_encoder.default(obj.molsetup)
            return {
                "raw_rdkit_mol": rdkit_or_none_to_json(obj.raw_rdkit_mol),
                "rdkit_mol": rdkit_or_none_to_json(obj.rdkit_mol),
                "mapidx_to_raw": obj.mapidx_to_raw,
                "residue_template_key": obj.residue_template_key,
                "input_resname": obj.input_resname,
                "atom_names": obj.atom_names,
                "mapidx_from_raw": obj.mapidx_from_raw,
                "padded_mol": rdkit_or_none_to_json(obj.padded_mol),
                "molsetup": molsetup_json,
                "is_flexres_atom": obj.is_flexres_atom,
                "is_movable": obj.is_movable,
                "molsetup_mapidx": obj.molsetup_mapidx,
            }
        return json.JSONEncoder.default(self, obj)


class ResidueTemplateEncoder(json.JSONEncoder):
    """
    JSON Encoder class for ResidueTemplate objects.
    """

    def default(self, obj):
        """
        Overrides the default JSON encoder for data structures for ResidueTemplate objects.

        Parameters
        ----------
        obj: object
            Can take any object as input, but will only create the ResidueTemplate JSON format for ResidueTemplate
            objects. For all other objects will return the default json encoding.

        Returns
        -------
        A JSON serializable object that represents the ResidueTemplate class or the default JSONEncoder output for an
        object.
        """
        if isinstance(obj, ResidueTemplate):
            output_dict = {
                "mol": rdMolInterchange.MolToJSON(obj.mol),
                "link_labels": obj.link_labels,
                "atom_names": obj.atom_names,
            }
            return output_dict
        return json.JSONEncoder.default(self, obj)


class ResiduePadderEncoder(json.JSONEncoder):
    """
    JSON Encoder class for ResiduePadder objects.
    """

    def default(self, obj):
        """
        Overrides the default JSON encoder for data structures for ResiduePadder objects.

        Parameters
        ----------
        obj: object
            Can take any object as input, but will only create the ResiduePadder JSON format for ResiduePadder
            objects. For all other objects will return the default json encoding.

        Returns
        -------
        A JSON serializable object that represents the ResiduePadder class or the default JSONEncoder output for an
        object.
        """
        if isinstance(obj, ResiduePadder):
            if obj.adjacent_smartsmol is None:
                adjacent_smarts = None
            else:
                # do not use JSON because it looses atom labels
                adjacent_smarts = Chem.MolToSmarts(obj.adjacent_smartsmol)
            output_dict = {
                "rxn_smarts": rdChemReactions.ReactionToSmarts(obj.rxn),
                "adjacent_smarts": adjacent_smarts,
            }
            # we are not serializing the adjacent_smartsmol_mapidx as that will
            # be rebuilt by the ResiduePadder init
            return output_dict
        return json.JSONEncoder.default(self, obj)


class ResidueChemTemplatesEncoder(json.JSONEncoder):
    """
    JSON Encoder class for ResidueChemTemplates objects.
    """

    residue_padder_encoder = ResiduePadderEncoder()
    residue_template_encoder = ResidueTemplateEncoder()

    def default(self, obj):
        """
        Overrides the default JSON encoder for data structures for ResidueChemTemplates objects.

        Parameters
        ----------
        obj: object
            Can take any object as input, but will only create the ResidueChemTemplates JSON format for
            ResidueChemTemplates objects. For all other objects will return the default json encoding.

        Returns
        -------
        A JSON serializable object that represents the ResidueChemTemplates class or the default JSONEncoder output for
        an object.
        """
        if isinstance(obj, ResidueChemTemplates):
            output_dict = {
                "residue_templates": {
                    k: self.residue_template_encoder.default(v)
                    for k, v in obj.residue_templates.items()
                },
                "ambiguous": obj.ambiguous,
                "padders": {
                    k: self.residue_padder_encoder.default(v)
                    for k, v in obj.padders.items()
                },
            }
            return output_dict
        return json.JSONEncoder.default(self, obj)


class LinkedRDKitChorizoEncoder(json.JSONEncoder):
    """
    JSON Encoder class for LinkedRDKitChorizo objects.
    """

    residue_chem_templates_encoder = ResidueChemTemplatesEncoder()
    chorizo_residue_encoder = ChorizoResidueEncoder()

    def default(self, obj):
        """
        Overrides the default JSON encoder for data structures for LinkedRDKitChorizo objects.

        Parameters
        ----------
        obj: object
            Can take any object as input, but will only create the LinkedRDKitChorizo JSON format for LinkedRDKitChorizo
            objects. For all other objects will return the default json encoding.

        Returns
        -------
        A JSON serializable object that represents the LinkedRDKitChorizo class or the default JSONEncoder output for an
        object.
        """
        if isinstance(obj, LinkedRDKitChorizo):
            output_dict = {
                "residue_chem_templates": self.residue_chem_templates_encoder.default(
                    obj.residue_chem_templates
                ),
                "residues": {
                    k: self.chorizo_residue_encoder.default(v)
                    for k, v in obj.residues.items()
                },
                "log": obj.log,
            }
            return output_dict
        return json.JSONEncoder.default(self, obj)


# endregion

# region JSON Decoders


def chorizo_residue_json_decoder(obj: dict):
    """
    Takes an object and attempts to decode it into a ChorizoResidue object.

    Parameters
    ----------
    obj: Object
        This can be any object, but it should be a dictionary generated by deserializing a JSON of a ChorizoResidue
        object.

    Returns
    -------
    If the input is a dictionary corresponding to a ChorizoResidue, will return a ChorizoResidue with data
    populated from the dictionary. Otherwise, returns the input object.

    """
    # if the input object is not a dict, we know that it will not be parsable and is unlikely to be usable or
    # safe data, so we should ignore it.
    if type(obj) is not dict:
        return obj

    # check that all the keys we expect are in the object dictionary as a safety measure
    expected_residue_keys = {
        "raw_rdkit_mol",
        "rdkit_mol",
        "mapidx_to_raw",
        "residue_template_key",
        "input_resname",
        "atom_names",
        "mapidx_from_raw",
        "padded_mol",
        "molsetup",
        "is_flexres_atom",
        "is_movable",
        "molsetup_mapidx",
    }

    if set(obj.keys()) != expected_residue_keys:
        return obj
    # Extracts init mols for ChorizoResidue:
    raw_rdkit_mol = rdkit_mol_from_json(obj["raw_rdkit_mol"])
    rdkit_mol = rdkit_mol_from_json(obj["rdkit_mol"])
    if obj["mapidx_to_raw"] is None:
        mapidx_to_raw = None
    else:
        mapidx_to_raw = {int(k): v for k, v in obj["mapidx_to_raw"].items()}

    residue = ChorizoResidue(raw_rdkit_mol, rdkit_mol, mapidx_to_raw)

    # sets remaining properties from JSON
    residue.residue_template_key = obj["residue_template_key"]
    residue.input_resname = obj["input_resname"]
    residue.atom_names = obj["atom_names"]

    if obj["mapidx_from_raw"] is None:
        residue.mapidx_from_raw = None
    else:
        residue.mapidx_from_raw = {int(k): v for k, v in obj["mapidx_from_raw"].items()}

    residue.padded_mol = rdkit_mol_from_json(obj["padded_mol"])
    residue.molsetup = RDKitMoleculeSetup.from_json(obj["molsetup"])
    if obj["molsetup_mapidx"] is None:
        residue.molsetup_mapidx = None
    else:
        residue.molsetup_mapidx = {int(k): v for k, v in obj["molsetup_mapidx"].items()}

    # boolean values
    residue.is_flexres_atom = obj["is_flexres_atom"]
    residue.is_movable = obj["is_movable"]

    return residue


def residue_template_json_decoder(obj: dict):
    """
    Takes an object and attempts to decode it into a ResidueTemplate object.

    Parameters
    ----------
    obj: Object
        This can be any object, but it should be a dictionary constructed by deserializing the JSON representation of a
        ResidueTemplate object.

    Returns
    -------
    If the input is a dictionary corresponding to a ResidueTemplate, will return a ResidueTemplate with data populated
    from the dictionary. Otherwise, returns the input object.
    """
    # if the input object is not a dict, we know that it will not be parsable and is unlikely to be usable or
    # safe data, so we should ignore it.
    if type(obj) is not dict:
        return obj

    # check that all the keys we expect are in the object dictionary as a safety measure
    expected_residue_keys = {"mol", "link_labels", "atom_names"}
    if set(obj.keys()) != expected_residue_keys:
        return obj

    # Converting ResidueTemplate init values that need conversion
    deserialized_mol = rdkit_mol_from_json(obj["mol"])
    # do not write canonical smiles to preserve original atom order
    mol_smiles = rdkit.Chem.MolToSmiles(deserialized_mol, canonical=False)
    link_labels = {int(k): v for k, v in obj["link_labels"].items()}

    # Construct a ResidueTemplate object
    residue_template = ResidueTemplate(mol_smiles, None, obj["atom_names"])
    # Separately ensure that link_labels is restored to the value we expect it to be so there are not errors in
    # the constructor
    residue_template.link_labels = link_labels

    return residue_template


def residue_padder_json_decoder(obj: dict):
    """
    Takes an object and attempts to decode it into a ResiduePadder object.

    Parameters
    ----------
    obj: Object
        This can be any object, but it should be a dictionary constructed by deserializing the JSON representation of a
        ResiduePadder object.

    Returns
    -------
    If the input is a dictionary corresponding to a ResiduePadder, will return a ResiduePadder with data populated
    from the dictionary. Otherwise, returns the input object.
    """
    # if the input object is not a dict, we know that it will not be parsable and is unlikely to be usable or
    # safe data, so we should ignore it.
    if type(obj) is not dict:
        return obj

    # check that all the keys we expect are in the object dictionary as a safety measure
    expected_residue_keys = {
        "rxn_smarts",
        "adjacent_smarts",
    }
    if set(obj.keys()) != expected_residue_keys:
        return obj

    # Constructs a ResiduePadder object and restores the expected attributes
    # adjacent_smartsmol_mapidx is rebuilt by ResiduePadder init
    residue_padder = ResiduePadder(obj["rxn_smarts"], obj["adjacent_smarts"])

    return residue_padder


def residue_chem_templates_json_decoder(obj: dict):
    """
    Takes an object and attempts to decode it into a ResiduePadder object.

    Parameters
    ----------
    obj: Object
        This can be any object, but it should be a dictionary constructed by deserializing the JSON representation of a
        ResiduePadder object.

    Returns
    -------
    If the input is a dictionary corresponding to a ResiduePadder, will return a ResiduePadder with data populated
    from the dictionary. Otherwise, returns the input object.
    """
    # if the input object is not a dict, we know that it will not be parsable and is unlikely to be usable or
    # safe data, so we should ignore it.
    if type(obj) is not dict:
        return obj

    # Check that all the keys we expect are in the object dictionary as a safety measure
    expected_residue_keys = {
        "residue_templates",
        "ambiguous",
        "padders",
    }
    if set(obj.keys()) != expected_residue_keys:
        return obj

    # Extracting the constructor args from the json representation and creating a ResidueChemTemplates instance
    templates = {
        k: residue_template_json_decoder(v) for k, v in obj["residue_templates"].items()
    }
    padders = {k: residue_padder_json_decoder(v) for k, v in obj["padders"].items()}

    residue_chem_templates = ResidueChemTemplates(templates, padders, obj["ambiguous"])

    return residue_chem_templates


def linked_rdkit_chorizo_json_decoder(obj: dict):
    """
    Takes an object and attempts to deserialize it into a LinkedRDKitChorizo object.

    Parameters
    ----------
    obj: Object
        This can be any object, but it should be a dictionary constructed by deserializing the JSON representation of a
        LinkedRDKitChorizo object.

    Returns
    -------
    If the input is a dictionary corresponding to a LinkedRDKitChorizo, will return a LinkedRDKitChorizo with data
    populated from the dictionary. Otherwise, returns the input object.
    """
    # if the input object is not a dict, we know that it will not be parsable and is unlikely to be usable or
    # safe data, so we should ignore it.
    if type(obj) is not dict:
        return obj

    # Check that all the keys we expect are in the object dictionary as a safety measure
    expected_json_keys = {
        "residue_chem_templates",
        "residues",
        "log",
    }
    if set(obj.keys()) != expected_json_keys:
        return obj

    # Deserializes ResidueChemTemplates from the dict to use as an input, then constructs a LinkedRDKit Chorizo object
    # and sets its values using deserialized JSON values.
    residue_chem_templates = residue_chem_templates_json_decoder(
        obj["residue_chem_templates"]
    )

    linked_rdkit_chorizo = LinkedRDKitChorizo({}, {}, residue_chem_templates)
    linked_rdkit_chorizo.residues = {
        k: chorizo_residue_json_decoder(v) for k, v in obj["residues"].items()
    }
    linked_rdkit_chorizo.log = obj["log"]

    return linked_rdkit_chorizo
# endregion
