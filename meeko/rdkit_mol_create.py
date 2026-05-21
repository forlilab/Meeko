#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko
#


from io import StringIO
import json

from rdkit import Chem
from rdkit.Geometry import Point3D

from meeko.utils.rdkitutils import set_h_isotope_atom_coords


def clean_extend(existing_dict, new_row):
    nr_rows = []
    for key in existing_dict:
        nr_rows.append(len(existing_dict[key]))
        if key not in new_row:
            existing_dict[key].append(None)
    if len(nr_rows) == 0:  # existing_dict is empty
        nr_rows = 0
    elif len(set(nr_rows)) != 1:
        msg = "existing_dict has different nr of items for different attributes"
        raise ValueError(msg)
    else:
        nr_rows = set(nr_rows).pop()
    for key, value in new_row.items():
        if key not in existing_dict:
            existing_dict[key] = [None] * nr_rows
        existing_dict[key].append(value)


# ---------------------------------------------------------------------------
# Module-level data: flexible-residue SMILES templates
# ---------------------------------------------------------------------------

AMBIGUOUS_FLEXRES_CHOICES = {
    "HIS": ["HIE", "HID", "HIP"],
    "ASP": ["ASP", "ASH"],
    "GLU": ["GLU", "GLH"],
    "CYS": ["CYS", "CYM"],
    "LYS": ["LYS", "LYN"],
    "ARG": ["ARG", "ARG_mgltools"],
    "ASN": ["ASN", "ASN_mgltools"],
    "GLN": ["GLN", "GLN_mgltools"],
}

FLEXRES = {
    "CYS": {
        "smiles": "CCS",
        "atom_names_in_smiles_order": ["CA", "CB", "SG"],
        "h_to_parent_index": {"HG": 2},
    },
    "CYM": {
        "smiles": "CC[S-]",
        "atom_names_in_smiles_order": ["CA", "CB", "SG"],
        "h_to_parent_index": {},
    },
    "ASP": {
        "smiles": "CCC(=O)[O-]",
        "atom_names_in_smiles_order": ["CA", "CB", "CG", "OD1", "OD2"],
        "h_to_parent_index": {},
    },
    "ASH": {
        "smiles": "CCC(=O)O",
        "atom_names_in_smiles_order": ["CA", "CB", "CG", "OD1", "OD2"],
        "h_to_parent_index": {"HD2": 4},
    },
    "GLU": {
        "smiles": "CCCC(=O)[O-]",
        "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD", "OE1", "OE2"],
        "h_to_parent_index": {},
    },
    "GLH": {
        "smiles": "CCCC(=O)O",
        "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD", "OE1", "OE2"],
        "h_to_parent_index": {"HE2": 5},
    },
    "PHE": {
        "smiles": "CCc1ccccc1",
        "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD1", "CE1", "CZ", "CE2", "CD2"],
        "h_to_parent_index": {},
    },
    "HIE": {
        "smiles": "CCc1c[nH]cn1",
        "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD2", "NE2", "CE1", "ND1"],
        "h_to_parent_index": {"HE2": 4},
    },
    "HID": {
        "smiles": "CCc1cnc[nH]1",
        "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD2", "NE2", "CE1", "ND1"],
        "h_to_parent_index": {"HD1": 6},
    },
    "HIP": {
        "smiles": "CCc1c[nH+]c[nH]1",
        "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD2", "NE2", "CE1", "ND1"],
        "h_to_parent_index": {"HE2": 4, "HD1": 6},
    },
    "ILE": {
        "smiles": "CC(C)CC",
        "atom_names_in_smiles_order": ["CA", "CB", "CG2", "CG1", "CD1"],
        "h_to_parent_index": {},
    },
    "LYS": {
        "smiles": "CCCCC[NH3+]",
        "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD", "CE", "NZ"],
        "h_to_parent_index": {"HZ1": 5, "HZ2": 5, "HZ3": 5},
    },
    "LYN": {
        "smiles": "CCCCCN",
        "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD", "CE", "NZ"],
        "h_to_parent_index": {"HZ2": 5, "HZ3": 5},
    },
    "LEU": {
        "smiles": "CCC(C)C",
        "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD1", "CD2"],
        "h_to_parent_index": {},
    },
    "MET": {
        "smiles": "CCCSC",
        "atom_names_in_smiles_order": ["CA", "CB", "CG", "SD", "CE"],
        "h_to_parent_index": {},
    },
    "ASN": {
        "smiles": "CCC(=O)N",
        "atom_names_in_smiles_order": ["CA", "CB", "CG", "OD1", "ND2"],
        "h_to_parent_index": {"HD21": 4, "HD22": 4},
    },
    "ASN_mgltools": {
        "smiles": "CCC(=O)N",
        "atom_names_in_smiles_order": ["CA", "CB", "CG", "OD1", "ND2"],
        "h_to_parent_index": {"1HD2": 4, "2HD2": 4},
    },
    "GLN": {
        "smiles": "CCCC(=O)N",
        "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD", "OE1", "NE2"],
        "h_to_parent_index": {"HE21": 5, "HE22": 5},
    },
    "GLN_mgltools": {
        "smiles": "CCCC(=O)N",
        "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD", "OE1", "NE2"],
        "h_to_parent_index": {"1HE2": 5, "2HE2": 5},
    },
    "ARG": {
        "smiles": "CCCCNC(N)=[NH2+]",
        "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
        "h_to_parent_index": {"HE": 4, "HH11": 6, "HH12": 6, "HH21": 7, "HH22": 7},
    },
    "ARG_mgltools": {
        "smiles": "CCCCNC(N)=[NH2+]",
        "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
        "h_to_parent_index": {"HE": 4, "1HH1": 6, "2HH1": 6, "1HH2": 7, "2HH2": 7},
    },
    "SER": {
        "smiles": "CCO",
        "atom_names_in_smiles_order": ["CA", "CB", "OG"],
        "h_to_parent_index": {"HG": 2},
    },
    "THR": {
        "smiles": "CC(C)O",
        "atom_names_in_smiles_order": ["CA", "CB", "CG2", "OG1"],
        "h_to_parent_index": {"HG1": 3},
    },
    "VAL": {
        "smiles": "CC(C)C",
        "atom_names_in_smiles_order": ["CA", "CB", "CG1", "CG2"],
        "h_to_parent_index": {},
    },
    "TRP": {
        "smiles": "CCc1c[nH]c2c1cccc2",
        "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD1", "NE1", "CE2", "CD2", "CE3", "CZ3", "CH2", "CZ2"],
        "h_to_parent_index": {"HE1": 4},
    },
    "TYR": {
        "smiles": "CCc1ccc(cc1)O",
        "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD1", "CE1", "CZ", "CE2", "CD2", "OH"],
        "h_to_parent_index": {"HH": 8},
    },
}


# ---------------------------------------------------------------------------
# Module-level functions (the actual implementations)
# ---------------------------------------------------------------------------

def from_pdbqt_mol(
    pdbqt_mol,
    only_cluster_leads=False,
    keep_flexres=False,
    only_hs_with_coords=False,
):
    if only_cluster_leads and len(pdbqt_mol._pose_data["cluster_leads_sorted"]) == 0:
        raise RuntimeError("no cluster_leads in pdbqt_mol but only_cluster_leads=True")
    mol_list = []
    for mol_index in pdbqt_mol._atom_annotations["mol_index"]:
        flexres_id = pdbqt_mol._pose_data["mol_index_to_flexible_residue"][mol_index]
        if flexres_id is not None and not keep_flexres:
            continue
        smiles = pdbqt_mol._pose_data["smiles"][mol_index]
        index_map = pdbqt_mol._pose_data["smiles_index_map"][mol_index]
        h_parent = pdbqt_mol._pose_data["smiles_h_parent"][mol_index]
        atom_idx = pdbqt_mol._atom_annotations["mol_index"][mol_index]
        atom_is_flex = [
            i in pdbqt_mol._atom_annotations["flexible_residue"] for i in atom_idx
        ]
        if any(atom_is_flex) and all(atom_is_flex):
            is_sidechain = True
        elif any(atom_is_flex):
            raise ValueError(
                "some (but not all!) atoms of a ligand were parsed as sidechain"
            )
        else:
            is_sidechain = False

        if smiles is None:  # probably a flexible sidechain
            residue_names = set()
            atom_names = []
            for atom in pdbqt_mol.atoms(atom_idx):
                residue_names.add(atom[4])
                atom_names.append(atom[2])
            if len(residue_names) == 1:
                resname = residue_names.pop()
                smiles, index_map, h_parent = guess_flexres_smiles(resname, atom_names)
                if smiles is None:
                    mol_list.append(None)
                    continue

        if only_cluster_leads:
            pose_ids = pdbqt_mol._pose_data["cluster_leads_sorted"]
        else:
            pose_ids = range(pdbqt_mol._pose_data["n_poses"])

        mol = Chem.MolFromSmiles(smiles)
        mol.SetProp("meeko", json.dumps({"is_sidechain": is_sidechain}))
        coordinates_all_poses = []
        for i in pose_ids:
            pdbqt_mol._current_pose = i
            coordinates = pdbqt_mol.positions(atom_idx)
            mol = add_pose_to_mol(mol, coordinates, index_map)
            coordinates_all_poses.append(coordinates)

        mol = add_hydrogens(mol, coordinates_all_poses, h_parent, only_hs_with_coords)
        mol_list.append(mol)
    return mol_list


def guess_flexres_smiles(resname, atom_names):
    """Determine a SMILES string for a flexres based on atom names.

    Returns (smiles, index_map, h_parent) — or (None, None, None) on failure.
    See the original docstring for the meaning of each return value.
    """
    if len(set(atom_names)) != len(atom_names):
        return None, None, None
    candidate_resnames = AMBIGUOUS_FLEXRES_CHOICES.get(resname, [resname])
    for resname in candidate_resnames:
        is_match = False
        if resname not in FLEXRES:
            continue
        atom_names_in_smiles_order = FLEXRES[resname]["atom_names_in_smiles_order"]
        h_to_parent_index = FLEXRES[resname]["h_to_parent_index"]
        expected_names = atom_names_in_smiles_order + list(h_to_parent_index.keys())
        if len(atom_names) != len(expected_names):
            continue
        nr_matched_atom_names = sum([int(n in atom_names) for n in expected_names])
        if nr_matched_atom_names == len(expected_names):
            is_match = True
            break
    if not is_match:
        return None, None, None
    smiles = FLEXRES[resname]["smiles"]
    index_map = []
    for smiles_index, name in enumerate(atom_names_in_smiles_order):
        index_map.append(smiles_index + 1)
        index_map.append(atom_names.index(name) + 1)
    h_parent = []
    for name, smiles_index in h_to_parent_index.items():
        h_parent.append(smiles_index + 1)
        h_parent.append(atom_names.index(name) + 1)
    return smiles, index_map, h_parent


def add_pose_to_mol(mol, ligand_coordinates, index_map):
    """Add given coordinates to a molecule as a new conformer.

    index_map maps order of coordinates to order in the SMILES string used to
    generate the RDKit mol.
    """
    n_atoms = mol.GetNumAtoms()
    n_mappings = int(len(index_map) / 2)
    conf = Chem.Conformer(n_atoms)
    if n_atoms < n_mappings:
        raise RuntimeError(
            "Number of atom is rdmol {n_atoms} mismatches"
            "number of pairs in index map {n_at}!".format(
                n_coords=n_atoms, n_at=n_mappings
            )
        )
    coord_is_set = [False] * n_atoms
    for i in range(n_mappings):
        pdbqt_index = int(index_map[i * 2 + 1]) - 1
        mol_index = int(index_map[i * 2]) - 1
        x, y, z = [float(coord) for coord in ligand_coordinates[pdbqt_index]]
        conf.SetAtomPosition(mol_index, Point3D(x, y, z))
        coord_is_set[mol_index] = True

    h_isotope_pos_assignment = set_h_isotope_atom_coords(mol, conf=conf)
    if h_isotope_pos_assignment:
        for idx in h_isotope_pos_assignment:
            conf.SetAtomPosition(idx, h_isotope_pos_assignment[idx])
            coord_is_set[idx] = True
    mol.AddConformer(conf, assignId=True)

    for i, is_set in enumerate(coord_is_set):
        if not is_set:
            raise RuntimeError(
                f"Unable to set position for atom # {i} from docked pose in the created RDKit mol. "
            )

    return mol


def add_hydrogens(mol, coordinates_list, h_parent, only_hs_with_coords):
    """Add hydrogens and adjust polar-H positions to match PDBQT."""
    nr_atoms_before_add_hs = mol.GetNumAtoms()
    mol = Chem.AddHs(mol, addCoords=True)
    conformers = list(mol.GetConformers())
    num_hydrogens = int(len(h_parent) / 2)
    for conformer_idx, atom_coordinates in enumerate(coordinates_list):
        conf = conformers[conformer_idx]
        used_h = []
        for i in range(num_hydrogens):
            parent_rdkit_index = h_parent[2 * i] - 1
            h_pdbqt_index = h_parent[2 * i + 1] - 1
            x, y, z = [float(coord) for coord in atom_coordinates[h_pdbqt_index]]
            parent_atom = mol.GetAtomWithIdx(parent_rdkit_index)
            candidate_hydrogens = [
                atom.GetIdx()
                for atom in parent_atom.GetNeighbors()
                if atom.GetAtomicNum() == 1
            ]
            for h_rdkit_index in candidate_hydrogens:
                if h_rdkit_index not in used_h:
                    break
            used_h.append(h_rdkit_index)
            conf.SetAtomPosition(h_rdkit_index, Point3D(x, y, z))
    if only_hs_with_coords:
        with Chem.RWMol(mol) as rwmol:
            for idx in range(nr_atoms_before_add_hs, mol.GetNumAtoms()):
                atom = rwmol.GetAtomWithIdx(idx)
                if atom.GetAtomicNum() != 1 or idx in used_h:
                    continue
                rwmol.RemoveAtom(idx)
        Chem.SanitizeMol(rwmol)
        mol = rwmol.GetMol()
    return mol


def combine_rdkit_mols(mol_list):
    """Combine a list of RDKit molecules into one; None entries are ignored."""
    combined_mol = None
    props = {}
    for mol in mol_list:
        if mol is None:
            continue
        if mol.HasProp("meeko"):
            data = json.loads(mol.GetProp("meeko"))
            clean_extend(props, data)
        if combined_mol is None:
            combined_mol = mol
        else:
            combined_mol = Chem.CombineMols(combined_mol, mol)
    if len(props) > 0:
        combined_mol.SetProp("meeko", json.dumps(props))
    return combined_mol


def _verify_flexres():
    for resname in FLEXRES:
        atom_names_in_smiles_order = FLEXRES[resname]["atom_names_in_smiles_order"]
        h_to_parent_index = FLEXRES[resname]["h_to_parent_index"]
        expected_names = atom_names_in_smiles_order + list(h_to_parent_index.keys())
        if len(expected_names) != len(set(expected_names)):
            raise RuntimeError("repeated atom names in FLEXRES[%s]" % resname)


def write_sd_string(
    pdbqt_mol,
    only_cluster_leads=False,
    keep_flexres=False,
    only_hs_with_coords=False,
):
    sio = StringIO()
    f = Chem.SDWriter(sio)
    mol_list = from_pdbqt_mol(
        pdbqt_mol, only_cluster_leads, keep_flexres, only_hs_with_coords
    )
    failures = [i for i, mol in enumerate(mol_list) if mol is None]
    combined_mol = combine_rdkit_mols(mol_list)
    if combined_mol is None:
        return "", failures
    keys_map_mol_to_pdbqt = {
        "free_energy": "free_energies",
        "intermolecular_energy": "intermolecular_energies",
        "internal_energy": "internal_energies",
        "cluster_size": "cluster_size",
        "cluster_id": "cluster_id",
        "rank_in_cluster": "rank_in_cluster",
    }
    nr_poses = pdbqt_mol._pose_data["n_poses"]
    if only_cluster_leads:
        pose_idxs = pdbqt_mol._pose_data["cluster_leads_sorted"]
    else:
        pose_idxs = list(range(nr_poses))

    available_properties = {}
    for key_in_mol, key_in_pdbqt in keys_map_mol_to_pdbqt.items():
        if len(pdbqt_mol._pose_data[key_in_pdbqt]) == nr_poses:
            available_properties[key_in_mol] = key_in_pdbqt
    mol_level_data = json.loads(combined_mol.GetProp("meeko"))
    if pdbqt_mol.name is not None:
        combined_mol.SetProp("_Name", pdbqt_mol.name)
    for conformer in combined_mol.GetConformers():
        i = conformer.GetId()
        j = pose_idxs[i]
        conformer_data = json.loads(json.dumps(mol_level_data))
        for (key_in_mol, key_in_pdbqt) in available_properties.items():
            if key_in_mol in conformer_data:
                msg = (
                    "key %s conflict between combined_mol and write_sd_string"
                    % key_in_mol
                )
                raise NotImplementedError(msg)
            conformer_data[key_in_mol] = pdbqt_mol._pose_data[key_in_pdbqt][j]
        if len(conformer_data):
            combined_mol.SetProp("meeko", json.dumps(conformer_data))
        f.write(combined_mol, i)
    f.close()
    output_string = sio.getvalue()
    return output_string, failures


# ---------------------------------------------------------------------------
# Thin shim: preserves external ``RDKitMolCreate.x(...)`` callers
# ---------------------------------------------------------------------------

class RDKitMolCreate:
    """Backward-compat shim. Prefer the module-level functions for new code."""

    ambiguous_flexres_choices = AMBIGUOUS_FLEXRES_CHOICES
    flexres = FLEXRES

    from_pdbqt_mol = staticmethod(from_pdbqt_mol)
    guess_flexres_smiles = staticmethod(guess_flexres_smiles)
    add_pose_to_mol = staticmethod(add_pose_to_mol)
    add_hydrogens = staticmethod(add_hydrogens)
    combine_rdkit_mols = staticmethod(combine_rdkit_mols)
    _verify_flexres = staticmethod(_verify_flexres)
    write_sd_string = staticmethod(write_sd_string)
