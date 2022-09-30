#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko
#


from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import AllChem
import json
import os
import warnings


class RDKitMolCreate:

    ambiguous_flexres_choices = {
        "HIS": ["HIE", "HID", "HIP"],
        "ASP": ["ASP", "ASH"],
        "GLU": ["GLU", "GLH"],
        "CYS": ["CYS", "CYM"],
        "LYS": ["LYS", "LYN"],
    }

    flexres = {
        # "CYS": {
        # },
        # "CYM": {
        # },
        # "ASP": {},
        # "ASH": {},
        # "GLU": {},
        # "GLH": {},
        "HIE" : {
            "smiles": "CCc1c[nH]cn1",
            "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD2", "NE2", "CE1", "ND1"],
            "h_to_parent_index": {"HE2": 4},
        },
        # "HID": {},
        # "HIP": {},
        # "ILE": {},
        # "LYS": {},
        # "LYN": {},
        # "MET": {},
        "ASN": {
            "smiles": "CCC(=O)N",
            "atom_names_in_smiles_order": ["CA", "CB", "CG", "OD1", "ND2"],
            "h_to_parent_index": {"1HD2": 4, "2HD2": 4},
        },
        "PHE": {
            "smiles": "CCc1ccccc1",
            "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD1", "CE1", "CZ", "CE2", "CD2"],
            "h_to_parent_index": {},
        },
        # "GLN": {},
        # "ARG": {},
        # "SER": {},
        # "THR": {},
        # "VAL": {},
        # "TRP": {},
        # "TYR": {},
    }

    @classmethod
    def from_pdbqt_mol(cls, pdbqt_mol): # TODO add pseudo-water (W atoms, variable nr each pose)
        mol_list = []
        for mol_index in pdbqt_mol._atom_annotations["mol_index"]:
            smiles = pdbqt_mol._pose_data['smiles'][mol_index]
            index_map = pdbqt_mol._pose_data['smiles_index_map'][mol_index]
            h_parent = pdbqt_mol._pose_data['smiles_h_parent'][mol_index]
            atom_idx = pdbqt_mol._atom_annotations["mol_index"][mol_index]

            if smiles is None: # probably a flexible sidechain, but can be another ligand
                residue_names = []
                atom_names = []
                for atom in pdbqt_mol.atoms(atom_idx):
                    residue_names.append(atom[4])
                    atom_names.append(atom[2])
                smiles, index_map, h_parent = cls.guess_flexres_smiles(residue_names, atom_names)
                if smiles is None: # failed guessing smiles for possible flexres
                    mol_list.append(None)
                    continue

            mol = Chem.MolFromSmiles(smiles)

            coordinates_all_poses = []
            i = 0
            for pose in pdbqt_mol:
                i += 1
                coordinates = pose.positions(atom_idx)
                mol = cls.add_pose_to_mol(mol, coordinates, index_map) 
                coordinates_all_poses.append(coordinates) 

            # add Hs only after all poses are added as conformers
            # because Chem.AddHs() will affect all conformers at once 
            mol = cls.add_hydrogens(mol, coordinates_all_poses, h_parent) 

            mol_list.append(mol)
        return mol_list

    @classmethod
    def guess_flexres_smiles(cls, residue_names, atom_names):
        if len(set(residue_names)) != 1:
            return None, None, None
        if len(set(atom_names)) != len(atom_names):
            return None, None, None
        resname = set(residue_names).pop()
        candidate_resnames = cls.ambiguous_flexres_choices.get(resname, [resname])
        for resname in candidate_resnames:
            is_match = False
            atom_names_in_smiles_order = cls.flexres[resname]["atom_names_in_smiles_order"]
            h_to_parent_index = cls.flexres[resname]["h_to_parent_index"]
            expected_names = atom_names_in_smiles_order + list(h_to_parent_index.keys())
            if len(atom_names) != len(expected_names):
                continue
            nr_matched_atom_names = sum([int(n in atom_names) for n in expected_names])
            if nr_matched_atom_names == len(expected_names):
                is_match = True
                break
        if not is_match:
            return None, None, None
        else:
            smiles = cls.flexres[resname]["smiles"]
            index_map = []
            for smiles_index, name in enumerate(atom_names_in_smiles_order):
                index_map.append(smiles_index + 1) 
                index_map.append(atom_names.index(name) + 1)
            h_parent = []
            for name, smiles_index in h_to_parent_index.items():
                h_parent.append(smiles_index + 1)
                h_parent.append(atom_names.index(name) + 1)
            return smiles, index_map, h_parent

    @classmethod
    def add_pose_to_mol(cls, mol, ligand_coordinates, index_map):
        """add given coordinates to given molecule as new conformer.
        Index_map maps order of coordinates to order in smile string
        used to generate rdkit mol

        Args:
            ligand_coordinates (list): Ligand coordinate as list of 3d sets.

        Raises:
            RuntimeError: Will raise error if number of coordinates provided does not
                match the number of atoms there should be coordinates for.
        """

        n_atoms = mol.GetNumAtoms()
        conf = Chem.Conformer(n_atoms)
        if n_atoms != len(index_map) / 2:
            raise RuntimeError(
                "ERROR! Given {n_coords} atom coordinates"
                "but index_map is for {n_at} atoms.".format(
                    n_coords=n_atoms, n_at=len(index_map) / 2))
        for i in range(n_atoms):
            pdbqt_index = int(index_map[i * 2 + 1]) - 1
            x, y, z = [float(coord) for coord in ligand_coordinates[pdbqt_index]]
            conf.SetAtomPosition(int(index_map[i * 2]) - 1, Point3D(x, y, z))
        mol.AddConformer(conf, assignId=True)

        return mol


    @staticmethod
    def add_hydrogens(mol, coordinates_list, h_parent):
        """Add hydrogen atoms to ligand RDKit mol, adjust the positions of
            polar hydrogens to match pdbqt
        """
        mol = Chem.AddHs(mol, addCoords=True)
        conformers = list(mol.GetConformers())
        num_hydrogens = int(len(h_parent) / 2)
        for conformer_idx, atom_coordinates in enumerate(coordinates_list):
            conf = conformers[conformer_idx]
            used_h = []
            for i in range(num_hydrogens):
                parent_rdkit_index = h_parent[2 * i] - 1
                h_pdbqt_index = h_parent[2 * i + 1] - 1
                x, y, z = [
                    float(coord) for coord in atom_coordinates[h_pdbqt_index]
                ]
                parent_atom = mol.GetAtomWithIdx(parent_rdkit_index)
                candidate_hydrogens = [
                    atom.GetIdx() for atom in parent_atom.GetNeighbors()
                    if atom.GetAtomicNum() == 1
                ]
                for h_rdkit_index in candidate_hydrogens:
                    if h_rdkit_index not in used_h:
                        break
                used_h.append(h_rdkit_index)
                conf.SetAtomPosition(h_rdkit_index, Point3D(x, y, z))
        return mol

    @staticmethod
    def combine_rdkit_mols(mol_list):
        """Combines list of rdkit molecules into a single one
            None's are ignored
            returns None if input is empty list or all molecules are None
        """
        combined_mol = None
        for mol in mol_list:
            if mol is None:
                continue
            if combined_mol is None: # first iteration
                combined_mol = mol
            else:
                combined_mol = Chem.CombineMols(combined_mol, mol)
        return combined_mol

    @classmethod
    def _verify_flexres(cls):
        for resname in cls.flexres:
            atom_names_in_smiles_order = cls.flexres[resname]["atom_names_in_smiles_order"]
            h_to_parent_index = cls.flexres[resname]["h_to_parent_index"]
            expected_names = atom_names_in_smiles_order + list(h_to_parent_index.keys())
            if len(expected_names) != len(set(expected_names)):
                raise RuntimeError("repeated atom names in cls.flexres[%s]" % resname)

RDKitMolCreate._verify_flexres()
