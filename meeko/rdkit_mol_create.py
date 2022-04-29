#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko
#


from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import AllChem
import json
from rdkit.Chem import SDWriter

class RDKitMolCreate:

    # flexible residue smiles with atom indices
    # corresponding to flexres heteroatoms in pdbqt
    flex_residue_smiles = {
        "LYS": 'CCCCCN',
        "CYS": 'CCS',
        "TYR": 'CC(c4c1).c24.c13.c2c3O',
        "SER": 'CCO',
        "ARG": 'CCCCN=C(N)N',
        "HIP": 'CCC1([N+]=CNC=1)',
        "VAL": 'CC(C)C',
        "ASH": 'CCC(=O)O',
        "GLH": 'CCCC(=O)O',
        "HIE": 'CCC1(N=CNC=1)',
        "GLU": 'CCCC(=O)[O-]',
        "LEU": 'CCC(C)C',
        "PHE": 'CC(c4c1).c24.c13.c2c3',
        "GLN": 'CCCC(N)=O',
        "ILE": 'CC(C)CC',
        "MET": 'CCCSC',
        "ASN": 'CCC(=O)N',
        "ASP": 'CCC(=O)O',
        "HID": 'CCC1(NC=NC=1)',
        "THR": 'CC(C)O',
        "TRP": 'C1=CC=C2C(=C1)C(=CN2)CC'
    }

    @classmethod
    def from_pdbqt_mol(cls, pdbqt_mol): # TODO flexres
        smiles = pdbqt_mol._pose_data['smiles']
        index_map = pdbqt_mol._pose_data['smiles_index_map']
        h_parent = pdbqt_mol._pose_data['smiles_h_parent']
        mol = Chem.MolFromSmiles(smiles)
        coordinates_list = []
        for pose in pdbqt_mol:
            coordinates = pose.positions()
            coordinates_list.append(coordinates)
            cls.add_pose_to_mol(mol, coordinates, index_map)
        mol = cls.add_hydrogens(mol, coordinates_list, h_parent)
        return mol
     
    @staticmethod
    def add_pose_to_mol(mol, ligand_coordinates, index_map, flexres_poses=[], flexres_names=[]):
        """add given coordinates to given molecule as new conformer.
        Index_map maps order of coordinates to order in smile string
        used to generate rdkit mol
        
        Args:
            ligand_coordinates (list): Ligand coordinate as list of 3d sets.
            flexres_poses (list): list of strings of PDBQT lines
                for flexible residues.
            flexres_names (string): List of residue names
                for flexible residues.
        
        Raises:
            RuntimeError: Will raise error if number of coordinates provided does not
                match the number of atoms there should be coordinates for.
        
        """

        n_atoms = mol.GetNumAtoms()
        conf = Chem.Conformer(n_atoms)
        if n_atoms != len(index_map) / 2:
            raise RuntimeError(
                "ERROR! Incorrect number of coordinates! Given {n_coords} "\
                "atom coordinates for {n_at} atoms!".format(
                    n_coords=n_atoms, n_at=len(index_map) / 2))
        for i in range(n_atoms):
            pdbqt_index = int(index_map[i * 2 + 1]) - 1
            x, y, z = [float(coord) for coord in ligand_coordinates[pdbqt_index]]
            conf.SetAtomPosition(int(index_map[i * 2]) - 1, Point3D(x, y, z))
        mol.AddConformer(conf, assignId=True)

        # generate flexible residue mols if we haven't yet
        for idx, resname in enumerate(flexres_names):
            flexres_pdbqt = self._replace_pdbqt_atomtypes(flexres_poses[idx])
            if resname not in self._flexres_mols and resname != '':
                self._create_flexres_molecule(flexres_pdbqt, resname)
            else:
                # add new pose to each of the flexible residue molecules
                # make a new conformer
                flex_res = self._flexres_mols[resname]
                n_atoms = flex_res.GetNumAtoms()
                conf = Chem.Conformer(n_atoms)

                # make an RDKit molecule from the flexres pdbqt to use as a
                # template for setting the coordinates of the conformer
                template = AllChem.MolFromPDBBlock(flexres_pdbqt, removeHs=False)

                # iterate through atoms in template, set corresponding atom in
                # new conformer to the position of the template atom
                for j in range(n_atoms):
                    position = template.GetConformer().GetAtomPosition(j)
                    conf.SetAtomPosition(j, position)

                # add new conformer to flex_res object and add object back
                # to flex_res_mols
                flex_res.AddConformer(conf, assignId=True)
                self._flexres_mols[resname] = flex_res

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
                parent_atom = mol.GetAtomWithIdx(parent_rdkit_index )
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

    def export_combined_rdkit_mol(self):
        """Exports combined ligand and flexres rdkit mol
        """
        self._add_hydrogens_to_pose()  # will only do anything if there were explicit hydrogens included in the source
        return self._combine_ligand_flexres()

    def write_sdf(self, filename):
        mol = self.export_combined_rdkit_mol()
        with SDWriter(filename) as w:
            for conf in mol.GetConformers():
                w.write(mol, conf.GetId())

    def _create_flexres_molecule(self, flexres_pdbqt, flexres_name):
        """Creates RDKit molecules for flexible residues,
            returns list of RDKit Mol objects

        Args:
            flexres_lines (string): flexres pdbqt lines
            flexres_names (list): list of strings of flexres names

        Returns:
            List: list of rdkit mol objects, with one object for each flexres
        """

        # make flexres rdkit molecule, add to our dict of flexres_mols
        # get the residue smiles string and pdbqt we need
        # to make the required rdkit molecules
        try:
            res_smile = self.flex_residue_smiles[flexres_name]
        except KeyError:
            raise KeyError(f"Flexible residue {flexres_name} not recognized.")

        # make rdkit molecules and use template to
        # ensure correct bond order
        template = AllChem.MolFromSmiles(res_smile)
        res_mol = AllChem.MolFromPDBBlock(flexres_pdbqt, removeHs=False)
        res_mol = AllChem.AssignBondOrdersFromTemplate(template, res_mol)

        # Add to dict of all flexible residue molecules
        self._flexres_mols[flexres_name] = res_mol

    def _replace_pdbqt_atomtypes(self, pdbqt, check_atom_line=False):
        """replaces autodock-specific atomtypes with general ones. Reads AD->
        general atomtype mapping from AD_to_STD_ATOMTYPES.json
        
        Args:
            pdbqt (string): String representing pdbqt block with native AD atomtypes
            check_atom_line (bool, optional): flag to check that a line is an atom before trying to modify it
        
        Returns:
            String: pdbqt_line with atomtype replaced with general
                atomtypes recognized by RDKit
        
        Raises:
            RuntimeError: Will raise error if atomtype
                is not in AD_to_STD_ATOMTYPES.json
        """
        new_lines = []
        for pdbqt_line in pdbqt.split("\n"):
            if check_atom_line and not pdbqt_line.startswith("ATOM") and not pdbqt_line.startswith("HETATM"):
                # do not modify non-atom lines
                new_lines.append(pdbqt_line)
                continue

            old_atomtype = pdbqt_line.split()[-1]

            # load autodock to standard atomtype dict if not loaded
            if self.ad_to_std_atomtypes is None:
                with open(
                        os.path.join(os.path.dirname(__file__),
                                     'AD_to_STD_ATOMTYPES.json'), 'r') as f:
                    self.ad_to_std_atomtypes = json.load(f)

            # fetch new atomtype
            try:
                new_atomtype = self.ad_to_std_atomtypes[old_atomtype]
            except KeyError:
                raise RuntimeError(
                    "ERROR! Unrecognized atomtype {at} in flexible residue pdbqt!".
                    format(at=old_atomtype))

            new_lines.append(pdbqt_line.replace(f" {old_atomtype}", f" {new_atomtype}"))  # need space before atomtype to avoid changing other parts of string

        return "\n".join([line.lstrip(" ") for line in list(filter(None, new_lines))])  # formating to keep the new pdbqt block clean

    def _combine_ligand_flexres(self):
        """Combine RDKit mols for ligand and flexible residues
        """
        combined_mol = self.ligand_rdkit_mol
        for flex_res in self._flexres_mols:
            combined_mol = Chem.CombineMols(combined_mol, self._flexres_mols[flex_res])

        return combined_mol
