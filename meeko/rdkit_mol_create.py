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

    # flexible residue smarts
    flex_residues_noTemplate = {"LYS", "CYS", "SER", "VAL", "LEU", "ILE", "MET", "THR"}
    flex_residue_smiles = {
        "TYR": 'CCc1ccc(O)cc1',
        "ASH": 'CCC(=O)O',
        "GLH": 'CCCC(=O)O',
        "GLU": 'CCCC(=O)[O-]',
        "PHE": 'CCc1ccccc1',
        "GLN": 'CCCC(N)=O',
        "ASN": 'CCC(=O)N',
        "ASP": 'CCC(=O)O',
        "TRP": 'CCc1c[nH]c2ccccc12'
    }
    flex_residue_smarts = {
        "HIP": ('[C]~[C]~[C]1~[N+]~[C]~[N]~[C]1', [(4, 5), (6, 2)]),
        "HIE": ('[C]~[C]~[C]1~[N]~[C]~[N]~[C]1', [(3, 4), (6, 2)]),
        "HID": ('[C]~[C]~[C]1~[N]~[C]~[N]~[C]1', [(4, 5), (6, 2)]),
        "ARG": ('[C]~[C]~[C]~[N]~[C](~[N])~[N+]', [(4, 6)])
    }

    ad_to_std_atomtypes = None

    @classmethod
    def from_pdbqt_mol(cls, pdbqt_mol):  # TODO: add water
        smiles = pdbqt_mol._pose_data['smiles']
        index_map = pdbqt_mol._pose_data['smiles_index_map']
        h_parent = pdbqt_mol._pose_data['smiles_h_parent']
        # make set of flexible residue names
        flexres_names = []
        for atom in pdbqt_mol.atoms_by_properties("flexible_residue"):
            res_name = ":".join([str(atom[4]), str(atom[5]), str(atom[3])])
            if res_name not in flexres_names:
                flexres_names.append(res_name)

        flexres_mols = {}
        mol = Chem.MolFromSmiles(smiles)
        coordinates_list = []
        for pose in pdbqt_mol:
            full_pdbqt = pdbqt_mol.write_pdbqt_string()
            flexres_poses = []
            for res in flexres_names:
                res_lines = []
                for line in full_pdbqt.split("\n"):
                    if line.startswith("ATOM") or line.startswith("HETATOM"):
                        resname = line[17:20]
                        chain = line[21].strip()
                        resnum = int(line[22:26])
                        if "%s:%s:%s" % (resname, chain, resnum) == res:
                            res_lines.append(line)
                flexres_poses.append("\n".join(res_lines))
            coordinates = pose.positions()
            coordinates_list.append(coordinates)
            mol, flexres_mols = cls.add_pose_to_mol(mol, coordinates, index_map,
                                                    flexres_mols=flexres_mols,
                                                    flexres_poses=flexres_poses,
                                                    flexres_names=flexres_names)
        return cls.export_combined_rdkit_mol(mol, flexres_mols, coordinates_list, h_parent)

    @classmethod
    def replace_pdbqt_atomtypes(cls, pdbqt, check_atom_line=True):
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
            if cls.ad_to_std_atomtypes is None:
                with open(
                        os.path.join(os.path.dirname(__file__),
                                     'AD_to_STD_ATOMTYPES.json'), 'r') as f:
                    cls.ad_to_std_atomtypes = json.load(f)

            # fetch new atomtype
            try:
                new_atomtype = cls.ad_to_std_atomtypes[old_atomtype]
            except KeyError:
                raise RuntimeError(
                    "ERROR! Unrecognized atomtype {at} in flexible residue pdbqt!".
                    format(at=old_atomtype))

            # need space before atomtype to avoid changing other parts of string
            new_lines.append(pdbqt_line.replace(f" {old_atomtype}", f" {new_atomtype}"))

        return "\n".join([line.lstrip(" ") for line in list(filter(None, new_lines))])  # formating to keep the new pdbqt block clean

    @classmethod
    def create_flexres_molecule(cls, flexres_pdbqt, flexres_name):
        """Creates RDKit molecules for flexible residues,
            returns list of RDKit Mol objects

        Args:
            flexres_lines (string): flexres pdbqt lines
            flexres_name (str): Name for flexible residue. Expects 3-letter code as first 3 characters

        Returns:
            List: list of rdkit mol objects, with one object for each flexres
            Returns None if keyerror while finding flexible residue name
        """

        # make flexres rdkit molecule, add to our dict of flexres_mols
        # get the residue smiles string and pdbqt we need
        # to make the required rdkit molecules
        res_mol = AllChem.MolFromPDBBlock(flexres_pdbqt, removeHs=False)
        # return mols with only single bonds
        if flexres_name[:3] in cls.flex_residues_noTemplate:
            return res_mol
        try:
            # strip out 3-letter residue code
            res_smile = cls.flex_residue_smiles[flexres_name[:3]]
            template = AllChem.MolFromSmiles(res_smile)
            res_mol = AllChem.AssignBondOrdersFromTemplate(template, res_mol)
        # catch residues without template smile, try template smarts
        except KeyError:
            try:
                res_smarts, double_bond_list = cls.flex_residue_smarts[flexres_name[:3]]
                template = AllChem.MolFromSmarts(res_smarts)
                matches = res_mol.GetSubstructMatches(template)
                editable_res = Chem.EditableMol(res_mol)
                for bond in double_bond_list:
                    editable_res.RemoveBond(matches[0][bond[0]], matches[0][bond[1]])
                    editable_res.AddBond(matches[0][bond[0]], matches[0][bond[1]], order=Chem.rdchem.BondType.DOUBLE)
                res_mol = editable_res.GetMol()
            # catchn additional keyerror indicating unrecognized residue, give warning
            except KeyError:
                warnings.warn(f"Flexible residue {flexres_name} not recognized. Will not be included in output SDFs")
                return None

        # make rdkit molecules and use template to
        # ensure correct bond order

        return res_mol

    @classmethod
    def add_pose_to_mol(cls, mol, ligand_coordinates, index_map, flexres_mols={}, flexres_poses=[], flexres_names=[]):
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
                "ERROR! Incorrect number of coordinates! Given {n_coords} "
                "atom coordinates for {n_at} atoms!".format(
                    n_coords=n_atoms, n_at=len(index_map) / 2))
        for i in range(n_atoms):
            pdbqt_index = int(index_map[i * 2 + 1]) - 1
            x, y, z = [float(coord) for coord in ligand_coordinates[pdbqt_index]]
            conf.SetAtomPosition(int(index_map[i * 2]) - 1, Point3D(x, y, z))
        mol.AddConformer(conf, assignId=True)

        # generate flexible residue mols if we haven't yet
        for idx, resname in enumerate(flexres_names):
            flexres_pdbqt = cls.replace_pdbqt_atomtypes(flexres_poses[idx])
            if resname not in flexres_mols and resname != '':
                resmol = cls.create_flexres_molecule(flexres_pdbqt, resname)
                # Unrecognized residues returned as None
                if resmol is not None:
                    flexres_mols[resname] = resmol
            else:
                # add new pose to each of the flexible residue molecules
                # make a new conformer
                flex_res = flexres_mols[resname]
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
                flexres_mols[resname] = flex_res

        return mol, flexres_mols

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

    @classmethod
    def export_combined_rdkit_mol(cls, mol, flexres_mols, coordinates_list, h_parent):
        """Exports combined ligand and flexres rdkit mol
        """
        # will only do anything if there were explicit hydrogens included in the source
        mol = cls.add_hydrogens(mol, coordinates_list, h_parent)
        combined_mol = mol
        for flex_res in flexres_mols:
            combined_mol = Chem.CombineMols(combined_mol, flexres_mols[flex_res])

        return combined_mol

    @staticmethod
    def add_properties_to_mol(mol, information_dictionary):
        """Takes RDKit mol, and given information as properties

        Args:
            mol (RDKit mol): input molecule to which properties will be added
            information_dictionary (dict): Dictionary of info to include as properties. Key will become property name, value will be value

        """

        for k, v in information_dictionary.items():
            mol.SetProp(k, v)
