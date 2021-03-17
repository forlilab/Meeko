#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

import argparse
import os
import sys

from openbabel import openbabel as ob

from meeko import PDBQTMolecule
from meeko import obutils


def cmd_lineparser():
    parser = argparse.ArgumentParser(description="Copy coords from PDBQT (or DLG) into original file")
    parser.add_argument("-i", "--original_input", dest="input_filename", required=True,
                        action="store", help="input molecule file (e.g. SDF) that was used to prepare the PDBQT")
    parser.add_argument("-c", "--conformers", dest="coords_filename", required=True,
                        action="store", help="PDBQT or DLG file to get coordinates from")
    parser.add_argument("-o", "--output_filename", dest="output_filename",
                        action="store", help="output molecule file in SDF format")
    return parser.parse_args()

def copy_coords(obmol, coords, index_map):
    """ Args:
        obmol (OBMol): coordinates will be changed in this object
        coords (2D array): coordinates to copy
        index_map (dict): map of atom indices from obmol (keys) to coords (values)
    """
    n_atoms = obmol.NumAtoms()
    n_matched_atoms = 0
    hydrogens_to_delete = []
    heavy_parents = []
    for atom in ob.OBMolAtomIter(obmol):
        ob_index = atom.GetIdx() # 1-index
        if ob_index in pdbqt_mol._index_map:
            pdbqt_index = pdbqt_mol._index_map[ob_index]-1
            x, y, z = pose_xyz[pdbqt_index, :] 
            atom.SetVector(x, y, z)
            n_matched_atoms += 1
        elif atom.GetAtomicNum() != 1:
            raise RuntimeError('obmol heavy atom missing in pdbqt_mol, only hydrogens can be missing')
        else:
            hydrogens_to_delete.append(atom)
            bond_counter = 0
            for bond in ob.OBAtomBondIter(atom):
                bond_counter += 1
            if bond_counter != 1:
                raise RuntimeError("hydrogen atom has %d bonds, must have 1" % bond_counter)
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            if atom == begin_atom:
                heavy_parents.append(end_atom)
            elif atom == end_atom:
                heavy_parents.append(begin_atom)
            else:
                raise RuntimeError("hydrogen isn't either Begin or End atom of its own bond")

    if n_matched_atoms != len(index_map):
        raise RuntimeError("Not all pdbqt_mol atoms were considered")

    # delete explicit hydrogens
    for hydrogen in hydrogens_to_delete:
        obmol.DeleteHydrogen(hydrogen)

    # increment implicit H count of heavy atom parents
    for heavy_parent in heavy_parents:
        n_implicit = heavy_parent.GetImplicitHCount()
        heavy_parent.SetImplicitHCount(n_implicit + 1)

    # add back explicit hydrogens
    obmol.AddHydrogens()
    if obmol.NumAtoms() != n_atoms:
        raise RuntimeError("number of atoms changed after deleting and adding hydrogens")

    return



if __name__ == '__main__':

    args = cmd_lineparser()
    input_filename = args.input_filename
    coords_filename = args.coords_filename
    output_filename = args.output_filename

    output_string = ""

    obmol = obutils.load_molecule_from_file(input_filename)
    is_dlg = coords_filename.endswith('.dlg')
    pdbqt_mol = PDBQTMolecule(coords_filename, is_dlg=is_dlg)

    output_format = 'sdf'
    if output_filename is not None:
        output_format = os.path.splitext(output_filename)[1][1:]
    conv = ob.OBConversion()
    success = conv.SetOutFormat(output_format)
    if not success:
        raise RuntimeError("file format %s not recognized by openbabel" % output_format)

    for pose_xyz in pdbqt_mol._positions: # iterate over poses
        copy_coords(obmol, pose_xyz, pdbqt_mol._index_map)
        output_string += conv.WriteString(obmol)

    if output_filename is None:
        print(output_string)
    else:
        print(output_string, file=open(output_filename, 'w'))

