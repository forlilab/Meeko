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

    for pose in pdbqt_mol:
        pose.copy_coordinates_to_obmol(obmol)
        output_string += conv.WriteString(obmol)

    if output_filename is None:
        print(output_string)
    else:
        print(output_string, file=open(output_filename, 'w'))

