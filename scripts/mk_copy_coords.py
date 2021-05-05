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
                        action="store", help="default is to print to STDOUT. Overrides -w")
    parser.add_argument("-w", "--outfn_auto", dest="outfn_auto",
            action="store_true", help="Output filename takes basename and path from --conformers and extension from --original_input. Ignored if -o is specified. WARNING: original input may be overwritten.")
    return parser.parse_args()


if __name__ == '__main__':

    args = cmd_lineparser()
    input_filename = args.input_filename
    coords_filename = args.coords_filename
    output_filename = args.output_filename
    outfn_auto = args.outfn_auto

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
        tmp_obmol = ob.OBMol(obmol) # connectivity may be corrupted by removing and adding Hs multiple times
        pose.copy_coordinates_to_obmol(tmp_obmol)
        output_string += conv.WriteString(tmp_obmol)

    if (output_filename is None) and (outfn_auto):
        outfn = os.path.splitext(coords_filename)[0] + os.path.splitext(input_filename)[1]
        print(output_string, file=open(outfn, 'w'))
    elif (output_filename is None) and (not outfn_auto):
        print(output_string)
    else:
        print(output_string, file=open(output_filename, 'w'))

