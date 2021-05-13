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
    parser = argparse.ArgumentParser(description='Copy atom coordinates from PDBQT (or DLG) file \
                                                  to original molecule file format (SDF or MOL2)')
    parser.add_argument('-i', '--original_input', dest='input_filename', required=True,
                        action='store', help='Input molecule file (e.g. SDF or MOL2) that was used to \
                        prepare the PDBQT filename.')
    parser.add_argument('-c', '--conformers', dest='coords_filename', required=True,
                        action='store', help='PDBQT or DLG file to get coordinates from.')
    parser.add_argument('-o', '--output_filename', dest='output_filename',
                        action='store', help='Output molecule filename. If not specified, suffix _docked is \
                        added to the filename based on the input molecule file, and using the same \
                        molecule file format')
    parser.add_argument('-s', '--suffix', dest='suffix_name', default='_docked',
                        action='store', help='Add suffix to output filename if -o/--output_filename \
                        not specified. WARNING: If specified as empty string (\'\'), this will overwrite \
                        the original molecule input file (default: _docked).')
    parser.add_argument('-', '--',  dest='redirect_stdout', action='store_true',
                        help='do not write file, redirect output to STDOUT. Arguments -o/--output_filename \
                        is ignored.')
    return parser.parse_args()


if __name__ == '__main__':
    args = cmd_lineparser()
    input_filename = args.input_filename
    coords_filename = args.coords_filename
    output_filename = args.output_filename
    suffix_name = args.suffix_name
    redirect_stdout = args.redirect_stdout

    output_string = ''

    ori_obmol = obutils.load_molecule_from_file(input_filename)

    is_dlg = coords_filename.endswith('.dlg')
    pdbqt_mol = PDBQTMolecule(coords_filename, is_dlg=is_dlg)

    if not redirect_stdout and output_filename is not None:
        # If no output_filename is specified, the format will be the
        # same as the input molecule format
        output_format = os.path.splitext(output_filename)[1][1:]
    else:
        output_format = os.path.splitext(input_filename)[1][1:]

    conv = ob.OBConversion()
    success = conv.SetOutFormat(output_format)
    if not success:
        raise RuntimeError('Input molecule file format %s not recognized by OpenBabel.' % output_format)

    for pose in pdbqt_mol:
        copy_obmol = ob.OBMol(ori_obmol) # connectivity may be corrupted by removing and adding Hs multiple times
        pose.copy_coordinates_to_obmol(copy_obmol)
        output_string += conv.WriteString(copy_obmol)

    if not redirect_stdout:
        if output_filename is None:
            output_filename = '%s%s.%s' % (os.path.splitext(coords_filename)[0], suffix_name, output_format)

        print(output_string, file=open(output_filename, 'w'))
    else:
        print(output_string)
