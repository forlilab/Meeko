#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

import argparse
import os
import sys

from rdkit import Chem
from rdkit.six import StringIO

from meeko import PDBQTMolecule


def cmd_lineparser():
    parser = argparse.ArgumentParser(description='Copy atom coordinates from PDBQT (or DLG) file \
                                                  to original molecule file format (SDF or MOL2)')
    parser.add_argument(dest='docking_results_filename',
                        action='store', help='Docking output file to get coordinates. Either a PDBQT \
                        file from Vina or a DLG file from AD-GPU.')
    parser.add_argument('-i', '--original_input', dest='template_filename',
                        action='store', help='Template molecule file, i.e. the original file that was \
                        used to prepare the PDBQT filename (hopefully SDF). If no template is provided, \
                        the SMILES string in the PDBQT remarks will be used to generate an SDF file.')
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
    docking_results_filename = args.docking_results_filename
    template_filename = args.template_filename
    output_filename = args.output_filename
    suffix_name = args.suffix_name
    redirect_stdout = args.redirect_stdout

    output_string = ''

    is_dlg = docking_results_filename.endswith('.dlg')
    pdbqt_mol = PDBQTMolecule.from_file(docking_results_filename, is_dlg=is_dlg, skip_typing=True)

    if template_filename is not None: # OBMol from template_filename
        if output_filename is not None:
            output_format = os.path.splitext(output_filename)[1][1:]
        else:
            output_format = os.path.splitext(template_filename)[1][1:]
        conv = ob.OBConversion()
        success = conv.SetOutFormat(output_format)
        if not success:
            raise RuntimeError('Input molecule file format %s not recognized by OpenBabel.' % output_format)
        ori_obmol = obutils.load_molecule_from_file(template_filename)
        for pose in pdbqt_mol:
            copy_obmol = ob.OBMol(ori_obmol) # connectivity may be corrupted by removing and adding Hs multiple times
            pose.copy_coordinates_to_obmol(copy_obmol)
            output_string += conv.WriteString(copy_obmol)
    else: # RDKit mol from SMILES in docking output PDBQT remarks
        if pdbqt_mol._pose_data['smiles'] is None:
            msg = "\n\n    \"REMARK SMILES\" not found in %s.\n" % docking_results_filename
            msg += "    Consider using -i/--original_input\n"
            raise RuntimeError(msg)
        sio = StringIO()
        f = Chem.SDWriter(sio)
        for pose in pdbqt_mol:
            rdmol = pose.export_rdkit_mol()
            f.write(rdmol)
        f.close()
        output_string += sio.getvalue()
        output_format = 'sdf'

    if not redirect_stdout:
        if output_filename is None:
            output_filename = '%s%s.%s' % (os.path.splitext(docking_results_filename)[0], suffix_name, output_format)

        print(output_string, file=open(output_filename, 'w'))
    else:
        print(output_string)
