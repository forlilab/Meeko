#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

import argparse
import os
import sys
import warnings

from rdkit import Chem

from meeko import PDBQTMolecule
from meeko import RDKitMolCreate

try:
    from openbabel import openbabel as ob
    from meeko import obutils # fails if openbabel not available
except:
    _has_openbabel = False
else:
    _has_openbabel = True


def cmd_lineparser():
    parser = argparse.ArgumentParser(description='Export docking results to SDF format')
    parser.add_argument(dest='docking_results_filename',
                        action='store', help='Docking output file, either a PDBQT \
                        file from Vina or a DLG file from AD-GPU.')
    parser.add_argument('-i', '--original_input', dest='template_filename',
                        action='store', help='[not recommended] Template molecule file, i.e. the original \
                        file that was used to prepare the PDBQT filename (hopefully SDF). \
                        This option requires OpenBabel, and is kept only for backwards compatibility with \
                        meeko v0.2.* and older, because it is only since v0.3.* that the SMILES string is \
                        added to a PDBQT remark and used to initialize an RDKit molecule.')
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

    is_dlg = docking_results_filename.endswith('.dlg')
    pdbqt_mol = PDBQTMolecule.from_file(docking_results_filename, is_dlg=is_dlg, skip_typing=True)

    if template_filename is not None: # OBMol from template_filename
        if not _has_openbabel:
            raise ImportError('-i/--original_input requires openbabel which is not available')
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
            output_string = conv.WriteString(copy_obmol)
    else: # RDKit mol from SMILES in docking output PDBQT remarks
        output_string, failures = RDKitMolCreate.write_sd_string(pdbqt_mol)
        output_format = 'sdf'
        for i in failures:
            warnings.warn("molecule %d not converted to RDKit/SD File" % i)
        if len(failures) == len(pdbqt_mol._atom_annotations["mol_index"]):
            msg = "\nCould not convert to RDKit. Maybe meeko was not used for preparing\n"
            msg += "the input PDBQT for docking, and the SMILES string is missing?\n"
            msg += "Except for standard protein sidechains, all ligands and flexible residues\n"
            msg += "require a REMARK SMILES line in the PDBQT, which is added automatically by meeko."
            raise RuntimeError(msg)
    if not redirect_stdout:
        if output_filename is None:
            output_filename = '%s%s.%s' % (os.path.splitext(docking_results_filename)[0], suffix_name, output_format)

        print(output_string, file=open(output_filename, 'w'))
    else:
        print(output_string)
