#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

import argparse
import os
import sys

from meeko import MoleculePreparation
from meeko import obutils


def cmd_lineparser():
    parser = argparse.ArgumentParser(description="waterkit")
    parser.add_argument("-i", "--mol", dest="input_molecule_file", required=True,
                        action="store", help="molecule file")
    parser.add_argument("-o", "--out", dest="output_pdbqt_file", default=None,
                        action="store", help="output pdbqt file")
    parser.add_argument("-v", "--verbose", dest="verbose", default=False,
                        action="store_true", help="output pdbqt file")
    return parser.parse_args()


def main():
	args = cmd_lineparser()
	input_molecule_file = args.input_molecule_file
	output_pdbqt_file = args.output_pdbqt_file
	verbose = args.verbose

	mol = obutils.load_molecule_from_file(input_molecule_file)
	preparator = MoleculePreparation(merge_hydrogens=True, macrocycle=True, amide_rigid=True)
	preparator.prepare(mol)

	# maybe verbose could be an option and it will show the various bond scores and breakdowns?
	if verbose:
	    preparator.show_setup()

	if output_pdbqt_file is None:
		# if output format not defined, use infile to infer outname
		name, ext = os.path.splitext(input_molecule_file)
		output_pdbqt_file = "%s_prepared.pdbqt" % (name)

	preparator.write(output_pdbqt_file)

if __name__ == '__main__':
	main()