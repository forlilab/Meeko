#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko multiprocessing manager
#

import platform
from time import sleep
import logging
import queue
import traceback
from .preparation import MoleculePreparation
import sys
from rdkit import Chem

if platform.system() == "Darwin":  # mac
    import multiprocess as multiprocessing
else:
    import multiprocessing


class Parallelizer:
    def __init__(self, max_proc, args, output, backend, is_covalent, preparator, covalent_builder) -> None:
        self.n_workers = max_proc - 1  # reserve one core for pdbqt writing

        self.args = args
        self.output = output
        self.backend = backend
        self.is_covalent = is_covalent
        self.preparator = preparator
        self.covalent_builder = covalent_builder

        self.processed_mols = 0
        self.input_mol_skipped = 0
        self.input_mol_with_failure = 0
        self.nr_failures = 0

    def _mp_wrapper(self, mol):
        output_bundle = MoleculePreparation.prep_single_mol(mol, self.args, self.output, self.backend, self.is_covalent, self.preparator, self.covalent_builder, write_output=False)
        return output_bundle

    def process_mols(self, mol_supplier):
        # set pickle options to prevent loss of names
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.MolProps |
                                        Chem.PropertyPickleOptions.PrivateProps)
        
        pool = multiprocessing.Pool(self.n_workers)
        for is_valid, this_mol_had_failure, nr_f, output_pdbqts_info in pool.imap_unordered(self._mp_wrapper, mol_supplier):
            for pdbqt_string, name, covLabel_suffix in output_pdbqts_info:
                self.output(pdbqt_string, name, covLabel_suffix)
                print(f"Done {name}")
                self.processed_mols += 1
            self.input_mol_skipped += int(is_valid==False)
            self.input_mol_with_failure += int(this_mol_had_failure)
            self.nr_failures += nr_f

        return self.input_mol_skipped, self.input_mol_with_failure, self.nr_failures