#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko config class
#

import json

class MeekoConfig:
    def __init__(self):
        self.input_molecule_filename = None
        self.break_macrocycle = False
        self.hydrate = False
        self.merge_hydrogens = True
        self.add_hydrogen = False
        self.pH = None
        self.is_protein_sidechain = False
        self.rigidify_bonds_smarts = []
        self.rigidify_bonds_indices = []
        self.params_filename = None
        self.double_bond_penalty = 50
        self.save_index_map = True
        self.output_pdbqt_filename = None
        self.multimol_output_directory = None
        self.multimol_prefix = None
        self.verbose = False
        self.redirect_stdout = False
        self.config_filename = None
        self.amide_rigid = True

    def update_from_json(self):
        with open(self.config_filename) as fin:
            self.__dict__.update(json.load(fin))

    def load_param_file(self):
        # read parameters JSON file
        self.parameters = {}
        if self.params_filename is not None:
            with open(self.params_filename) as f:
                self.parameters.update(json.load(f))

    
