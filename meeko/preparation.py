#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko preparation
#

from inspect import signature
import json
import os
import pathlib
import sys
from collections import OrderedDict
import warnings

from rdkit import Chem

from .molsetup import OBMoleculeSetup
from .molsetup import RDKitMoleculeSetup
from .atomtyper import AtomTyper
from .espalomatyper import EspalomaTyper
from .bondtyper import BondTyperLegacy
from .hydrate import HydrateMoleculeLegacy
from .macrocycle import FlexMacrocycle
from .flexibility import FlexibilityBuilder
from .writer import PDBQTWriterLegacy
from .reactive import assign_reactive_types
from .openff_xml_parser import load_openff

pkg_dir = pathlib.Path(__file__).parents[0]
params_dir = pkg_dir / "data" / "params"
# the above is controversial, see
# https://stackoverflow.com/questions/6028000/how-to-read-a-static-file-from-inside-a-python-package

try:
    from openbabel import openbabel as ob
except ImportError:
    _has_openbabel = False
else:
    _has_openbabel = True

# DeprecationWarning is not displayed by default
warnings.filterwarnings("default", category=DeprecationWarning)

class MoleculePreparation:

    packaged_params = {}
    for path in params_dir.glob("*.json"): # e.g. data/params/ad4_types.json
        name = path.with_suffix("").name # e.g. "ad4_types"
        packaged_params[name] = path

    def __init__(self,
            merge_these_atom_types=("H",),
            hydrate=False,
            flexible_amides=False,
            rigid_macrocycles=False,
            min_ring_size=7,
            max_ring_size=33,
            keep_chorded_rings=False,
            keep_equivalent_rings=False,
            double_bond_penalty=50,
            rigidify_bonds_smarts=[],
            rigidify_bonds_indices=[],
            input_atom_params=None,
            load_atom_params="ad4_types",
            add_atom_types=(),
            input_offatom_params=None, 
            load_offatom_params=None,
            charge_model="gasteiger",
            dihedral_model=None,
            reactive_smarts=None,
            reactive_smarts_idx=None,
            add_index_map=False,
            remove_smiles=False,
        ):

        self.deprecated_setup_access = None
        self.merge_these_atom_types = merge_these_atom_types
        self.hydrate = hydrate
        self.flexible_amides = flexible_amides
        self.rigid_macrocycles = rigid_macrocycles
        self.min_ring_size = min_ring_size
        self.max_ring_size = max_ring_size
        self.keep_chorded_rings = keep_chorded_rings
        self.keep_equivalent_rings = keep_equivalent_rings
        self.double_bond_penalty = double_bond_penalty
        self.rigidify_bonds_smarts = rigidify_bonds_smarts
        self.rigidify_bonds_indices = rigidify_bonds_indices

        self.input_atom_params = input_atom_params
        self.load_atom_params = load_atom_params
        self.add_atom_types = add_atom_types

        self.atom_params = self.get_atom_params(
                input_atom_params,
                load_atom_params,
                add_atom_types,
                self.packaged_params)

        self.load_offatom_params = load_offatom_params
        
        allowed_charge_models = ["espaloma", "gasteiger", "zero"]
        if charge_model not in allowed_charge_models:
            raise ValueError("unrecognized charge_model: %s, allowed options are: %s" % (charge_model, allowed_charge_models))

        self.charge_model = charge_model
       
        allowed_dihedral_models = [None,'openff','espaloma']
        if dihedral_model in (None, 'espaloma'):
            dihedral_list = []
        elif dihedral_model == "openff":
            _, dihedral_list, _ = load_openff()
        else: 
            raise ValueError("unrecognized dihedral_model: %s, allowed options are: %s" % (dihedral_model, allowed_dihedral_models))
        
        self.dihedral_model = dihedral_model
        self.dihedral_params = dihedral_list

        if dihedral_model == 'espaloma' or charge_model == "espaloma":
            self.espaloma_model = EspalomaTyper()

        self.reactive_smarts = reactive_smarts
        self.reactive_smarts_idx = reactive_smarts_idx
        self.add_index_map = add_index_map
        self.remove_smiles = remove_smiles

        self._bond_typer = BondTyperLegacy()
        self._macrocycle_typer = FlexMacrocycle(
                self.min_ring_size, self.max_ring_size, self.double_bond_penalty)
        self._flex_builder = FlexibilityBuilder()
        self._water_builder = HydrateMoleculeLegacy()
        self._classes_setup = {Chem.rdchem.Mol: RDKitMoleculeSetup}

        self.offatom_params = {}

    @staticmethod
    def get_atom_params(input_atom_params, load_atom_params, add_atom_types, packaged_params):
        atom_params = {}
        if type(load_atom_params) == str:
            load_atom_params = [load_atom_params]
        elif load_atom_params is None:
            load_atom_params = ()
        for name in load_atom_params:
            filename = None
            if name == "openff-2.0.0" or name == "openff": # TODO allow multiple versions
                vdw_list, _, _ = load_openff() 
                d = {"openff-2.0.0": vdw_list}
            elif name in packaged_params:
                filename = packaged_params[name]
            elif name.endswith(".json"):
                filename = name
            else:
                msg =  "names passed to 'load_atom_params' need to suffixed with .json" + os.linesep
                msg += "or be the unsuffixed basename of a JSON file in %s." % str(params_dir) + os.linesep
                msg += "name was %s" % name 
                raise ValueError(msg)
            if filename is not None:
                with open(filename) as f:
                    d = json.load(f)
            overlapping_groups = set(atom_params).intersection(set(d))
            if len(overlapping_groups):
                msg = "overlapping parameter groups: %s" % str(overlapping_groups)
                raise ValueError(msg)
            atom_params.update(d)

        if input_atom_params is not None:
            d = json.loads(json.dumps(input_atom_params))
            overlapping_groups = set(atom_params).intersection(set(d))
            if len(overlapping_groups): # todo: remove duplicated code
                msg = "overlapping parameter groups: %s" % str(overlapping_groups)
                raise ValueError(msg)
            params_set_here = set()
            for group_key, rows in input_atom_params.items():
                for row in rows:
                    for param_name in row:
                        if param_name not in ["smarts", "comment", "IDX"]:
                            params_set_here.add(param_name)
            params_set_before = set()
            for group_key, rows in atom_params.items():
                for row in rows:
                    for param_name in row:
                        if param_name not in ["smarts", "comment", "IDX"]:
                            params_set_before.add(param_name)
            overlap = params_set_before.intersection(params_set_here)
            if len(overlap):
                msg = f"input_atom_params {overlap} also set by one or more of {load_atom_params}\n"
                msg += "consider setting load_atom_params=None"
                raise ValueError(f"input_atom_params {overlap} also set by one or more of {load_atom_params}")
                
            atom_params.update(d)

        if len(add_atom_types) > 0:
            group_keys = list(atom_params.keys())
            if len(group_keys) != 1:
                msg = "add_atom_types is usable only when there is one group of parameters"
                msg += ", but there are %d groups: %s" % (len(keys), str(keys))
                raise RuntimeError(msg)
            key = group_keys[0]
            atom_params[key].extend(add_atom_types)
               
        return atom_params

                ### else:
                ###     msg = "When atom_params is a list, it must consist of either:" + os.linesep
                ###     msg += "  - strings (to look up default JSON files in the data dir)" + os.linesep
                ###     msg += "  - or dictionaries (to be used directly). For example:" + os.linesep
                ###     msg += "                  {\"SMARTS\": \"[B]\", \"atype\": \"C\"}" + os.linesep
                ###     msg += "A total of %d types were found in atom_params:" % (len(types)) + os.linesep
                ###     msg += str(types) + os.linesep
                ###     raise ValueError(msg)
                ### 
                ### #for value in atom_params:
                    

        self.offatom_params = offatom_params
        self.charge_model = charge_model
        self.dihedral_params = dihedral_params


        if _has_openbabel:
            self._classes_setup[ob.OBMol] = OBMoleculeSetup
        if keep_chorded_rings and keep_equivalent_rings==False:
            warnings.warn("keep_equivalent_rings=False ignored because keep_chorded_rings=True", RuntimeWarning)
        if (reactive_smarts is None) != (reactive_smarts_idx is None):
            raise ValueError("reactive_smarts and reactive_smarts_idx require each other")

    @property
    def setup(self):
        msg = "MoleculePreparation.setup is deprecated in Meeko v0.5."
        msg += " MoleculePreparation.prepare() returns a list of MoleculeSetup instances."
        warnings.warn(msg, DeprecationWarning)
        return self.deprecated_setup_access

    @classmethod
    def get_defaults_dict(cls):
        defaults = {}
        sig = signature(cls)
        for key in sig.parameters:
            defaults[key] = sig.parameters[key].default 
        return defaults

    @classmethod
    def from_config(cls, config):
        expected_keys = cls.get_defaults_dict().keys()
        bad_keys = [k for k in config if k not in expected_keys]
        if len(bad_keys) > 0:
            err_msg = "unexpected keys in MoleculePreparation.from_config():" + os.linesep
            for key in bad_keys:
                err_msg += "  - %s" % key + os.linesep
            raise ValueError(err_msg)
        p = cls(**config)
        return p

    def __call__(self, *args):
        return self.prepare(*args)

    def prepare(self,
            mol,
            root_atom_index=None,
            not_terminal_atoms=[],
            delete_ring_bonds=[],
            glue_pseudo_atoms={},
            conformer_id=-1,
        ):
        """ 
        Create molecule setup from RDKit molecule

        Args:
            mol (rdkit.Chem.rdchem.Mol): with explicit hydrogens and 3D coordinates
            root_atom_index (int): to set ROOT of torsion tree instead of searching
            not_terminal_atoms (list): make bonds with terminal atoms rotatable
                                       (e.g. C-Alpha carbon in flexres)
            delete_ring_bonds (list): bonds deleted for macrocycle flexibility
                                      each bond is a tuple of two ints (atom 0-indices)
            glue_pseudo_atoms (dict): keys are parent atom indices, values are (x, y, z)
        """
        mol_type = type(mol)
        if not mol_type in self._classes_setup:
            raise TypeError("Molecule is not an instance of supported types: %s" % type(mol))
        setup_class = self._classes_setup[mol_type]
        setup = setup_class.from_mol(mol,
            keep_chorded_rings=self.keep_chorded_rings,
            keep_equivalent_rings=self.keep_equivalent_rings,
            assign_charges=self.charge_model=="gasteiger",
            conformer_id=conformer_id,
            )

        self.check_external_ring_break(setup, delete_ring_bonds, glue_pseudo_atoms)

        # 1.  assign atom params
        AtomTyper.type_everything(
            setup,
            self.atom_params,
            self.charge_model,
            self.offatom_params,
            self.dihedral_params,
        )
        
        # Convert molecule to graph and apply trained Espaloma model
        if self.dihedral_model == "espaloma" or self.charge_model == "espaloma":
            molgraph = self.espaloma_model.get_espaloma_graph(setup)

        # Grab dihedrals from graph node and set them to the molsetup
        if self.dihedral_model == "espaloma":
            self.espaloma_model.set_espaloma_dihedrals(setup, molgraph)

        # Grab charges from graph node and set them to the molsetup
        if self.charge_model == "espaloma":
            self.espaloma_model.set_espaloma_charges(setup, molgraph)

        # merge hydrogens (or any terminal atoms)
        indices = set()
        for atype_to_merge in self.merge_these_atom_types:
            for index, atype in setup.atom_type.items():
                if atype == atype_to_merge:
                    indices.add(index)
        setup.merge_terminal_atoms(indices)

        # 3.  assign bond types by using SMARTS...
        #     - bonds should be typed even in rings (but set as non-rotatable)
        #     - if macrocycle is selected, they will be enabled (so they must be typed already!)
        self._bond_typer(setup, self.flexible_amides, self.rigidify_bonds_smarts, self.rigidify_bonds_indices, not_terminal_atoms)
        # 4 . hydrate molecule
        if self.hydrate:
            self._water_builder.hydrate(setup)
        # 5.  break macrocycles into open/linear form
        if self.rigid_macrocycles:
            break_combo_data = None
            bonds_in_rigid_rings = None # not true, but this is only needed when breaking macrocycles
        else:
            break_combo_data, bonds_in_rigid_rings = self._macrocycle_typer.search_macrocycle(setup, delete_ring_bonds)

        # 6.  build flexibility...
        # 6.1 if macrocycles typed:
        #     - walk the setup graph by skipping proposed closures
        #       and score resulting flex_trees basing on the lenght
        #       of the branches generated
        #     - actually break the best closure bond (THIS CHANGES SOME ATOM TYPES)
        # 6.2  - walk the graph and build the flextree
        # 7.  but disable all bonds that are in rings and not
        #     in flexible macrocycles
        # TODO restore legacy AD types for PDBQT
        #self._atom_typer.set_param_legacy(mol)

        setup = self._flex_builder(setup,
                                   root_atom_index=root_atom_index,
                                   break_combo_data=break_combo_data,
                                   bonds_in_rigid_rings=bonds_in_rigid_rings,
                                   glue_pseudo_atoms=glue_pseudo_atoms,
        )

        if self.reactive_smarts is None:
            setups = [setup]
        else:
            reactive_types_dicts = assign_reactive_types(
                    setup,
                    self.reactive_smarts,
                    self.reactive_smarts_idx,
            )
            setups = []
            for r in reactive_types_dicts:
                new_setup = setup.copy()
                new_setup.atom_type = r
                setups.append(new_setup)

        self.deprecated_setup_access = setups[0] # for a gentle introduction of the new API
        return setups


    @staticmethod
    def check_external_ring_break(molsetup, break_ring_bonds, glue_pseudo_atoms):
        for (index1, index2) in break_ring_bonds:
            has_bond = molsetup.get_bond_id(index1, index2) in molsetup.bond
            if not has_bond:
                raise ValueError("bond (%d, %d) not in molsetup" % (index1, index2))
            for index in (index1, index2):
                if index not in glue_pseudo_atoms:
                    raise ValueError("missing glue pseudo for atom %d" % index) 
                xyz = glue_pseudo_atoms[index]
                if len(xyz) != 3:
                    raise ValueError("expected 3 coordinates (got %d) for glue pseudo of atom %d" % (len(xyz), index)) 


    def write_pdbqt_string(self, add_index_map=None, remove_smiles=None):
        msg = "MoleculePreparation.write_pdbqt_string() is deprecated in Meeko v0.5."
        msg += " Pass the MoleculeSetup instance to PDBQTWriterLegacy.write_string()."
        msg += " MoleculePreparation.prepare() returns a list of MoleculeSetup instances."
        warnings.warn(msg, DeprecationWarning)
        pdbqt_string, is_ok, err_msg = PDBQTWriterLegacy.write_string(self.setup)
        if not is_ok:
            msg = 'Cannot generate PDBQT, error from PDBQTWriterLegacy:' + os.linesep
            msg += err_msg
            raise RuntimeError(msg)
        return pdbqt_string


    def write_pdbqt_file(self, pdbqt_filename, add_index_map=None, remove_smiles=None):
        warnings.warn("MoleculePreparation.write_pdbqt_file() is deprecated since Meeko v0.5", DeprecationWarning)
        with open(pdbqt_filename,'w') as w:
            w.write(self.write_pdbqt_string(add_index_map, remove_smiles))

