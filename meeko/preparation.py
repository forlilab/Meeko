#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko preparation
#

from inspect import signature
import json
eol="\n"
import pathlib
import warnings

from rdkit import Chem

import meeko.macrocycle
from .molsetup import Bond
from .molsetup import RDKitMoleculeSetup
from .atomtyper import AtomTyper
from .espalomatyper import EspalomaTyper
from .bondtyper import BondTyperLegacy
from .hydrate import HydrateMoleculeLegacy
from .macrocycle import FlexMacrocycle
from .flexibility import get_flexibility_model
from .flexibility import update_closure_atoms
from .flexibility import merge_terminal_atoms
from .writer import PDBQTWriterLegacy
from .reactive import assign_reactive_types
from .openff_xml_parser import load_openff

pkg_dir = pathlib.Path(__file__).parents[0]
params_dir = pkg_dir / "data" / "params"
# the above is controversial, see
# https://stackoverflow.com/questions/6028000/how-to-read-a-static-file-from-inside-a-python-package

# DeprecationWarning is not displayed by default
warnings.filterwarnings("default", category=DeprecationWarning)


class MoleculePreparation:
    """
    A class representing the protocol for preparing a molecule for docking.

    Attributes
    ----------
    deprecated_setup_access : list
        Deprecated access to the setup object (deprecated since v0.5).
    merge_these_atom_types : tuple
        A tuple of atom types to merge. 
        For example, ("H",) will merge all hydrogens.
    hydrate : bool
        If True, the molecule will be hydrated with water molecules. 
    flexible_amides : bool
        If True, amide bonds will be treated as flexible.
    rigid_macrocycles : bool
        If True, macrocycles will be treated as rigid.
    untyped_macrocycles : bool
        If True, macrocycles will not be typed.
    min_ring_size : int
        The minimum size of a ring to be considered a macrocycle.
    max_ring_size : int
        The maximum size of a ring to be considered a macrocycle.
    keep_chorded_rings : bool
        If True, chorded rings will be kept in the molecule.
    keep_equivalent_rings : bool
        If True, equivalent rings will be kept in the molecule.
    double_bond_penalty : float
        The penalty for double bonds in macrocycles.
    macrocycle_allow_A : bool
        If True, macrocycles will be allowed to have atom type A (aromatic carbon).
    rigidify_bonds_smarts : list
        A list of SMARTS patterns for bonds to be rigidified.
    rigidify_bonds_indices : list[tuple[int, int]]
        A list of tuples of indices representing bonds to be rigidified.
    input_atom_params : dict
        A dictionary of input atom parameters.
    load_atom_params : str
        A string representing the name of the atom parameters to load. 
    add_atom_types : list
        A list of additional atom types to add.
    charge_model : str
        The charge model to use. Options are "espaloma", "gasteiger", "zero", or "read".
    charge_atom_prop : str
        The name of the atom property to use for charges.
    dihedral_model : str 
        The dihedral model to use. Options are None, "openff", or "espaloma".
    reactive_smarts : str
        A string representing the SMARTS pattern for reactive groups.
    reactive_smarts_idx : int
        An index representing the position of the reactive atom in the SMARTS pattern.
    add_index_map : bool
        If True, an index map will be added to the molecule.
    remove_smiles : bool
        If True, the SMILES representation of the molecule will be removed.
    atom_params : dict
        A dictionary of atom parameters.
    offatom_params : dict
        A dictionary of off-atom parameters.
    dihedral_params : list
        A list of dihedral parameters.
    espaloma_model : EspalomaTyper
        An instance of the EspalomaTyper class. 
    """

    packaged_params = {}
    for path in params_dir.glob("*.json"):  # e.g. data/params/ad4_types.json
        name = path.with_suffix("").name  # e.g. "ad4_types"
        packaged_params[name] = path

    def __init__(
        self,
        merge_these_atom_types=("H",),
        hydrate=False,
        flexible_amides=False,
        rigid_macrocycles=False,
        untyped_macrocycles=False,
        min_ring_size=meeko.macrocycle.DEFAULT_MIN_RING_SIZE,
        max_ring_size=meeko.macrocycle.DEFAULT_MAX_RING_SIZE,
        keep_chorded_rings=False,
        keep_equivalent_rings=False,
        double_bond_penalty=meeko.macrocycle.DEFAULT_DOUBLE_BOND_PENALTY,
        macrocycle_allow_A=False,
        rigidify_bonds_smarts=[],
        rigidify_bonds_indices=[],
        input_atom_params=None,
        load_atom_params="ad4_types",
        add_atom_types=None,
        input_offatom_params=None,
        load_offatom_params=None,
        charge_model="gasteiger",
        charge_atom_prop=None,
        dihedral_model=None,
        reactive_smarts=None,
        reactive_smarts_idx=None,
        add_index_map=False,
        remove_smiles=False,
    ):
        """
        Initialize the MoleculePreparation class.

        Parameters
        ----------
        merge_these_atom_types : tuple
            A tuple of atom types to merge. 
            Default is ("H",) which will merge all hydrogens.
        hydrate : bool
            If True, the molecule will be hydrated with water molecules. 
            Default is False (no hydration).
        flexible_amides : bool
            If True, amide bonds will be treated as flexible.
            Default is False (amide bonds are rigid).
        rigid_macrocycles : bool
            If True, macrocycles will be treated as rigid.
            Default is False (macrocycles are flexible).
        untyped_macrocycles : bool
            If True, macrocycles will not be typed.
            Default is False (macrocycles are typed).
        min_ring_size : int
            The minimum size of a ring to be considered a macrocycle.
            Default is meeko.macrocycle.DEFAULT_MIN_RING_SIZE.
        max_ring_size : int
            The maximum size of a ring to be considered a macrocycle.
            Default is meeko.macrocycle.DEFAULT_MAX_RING_SIZE.
        keep_chorded_rings : bool
            If True, chorded rings will be kept in the molecule.
            Default is False (chorded rings are removed).
        keep_equivalent_rings : bool
            If True, equivalent rings will be kept in the molecule.
            Default is False (equivalent rings are removed).
        double_bond_penalty : float
            The penalty for double bonds in macrocycles.
        macrocycle_allow_A : bool
            If True, macrocycles will be allowed to have atom type A (aromatic carbon).
        rigidify_bonds_smarts : list, optional
            A list of SMARTS patterns for bonds to be rigidified.
        rigidify_bonds_indices : list[tuple[int, int]], optional
            A list of tuples of indices representing bonds to be rigidified.
        input_atom_params : dict, optional
            A dictionary of input atom parameters.
        load_atom_params : str
            A string representing the name of the atom parameters to load. 
            Default is "ad4_types".
        add_atom_types : list, optional
            A list of additional atom types to add.
        charge_model : str
            The charge model to use. Options are "espaloma", "gasteiger", "zero", or "read".
            Default is "gasteiger".
        charge_atom_prop : str, optional
            The name of the atom property to use for charges. Only used if charge_model is "read".
        dihedral_model : str
            The dihedral model to use. Options are None, "openff", or "espaloma".
            Default is None.
        reactive_smarts : str, optional
            A string representing the SMARTS pattern for reactive groups.
        reactive_smarts_idx : int, optional
            An index representing the position of the reactive atom in the SMARTS pattern.
        add_index_map : bool, optional
            If True, an index map will be added to the molecule.
        remove_smiles : bool, optional
            If True, the SMILES representation of the molecule will be removed.

        Raises
        ------
        ValueError
            If unrecognized or conflicting initializer parameters are passed. 
        NotImplementedError
            If load_offatom_params is not None.
        """
        self.deprecated_setup_access = None
        self.merge_these_atom_types = merge_these_atom_types
        self.hydrate = hydrate
        self.flexible_amides = flexible_amides
        self.rigid_macrocycles = rigid_macrocycles
        self.untyped_macrocycles = untyped_macrocycles
        self.min_ring_size = min_ring_size
        self.max_ring_size = max_ring_size
        self.keep_chorded_rings = keep_chorded_rings
        self.keep_equivalent_rings = keep_equivalent_rings
        self.double_bond_penalty = double_bond_penalty
        self.macrocycle_allow_A = macrocycle_allow_A
        self.rigidify_bonds_smarts = rigidify_bonds_smarts
        self.rigidify_bonds_indices = rigidify_bonds_indices

        self.input_atom_params = input_atom_params
        self.load_atom_params = load_atom_params
        self.add_atom_types = add_atom_types

        self.atom_params = self.get_atom_params(
            input_atom_params, load_atom_params, add_atom_types, self.packaged_params
        )

        if load_offatom_params is not None:
            raise NotImplementedError("load_offatom_params not implemented")
        self.load_offatom_params = load_offatom_params

        allowed_charge_models = ["espaloma", "gasteiger", "zero", "read"]
        if charge_model not in allowed_charge_models:
            raise ValueError(
                "unrecognized charge_model: %s, allowed options are: %s"
                % (charge_model, allowed_charge_models)
            )

        self.charge_model = charge_model
        self.charge_atom_prop = charge_atom_prop

        if self.charge_model!="read" and self.charge_atom_prop: 
            raise ValueError(
                f"A charge_atom_prop ({charge_atom_prop}) is given to MoleculePreparation but its current charge_model is {charge_model}. " + eol + 
                "To read charges from atom properties in the input mol, set charge_model to 'read' and name the property as 'charge_atom_prop'. " 
            )
        if self.charge_model=="read":
            if not self.charge_atom_prop: 
                self.charge_atom_prop = "PartialCharge"
                warnings.warn(
                    "The charge_model of MoleculePreparation is set to be 'read', but a valid charge_atom_prop is not given. " + eol + 
                    "The default atom property ('PartialCharge') will be used. "
                )
            elif not isinstance(self.charge_atom_prop, str): 
                raise ValueError(
                    f"Invalid value for charge_atom_prop: expected a string (str), but got {type(self.charge_atom_prop).__name__} instead. "
                )
        
        allowed_dihedral_models = [None, "openff", "espaloma"]
        if dihedral_model in (None, "espaloma"):
            dihedral_list = []
        elif dihedral_model == "openff":
            _, dihedral_list, _ = load_openff()
        else:
            raise ValueError(
                "unrecognized dihedral_model: %s, allowed options are: %s"
                % (dihedral_model, allowed_dihedral_models)
            )

        self.dihedral_model = dihedral_model
        self.dihedral_params = dihedral_list

        if dihedral_model == "espaloma" or charge_model == "espaloma":
            self.espaloma_model = EspalomaTyper()

        self.reactive_smarts = reactive_smarts
        self.reactive_smarts_idx = reactive_smarts_idx
        self.add_index_map = add_index_map
        self.remove_smiles = remove_smiles

        self._bond_typer = BondTyperLegacy()
        self._macrocycle_typer = FlexMacrocycle(
            self.min_ring_size,
            self.max_ring_size,
            self.double_bond_penalty,
            allow_break_atype_A=self.macrocycle_allow_A,
            untyped=self.untyped_macrocycles,
        )
        self._water_builder = HydrateMoleculeLegacy()
        self._classes_setup = {Chem.rdchem.Mol: RDKitMoleculeSetup}

        if input_offatom_params is None:
            self.offatom_params = {}
        else:
            self.offatom_params = input_offatom_params

        if keep_chorded_rings and not keep_equivalent_rings:
            warnings.warn(
                "keep_equivalent_rings=False ignored because keep_chorded_rings=True",
                RuntimeWarning,
            )
        if (reactive_smarts is None) != (reactive_smarts_idx is None):
            raise ValueError(
                "reactive_smarts and reactive_smarts_idx require each other"
            )

    @classmethod
    def from_config(cls, config):
        """
        Create a MoleculePreparation instance from a configuration dictionary.

        Parameters
        ----------
        config : dict
            A dictionary containing the configuration parameters for the
            MoleculePreparation instance.

        Returns
        -------
        MoleculePreparation
            A new instance of the MoleculePreparation class with the specified
            configuration.
        """
        expected_keys = cls.get_defaults_dict().keys()
        bad_keys = [k for k in config if k not in expected_keys]
        if bad_keys:
            warnings.warn(f"Ignore unexpected keys: {bad_keys}")
        config = {k: v for k,v in config.items() if k in expected_keys}
        p = cls(**config)
        return p
    
    @classmethod
    def from_json_file(cls, filename): 
        """
        Create a MoleculePreparation instance from a JSON file containing the configuration. 

        Parameters
        ----------
        filename : str
            The path to the JSON file containing the configuration parameters.
        
        Returns
        -------
        MoleculePreparation
            A new instance of the MoleculePreparation class with the specified
            configuration.
        """
        with open(filename) as f:
            jsonstr = f.read()
        alldata = json.loads(jsonstr)

        return cls.from_config(alldata)

    def calc_flex(
        self,
        setup,
        root_atom_index=None,
        not_terminal_atoms=None,
        delete_ring_bonds=None,
        glue_pseudo_atoms=None,
    ):
        """
        Calculate the flexibility model for the molecule setup.

        Parameters
        ----------
        setup : RDKitMoleculeSetup
            The molecule setup for which to calculate the flexibility model.
        root_atom_index : int, optional
            The index of the root atom in the molecule setup. If None, the root atom
            will be determined automatically.
        not_terminal_atoms : list, optional
            A list of atom indices that should be treated as non-terminal atoms.
        delete_ring_bonds : list[tuple[int, int]], optional
            Bonds deleted for macrocycle flexibility. Each bond is a tuple of two ints (atom 0-indices).
        glue_pseudo_atoms : dict, optional
            A dictionary for pseudo atoms mapping atom indices to their corresponding coordinates.

        Returns
        -------
        None
        """
        if not_terminal_atoms is None:
            not_terminal_atoms = []
        if delete_ring_bonds is None:
            delete_ring_bonds = []
        # 5.  break macrocycles into open/linear form
        if self.rigid_macrocycles:
            break_combo_data = None
            bonds_in_rigid_rings = set()
            # every ring is rigid without macrocycle option
            for ring in setup.rings:
                for bond in setup.get_bonds_in_ring(ring):
                    bonds_in_rigid_rings.add(bond)
        else:
            break_combo_data, bonds_in_rigid_rings = (
                self._macrocycle_typer.search_macrocycle(setup, delete_ring_bonds)
            )

        # This must be done before calling get_flexibility_model
        for bond in bonds_in_rigid_rings:
            setup.bond_info[bond].rotatable = False

        flex_model, bonds_to_break = get_flexibility_model(
            setup, root_atom_index, break_combo_data
        )

        # disasble rotatable bonds that rotate nothing (e.g. -CH3 without H)
        # but glue atoms (i.e. CG) are manually marked as non terminal (by
        # passing them in the `glue_atoms` list) to guarantee that the bond
        # to a CG atom is rotatable and the G pseudo rotates
        glue_atoms = []
        for pair in bonds_to_break:
            for index in pair:
                glue_atoms.append(index)
        merge_terminal_atoms(flex_model, not_terminal_atoms + glue_atoms)

        # bond to a terminal atom, or in a ring that isn't flexible
        actual_rotatable = [v for k, v in flex_model["rigid_body_connectivity"].items()]
        actual_rotatable.extend(bonds_to_break)
        for bond in setup.bond_info:
            if bond not in actual_rotatable:
                setup.bond_info[bond].rotatable = False

        # calculate torsions that would be rotatable without macrocycle breaking
        if break_combo_data is not None and len(break_combo_data["bond_break_combos"]):
            ring_bonds = []
            for ring in setup.rings:
                for bond in setup.get_bonds_in_ring(ring):
                    ring_bonds.append(bond)
            flex_model["torsions_org"] = 0
            for bond in setup.bond_info:
                if setup.bond_info[bond].rotatable and bond not in ring_bonds:
                    flex_model["torsions_org"] += 1

        setup.flexibility_model = flex_model

        # add G pseudo atoms and set CG types
        if not self.untyped_macrocycles:
            update_closure_atoms(setup, bonds_to_break, glue_pseudo_atoms)

        for atom1, atom2 in bonds_to_break:
            bond_id = Bond.get_bond_id(atom1, atom2)
            setup.bond_info[bond_id].breakable = True

        return

    @staticmethod
    def get_atom_params(
        input_atom_params, load_atom_params, add_atom_types, packaged_params
    ):
        """
        Load and combine atom parameters from various sources.

        Parameters
        ----------
        input_atom_params : dict
            A dictionary of input atom parameters.
        load_atom_params : str
            A string representing the name of the atom parameters to load.
        add_atom_types : list
            A list of additional atom types to add.
        packaged_params : dict
            A dictionary of packaged parameters. Keys are parameter group names
            and values are file paths to the corresponding JSON files.

        Raises
        ------
        ValueError
            If there are overlapping parameter groups or if the name of the
            parameter group is not recognized.
        RuntimeError
            If there are multiple groups of parameters when using add_atom_types.
        
        Returns
        -------
        atom_params : dict
            A dictionary of combined atom parameters.
        """
        atom_params = {}
        if type(load_atom_params) == str:
            load_atom_params = [load_atom_params]
        elif load_atom_params is None:
            load_atom_params = ()
        for name in load_atom_params:
            filename = None
            if (
                name == "openff-2.0.0" or name == "openff"
            ):  # TODO allow multiple versions
                vdw_list, _, _ = load_openff()
                d = {"openff-2.0.0": vdw_list}
            elif name in packaged_params:
                filename = packaged_params[name]
            elif name.endswith(".json"):
                filename = name
            else:
                msg = (
                    "names passed to 'load_atom_params' need to suffixed with .json"
                    + eol
                )
                msg += (
                    "or be the unsuffixed basename of a JSON file in %s."
                    % str(params_dir)
                    + eol
                )
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
            if len(overlapping_groups):  # todo: remove duplicated code
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
                raise ValueError(
                    f"input_atom_params {overlap} also set by one or more of {load_atom_params}"
                )

            atom_params.update(d)

        if add_atom_types is not None and len(add_atom_types) > 0:
            group_keys = list(atom_params.keys())
            if len(group_keys) != 1:
                msg = "add_atom_types is usable only when there is one group of parameters"
                msg += ", but there are %d groups: %s" % (
                    len(group_keys),
                    str(group_keys),
                )
                raise RuntimeError(msg)
            key = group_keys[0]
            atom_params[key].extend(add_atom_types)

        return atom_params

    @property
    def setup(self):
        """
        Deprecated access to the setup object.

        Returns
        -------
        RDKitMoleculeSetup
            The setup object in the deprecated_setup_access list.

        Raises
        ------
        RuntimeError
            If there are multiple setups in the deprecated_setup_access list.
        """
        msg = "MoleculePreparation.setup is deprecated in Meeko v0.5."
        msg += (
            " MoleculePreparation.prepare() returns a list of MoleculeSetup instances."
        )
        warnings.warn(msg, DeprecationWarning)
        if len(self.deprecated_setup_access) > 1:
            raise RuntimeError(
                "got multiple setups, use new api: molsetup_list = mk_prep(mol)"
            )

        return self.deprecated_setup_access[0]

    @classmethod
    def get_defaults_dict(cls):
        """
        Get the default values for all parameters in the class.

        Returns
        -------
        dict
            A dictionary containing the default values for all parameters in the class.
        """
        defaults = {}
        sig = signature(cls)
        for key in sig.parameters:
            defaults[key] = sig.parameters[key].default
        return defaults

    def __call__(self, *args, **kwargs):
        """Shortcut to call `prepare()` directly on the MoleculePreparation instance."""
        return self.prepare(*args, **kwargs)

    def prepare(
        self,
        mol,
        root_atom_index=None,
        not_terminal_atoms=None,
        delete_ring_bonds=None,
        glue_pseudo_atoms=None,
        conformer_id=-1,
        rename_atoms=False,
    ):
        """
        Create an RDKitMoleculeSetup from an RDKit Mol object.

        Parameters
        ----------
        mol : Chem.Mol
            An RDKit Mol with explicit hydrogens and 3D coordinates.
        root_atom_index : int, optional
            Used to set ROOT of torsion tree instead of searching.
            Default is None, which means the root atom will be determined automatically.
        not_terminal_atoms : list, optional
            A list of atom indices that should be treated as non-terminal atoms.).
        delete_ring_bonds : list[tuple[int, int]], optional
            Bonds deleted for macrocycle flexibility. Each bond is a tuple of two ints (atom 0-indices).
        glue_pseudo_atoms : dict, optional
            Mapping from parent atom indices to coordinates.
        conformer_id : int
            Conformer ID to use for the molecule. 
            Default is -1, which means the current conformer will be used. 
        rename_atoms : bool
            If True, atoms will be renamed to include their index.
            Default is False, which means atoms will not be renamed.

        Returns
        -------
        setups: list[RDKitMoleculeSetup]
            Returns a list of generated RDKitMoleculeSetups
        """
        if not_terminal_atoms is None:
            not_terminal_atoms = []
        if delete_ring_bonds is None:
            delete_ring_bonds = []
        if glue_pseudo_atoms is None:
            glue_pseudo_atoms = {}
        mol_type = type(mol)
        if mol_type not in self._classes_setup:
            raise TypeError(
                "Molecule is not an instance of supported types: %s" % type(mol)
            )
        setup_class = self._classes_setup[mol_type]
        setup = setup_class.from_mol(
            mol,
            keep_chorded_rings=self.keep_chorded_rings,
            keep_equivalent_rings=self.keep_equivalent_rings,
            compute_gasteiger_charges=self.charge_model == "gasteiger",
            read_charges_from_prop=self.charge_atom_prop,
            conformer_id=conformer_id,
        )

        self.check_external_ring_break(setup, delete_ring_bonds, glue_pseudo_atoms)

        # 1.  assign atom params
        AtomTyper.type_everything(
            setup,
            self.atom_params,
            self.charge_model, # charge_model is not accessed in type_everything
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
            for atom in setup.atoms:
                if atom.atom_type == atype_to_merge:
                    indices.add(atom.index)
        setup.merge_terminal_atoms(indices)

        # 3.  assign bond types
        #     - all single bonds rotatable except some amides and SMARTS rigidification
        #     - macrocycle code breaks rings only at rotatable bonds
        #     - bonds in rigid rings are set as non-rotatable after the flex_model is built
        self._bond_typer(
            setup,
            self.flexible_amides,
            self.rigidify_bonds_smarts,
            self.rigidify_bonds_indices,
        )

        # 4 . hydrate molecule
        if self.hydrate:
            self._water_builder.hydrate(setup)

        self.calc_flex(
            setup,
            root_atom_index,
            not_terminal_atoms,
            delete_ring_bonds,
            glue_pseudo_atoms,
        )

        # 5 . rename atoms, new names will be original name + idx (1-based)
        if rename_atoms:
            for atom in setup.atoms:
                orig_pdbinfo = atom.pdbinfo
                orig_name = orig_pdbinfo.name.strip()
                new_name = f"{orig_name}{atom.index+1}"
                if len(new_name) > 4: 
                    warnings.warn(f"Attempted to rename atom with original name {orig_name} to {new_name}." + eol +
                                  f"But the new name is too long (> 4 characters). The original name will be kept.")
                else:
                    new_atom_info = orig_pdbinfo._replace(name=new_name)
                    atom.pdbinfo = new_atom_info

        if self.reactive_smarts is None:
            setups = [setup]
        else:
            reactive_types_dicts = assign_reactive_types(
                setup,
                self.reactive_smarts,
                self.reactive_smarts_idx,
            )

            if len(reactive_types_dicts) == 0:
                raise RuntimeError("reactive SMARTS didn't match")

            setups = []
            for r in reactive_types_dicts:
                new_setup = setup.copy()
                # There is no guarantee that the addition order in the dictionary will be the correct order to
                # create the list in, so first sorts the keys from the dictionary then extracts the values in order
                # to construct the new atom type list.
                for idx, atom_type in r.items():
                    new_setup.atoms[idx].atom_type = atom_type
                setups.append(new_setup)

        # for a gentle introduction of the new API
        self.deprecated_setup_access = setups

        return setups

    @staticmethod
    def check_external_ring_break(molsetup, break_ring_bonds, glue_pseudo_atoms):
        """
        Check that the external ring break bonds are in the molecule setup and that
        the glue pseudo atoms are present and have the correct number of coordinates.

        Parameters
        ----------
        molsetup : RDKitMoleculeSetup
            The molecule setup to check.
        break_ring_bonds : list[tuple[int, int]]
            A list of tuples representing the bonds to break.
        glue_pseudo_atoms: dict
            A dictionary mapping atom indices to their corresponding coordinates.

        Returns
        -------
        None

        Raises
        ------
        ValueError:
            If bonds are missing from the MoleculeSetup, if glue_pseudo_atoms is missing certain atom indices, and if
            there is an incorrect number of coordinates in glue_pseudo_atoms.
        """
        for index1, index2 in break_ring_bonds:
            has_bond = Bond.get_bond_id(index1, index2) in molsetup.bond_info
            if not has_bond:
                raise ValueError("bond (%d, %d) not in molsetup" % (index1, index2))
            for index in (index1, index2):
                if index not in glue_pseudo_atoms:
                    raise ValueError("missing glue pseudo for atom %d" % index)
                xyz = glue_pseudo_atoms[index]
                if len(xyz) != 3:
                    raise ValueError(
                        "expected 3 coordinates (got %d) for glue pseudo of atom %d"
                        % (len(xyz), index)
                    )
        return

    def write_pdbqt_string(self):
        """
        Writes a PDBQT string. Deprecated in Meeko v0.5.

        Returns
        -------
        str
            The PDBQT string representation of the molecule setup.

        Raises
        ------
        RuntimeError
            If there is an error generating the PDBQT string.
        """
        msg = "MoleculePreparation.write_pdbqt_string() is deprecated in Meeko v0.5."
        msg += " Pass the MoleculeSetup instance to PDBQTWriterLegacy.write_string()."
        msg += (
            " MoleculePreparation.prepare() returns a list of MoleculeSetup instances."
        )
        warnings.warn(msg, DeprecationWarning)
        pdbqt_string, is_ok, err_msg = PDBQTWriterLegacy.write_string(self.setup)
        if not is_ok:
            msg = "Cannot generate PDBQT, error from PDBQTWriterLegacy:" + eol
            msg += err_msg
            raise RuntimeError(msg)
        return pdbqt_string

    def write_pdbqt_file(self, pdbqt_filename):
        """
        Writes out a pdbqt file. Deprecated in Meeko v0.5

        Parameters
        ----------
        pdbqt_filename: str
            PDBQT filename to write to

        Returns
        -------
        None
        """
        warnings.warn(
            "MoleculePreparation.write_pdbqt_file() is deprecated since Meeko v0.5",
            DeprecationWarning,
        )
        with open(pdbqt_filename, "w") as w:
            w.write(self.write_pdbqt_string())
