import pathlib
import json
import logging
import traceback
from importlib.resources import files
from os import linesep as eol
from sys import exc_info
from typing import Union
from typing import Optional

import rdkit.Chem
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdChemReactions
from rdkit.Chem import rdMolInterchange
from rdkit.Geometry import Point3D

from .molsetup import RDKitMoleculeSetup
from .molsetup import MoleculeSetupEncoder
from .utils.jsonutils import rdkit_mol_from_json
from .utils.rdkitutils import mini_periodic_table
from .utils.rdkitutils import react_and_map
from .utils.rdkitutils import AtomField
from .utils.rdkitutils import build_one_rdkit_mol_per_altloc
from .utils.rdkitutils import _aux_altloc_mol_build
from .utils.rdkitutils import covalent_radius
from .utils.pdbutils import PDBAtomInfo
from .chemtempgen import export_chem_templates_to_json
from .chemtempgen import build_noncovalent_CC
from .chemtempgen import build_linked_CCs

import numpy as np



# region JSON Encoders
class MonomerEncoder(json.JSONEncoder):
    """
    JSON Encoder class for Monomer objects.
    """

    molecule_setup_encoder = MoleculeSetupEncoder()

    def default(self, obj):
        """
        Overrides the default JSON encoder for data structures for Monomer objects.

        Parameters
        ----------
        obj: object
            Can take any object as input, but will only create the Monomer JSON format for Monomer objects.
            For all other objects will return the default json encoding.

        Returns
        -------
        A JSON serializable object that represents the Monomer class or the default JSONEncoder output for an
        object.
        """
        if isinstance(obj, Monomer):
            if obj.molsetup is None:
                molsetup_json = None
            else:
                molsetup_json = self.molecule_setup_encoder.default(obj.molsetup)
            return {
                "raw_rdkit_mol": rdkit_or_none_to_json(obj.raw_rdkit_mol),
                "rdkit_mol": rdkit_or_none_to_json(obj.rdkit_mol),
                "mapidx_to_raw": obj.mapidx_to_raw,
                "residue_template_key": obj.residue_template_key,
                "input_resname": obj.input_resname,
                "atom_names": obj.atom_names,
                "mapidx_from_raw": obj.mapidx_from_raw,
                "padded_mol": rdkit_or_none_to_json(obj.padded_mol),
                "molsetup": molsetup_json,
                "is_flexres_atom": obj.is_flexres_atom,
                "is_movable": obj.is_movable,
                "molsetup_mapidx": obj.molsetup_mapidx,
            }
        return json.JSONEncoder.default(self, obj)


class ResidueTemplateEncoder(json.JSONEncoder):
    """
    JSON Encoder class for ResidueTemplate objects.
    """

    def default(self, obj):
        """
        Overrides the default JSON encoder for data structures for ResidueTemplate objects.

        Parameters
        ----------
        obj: object
            Can take any object as input, but will only create the ResidueTemplate JSON format for ResidueTemplate
            objects. For all other objects will return the default json encoding.

        Returns
        -------
        A JSON serializable object that represents the ResidueTemplate class or the default JSONEncoder output for an
        object.
        """
        if isinstance(obj, ResidueTemplate):
            output_dict = {
                "mol": rdMolInterchange.MolToJSON(obj.mol),
                "link_labels": obj.link_labels,
                "atom_names": obj.atom_names,
            }
            return output_dict
        return json.JSONEncoder.default(self, obj)


class ResiduePadderEncoder(json.JSONEncoder):
    """
    JSON Encoder class for ResiduePadder objects.
    """

    def default(self, obj):
        """
        Overrides the default JSON encoder for data structures for ResiduePadder objects.

        Parameters
        ----------
        obj: object
            Can take any object as input, but will only create the ResiduePadder JSON format for ResiduePadder
            objects. For all other objects will return the default json encoding.

        Returns
        -------
        A JSON serializable object that represents the ResiduePadder class or the default JSONEncoder output for an
        object.
        """
        if isinstance(obj, ResiduePadder):
            if obj.adjacent_smartsmol is None:
                adjacent_smarts = None
            else:
                # do not use JSON because it looses atom labels
                adjacent_smarts = Chem.MolToSmarts(obj.adjacent_smartsmol)
            output_dict = {
                "rxn_smarts": rdChemReactions.ReactionToSmarts(obj.rxn),
                "adjacent_smarts": adjacent_smarts,
                "auto_blunt": obj.auto_blunt,
            }
            # we are not serializing the adjacent_smartsmol_mapidx as that will
            # be rebuilt by the ResiduePadder init
            return output_dict
        return json.JSONEncoder.default(self, obj)


class ResidueChemTemplatesEncoder(json.JSONEncoder):
    """
    JSON Encoder class for ResidueChemTemplates objects.
    """

    residue_padder_encoder = ResiduePadderEncoder()
    residue_template_encoder = ResidueTemplateEncoder()

    def default(self, obj):
        """
        Overrides the default JSON encoder for data structures for ResidueChemTemplates objects.

        Parameters
        ----------
        obj: object
            Can take any object as input, but will only create the ResidueChemTemplates JSON format for
            ResidueChemTemplates objects. For all other objects will return the default json encoding.

        Returns
        -------
        A JSON serializable object that represents the ResidueChemTemplates class or the default JSONEncoder output for
        an object.
        """
        if isinstance(obj, ResidueChemTemplates):
            output_dict = {
                "residue_templates": {
                    k: self.residue_template_encoder.default(v)
                    for k, v in obj.residue_templates.items()
                },
                "ambiguous": obj.ambiguous,
                "padders": {
                    k: self.residue_padder_encoder.default(v)
                    for k, v in obj.padders.items()
                },
            }
            return output_dict
        return json.JSONEncoder.default(self, obj)


class PolymerEncoder(json.JSONEncoder):
    """
    JSON Encoder class for Polymer objects.
    """

    residue_chem_templates_encoder = ResidueChemTemplatesEncoder()
    monomer_encoder = MonomerEncoder()

    def default(self, obj):
        """
        Overrides the default JSON encoder for data structures for Polymer objects.

        Parameters
        ----------
        obj: object
            Can take any object as input, but will only create the Polymer JSON format for Polymer
            objects. For all other objects will return the default json encoding.

        Returns
        -------
        A JSON serializable object that represents the Polymer class or the default JSONEncoder output for an
        object.
        """
        if isinstance(obj, Polymer):
            output_dict = {
                "residue_chem_templates": self.residue_chem_templates_encoder.default(
                    obj.residue_chem_templates
                ),
                "monomers": {
                    k: self.monomer_encoder.default(v)
                    for k, v in obj.monomers.items()
                },
                "log": obj.log,
            }
            return output_dict
        return json.JSONEncoder.default(self, obj)


# endregion

