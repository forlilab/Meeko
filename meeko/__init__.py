#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko
#

__version__ = "0.6.0-alpha.3"

try:
    import prody
except ImportError:
    _has_prody = False
else:
    _has_prody = True

from .preparation import MoleculePreparation
from .molsetup import RDKitMoleculeSetup
from .molsetup import MoleculeSetup
from .molsetup import MoleculeSetupEncoder
from .molsetup import Restraint
from .molsetup import UniqAtomParams
from .utils import rdkitutils
from .utils import pdbutils
from .utils import geomutils
from .utils import utils
from .atomtyper import AtomTyper
from .receptor_pdbqt import PDBQTReceptor
from .linked_rdkit_chorizo import LinkedRDKitChorizo
from .linked_rdkit_chorizo import ChorizoResidue
from .linked_rdkit_chorizo import ResiduePadder
from .linked_rdkit_chorizo import ResidueTemplate
from .linked_rdkit_chorizo import ResidueChemTemplates
from .linked_rdkit_chorizo import LinkedRDKitChorizoEncoder
from .linked_rdkit_chorizo import ChorizoResidueEncoder
from .linked_rdkit_chorizo import ResiduePadderEncoder
from .linked_rdkit_chorizo import ResidueTemplateEncoder
from .linked_rdkit_chorizo import ResidueChemTemplatesEncoder
from .linked_rdkit_chorizo import add_rotamers_to_chorizo_molsetups
from .linked_rdkit_chorizo import linked_rdkit_chorizo_json_decoder
from .molecule_pdbqt import PDBQTMolecule
from .rdkit_mol_create import RDKitMolCreate
from .reactive import reactive_typer
from .reactive import get_reactive_config
from .writer import PDBQTWriterLegacy
from . import analysis
from .writer import oids_block_from_setup
from .openff_xml_parser import parse_offxml
from .openff_xml_parser import load_openff
from .openff_xml_parser import get_openff_epsilon_sigma
from .hydrate import Hydrate

__all__ = ['MoleculePreparation', 'RDKitMoleculeSetup', 'MoleculeSetupEncoder',
           'pdbutils', 'geomutils', 'rdkitutils', 'utils',
           'AtomTyper', 'PDBQTMolecule', 'PDBQTReceptor', 'analysis',
           'LinkedRDKitChorizo', 'ChorizoResidue', 'ResiduePadder', 'ResidueTemplate', 'ResidueChemTemplates',
           'LinkedRDKitChorizoEncoder', 'ChorizoResidueEncoder', 'ResiduePadderEncoder', 'ResidueTemplateEncoder',
           'ResidueChemTemplatesEncoder',
           'add_rotamers_to_chorizo_molsetups',
           'RDKitMolCreate',
           'PDBQTWriterLegacy',
           'reactive_typer',
           'get_reactive_config',
           'gridbox',
           'oids_block_from_setup',
           'parse_offxml',
           'Hydrate',
           'Restraint',
           ]

if _has_prody:
    from .covalentbuilder import CovalentBuilder
    __all__.append("CovalentBuilder")
