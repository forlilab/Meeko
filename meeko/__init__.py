#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko
#

__version__ = "0.6.0-alpha"

try:
    import openbabel
except ImportError:
    _has_openbabel = False
else:
    _has_openbabel = True

try:
    import prody
except ImportError:
    _has_prody = False
else:
    _has_prody = True

from .preparation import MoleculePreparation
from .molsetup import RDKitMoleculeSetup
from .molsetup import MoleculeSetup
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
from .linked_rdkit_chorizo import ResidueAdditionalConnection
from .linked_rdkit_chorizo import add_rotamers_to_chorizo_molsetups
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

__all__ = ['MoleculePreparation', 'RDKitMoleculeSetup',
        'pdbutils', 'geomutils', 'rdkitutils', 'utils',
        'AtomTyper', 'PDBQTMolecule', 'PDBQTReceptor', 'analysis',
        'LinkedRDKitChorizo', 'ChorizoResidue', 'ResidueAdditionalConnection',
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

if _has_openbabel:
    from .molsetup import OBMoleculeSetup
    from .utils import obutils
    __all__.append("OBMoleculeSetup")
    __all__.append("obutils")

if _has_prody:
    from .covalentbuilder import CovalentBuilder
    __all__.append("CovalentBuilder")
