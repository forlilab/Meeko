#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko
#

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
from .utils import rdkitutils
from .utils import pdbutils
from .utils import geomutils
from .utils import utils
from .atomtyper import AtomTyper
from .receptor_pdbqt import PDBQTReceptor
from .molecule_pdbqt import PDBQTMolecule
from .rdkit_mol_create import RDKitMolCreate
from . import analysis
from .writer import oids_block_from_setup
from .openff_xml_parser import parse_offxml
from .openff_xml_parser import load_openff
from .openff_xml_parser import get_openff_epsilon_sigma
from .hydrate import Hydrate

__all__ = ['MoleculePreparation', 'RDKitMoleculeSetup',
        'pdbutils', 'geomutils', 'rdkitutils', 'utils',
        'AtomTyper', 'PDBQTMolecule', 'PDBQTReceptor', 'analysis',
        'RDKitMolCreate',
        'oids_block_from_setup',
        'parse_offxml',
        'Hydrate',
        ]

if _has_openbabel:
    from .molsetup import OBMoleculeSetup
    from .utils import obutils
    __all__.append("OBMoleculeSetup")
    __all__.append("obutils")

if _has_prody:
    from .covalentbuilder import CovalentBuilder
    __all__.append("CovalentBuilder")
