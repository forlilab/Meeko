#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko
#

__version__ = "0.7.1"

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
from .polymer import Polymer
from .polymer import Monomer
from .polymer import ResiduePadder
from .polymer import ResidueTemplate
from .polymer import ResidueChemTemplates
from .polymer import add_rotamers_to_polymer_molsetups
from .polymer import PolymerCreationError
from .molecule_pdbqt import PDBQTMolecule
from .rdkit_mol_create import RDKitMolCreate
from .export_flexres import export_pdb_updated_flexres
from .reactive import reactive_typer
from .reactive import get_reactive_config
from .writer import PDBQTWriterLegacy
from . import analysis
from .writer import oids_block_from_setup
from .openff_xml_parser import parse_offxml
from .openff_xml_parser import load_openff
from .openff_xml_parser import get_openff_epsilon_sigma
from .hydrate import Hydrate

import logging
from rdkit import rdBase
rdkit_logger = logging.getLogger("rdkit")
rdkit_logger.handlers[0].setLevel("WARNING")
rdkit_logger.handlers[0].setFormatter(
    logging.Formatter('[RDKit] %(levelname)s:%(message)s'),
)
rdBase.LogToPythonLogger()

__all__ = ['MoleculePreparation', 'RDKitMoleculeSetup', 
           'pdbutils', 'geomutils', 'rdkitutils', 'utils',
           'AtomTyper', 'PDBQTMolecule', 'PDBQTReceptor', 'analysis',
           'Polymer', 'Monomer', 'ResiduePadder', 'ResidueTemplate', 'ResidueChemTemplates',
           'add_rotamers_to_polymer_molsetups',
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
