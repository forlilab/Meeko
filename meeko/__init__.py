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

from .preparation import MoleculePreparation
from .setup import RDKitMoleculeSetup
from .utils import rdkitutils
from .utils import pdbutils
from .utils import geomutils
from .atomtyper import AtomTyper
from .receptor_pdbqt import PDBQTReceptor
from .molecule_pdbqt import PDBQTMolecule
from . import analysis

if _has_openbabel:
    from .setup import OBMoleculeSetup
    from .utils import obutils
    __all__ = ['MoleculePreparation', 'OBMoleculeSetup', 'RDKitMoleculeSetup',
               'pdbutils', 'obutils', 'geomutils', 'rdkitutils',
               'PDBQTMolecule', 'PDBQTReceptor', 'analysis']
else:
    __all__ = ['MoleculePreparation', 'OBMoleculeSetup', 'RDKitMoleculeSetup',
               'pdbutils', 'obutils', 'geomutils', 'rdkitutils',
               'PDBQTMolecule', 'PDBQTReceptor', 'analysis']
