"""meeko.molsetup package.

Re-exports the names that lived in the former single-file ``meeko/molsetup.py``
so that existing imports keep working::

    from meeko.molsetup import Atom, Bond, Ring, RingClosureInfo, Restraint
    from meeko.molsetup import MoleculeSetup, RDKitMoleculeSetup, UniqAtomParams

Internally, the module is now split into focused submodules.
"""

from .atom import (
    Atom,
    DEFAULT_ATOM_TYPE,
    DEFAULT_ATOMIC_NUM,
    DEFAULT_CHARGE,
    DEFAULT_COORD,
    DEFAULT_GRAPH,
    DEFAULT_IS_IGNORE,
    DEFAULT_PDBINFO,
)
from .bond import Bond, DEFAULT_BOND_BREAKABLE, DEFAULT_BOND_ROTATABLE
from .flex_model import FlexibilityModel
from .ring import (
    Ring,
    RingClosureInfo,
    DEFAULT_RING_CLOSURE_BONDS_REMOVED,
    DEFAULT_RING_CLOSURE_PSEUDOS_BY_ATOM,
)
from .restraint import Restraint
from .uniq_atom_params import UniqAtomParams
from .setup import MoleculeSetup, MoleculeSetupExternalToolkit, RDKitMoleculeSetup

__all__ = [
    "Atom",
    "Bond",
    "FlexibilityModel",
    "Ring",
    "RingClosureInfo",
    "Restraint",
    "UniqAtomParams",
    "MoleculeSetup",
    "MoleculeSetupExternalToolkit",
    "RDKitMoleculeSetup",
    "DEFAULT_ATOM_TYPE",
    "DEFAULT_ATOMIC_NUM",
    "DEFAULT_CHARGE",
    "DEFAULT_COORD",
    "DEFAULT_GRAPH",
    "DEFAULT_IS_IGNORE",
    "DEFAULT_PDBINFO",
    "DEFAULT_BOND_BREAKABLE",
    "DEFAULT_BOND_ROTATABLE",
    "DEFAULT_RING_CLOSURE_BONDS_REMOVED",
    "DEFAULT_RING_CLOSURE_PSEUDOS_BY_ATOM",
]
