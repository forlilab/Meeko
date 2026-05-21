"""Atom dataclass for MoleculeSetup."""

from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np

from ..utils.jsonutils import BaseJSONParsable
from ..utils.pdbutils import PDBAtomInfo


DEFAULT_PDBINFO = None
DEFAULT_CHARGE = 0.0
DEFAULT_COORD = np.array([0.0, 0.0, 0.0], dtype="float")
DEFAULT_ATOMIC_NUM = None
DEFAULT_ATOM_TYPE = None
DEFAULT_IS_IGNORE = False
DEFAULT_GRAPH: list[int] = []


@dataclass
class Atom(BaseJSONParsable):
    index: int
    pdbinfo: Union[str, PDBAtomInfo] = DEFAULT_PDBINFO
    charge: float = DEFAULT_CHARGE
    coord: np.ndarray = field(default_factory=lambda: np.zeros(3))
    atomic_num: int = DEFAULT_ATOMIC_NUM
    atom_type: str = DEFAULT_ATOM_TYPE
    is_ignore: bool = DEFAULT_IS_IGNORE
    graph: list[int] = field(default_factory=list)

    is_dummy: bool = False
    is_pseudo_atom: bool = False

    @classmethod
    def json_encoder(cls, obj: "Atom") -> Optional[dict[str, Any]]:
        return {
            "index": obj.index,
            "pdbinfo": obj.pdbinfo,
            "charge": obj.charge,
            "coord": obj.coord.tolist(),
            "atomic_num": obj.atomic_num,
            "atom_type": obj.atom_type,
            "is_ignore": obj.is_ignore,
            "graph": obj.graph,
            "is_dummy": obj.is_dummy,
            "is_pseudo_atom": obj.is_pseudo_atom,
        }

    expected_json_keys = {
        "index",
        "pdbinfo",
        "charge",
        "coord",
        "atomic_num",
        "atom_type",
        "is_ignore",
        "graph",
        "is_dummy",
        "is_pseudo_atom",
    }

    @classmethod
    def _decode_object(cls, obj: dict[str, Any]):
        return cls(
            obj["index"],
            PDBAtomInfo(*obj["pdbinfo"]),
            obj["charge"],
            np.asarray(obj["coord"]),
            obj["atomic_num"],
            obj["atom_type"],
            obj["is_ignore"],
            obj["graph"],
            obj["is_dummy"],
            obj["is_pseudo_atom"],
        )
