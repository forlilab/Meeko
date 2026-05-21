"""Ring and RingClosureInfo dataclasses for MoleculeSetup."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from ..utils.jsonutils import BaseJSONParsable, string_to_tuple, tuple_to_string


DEFAULT_RING_CLOSURE_BONDS_REMOVED: list = []
DEFAULT_RING_CLOSURE_PSEUDOS_BY_ATOM = defaultdict


@dataclass
class Ring(BaseJSONParsable):
    ring_id: tuple

    @classmethod
    def json_encoder(cls, obj: "Ring") -> Optional[dict[str, Any]]:
        return {"ring_id": tuple_to_string(obj.ring_id)}

    expected_json_keys = {"ring_id"}

    @classmethod
    def _decode_object(cls, obj: dict[str, Any]):
        return cls(string_to_tuple(obj["ring_id"], int))


@dataclass
class RingClosureInfo:
    bonds_removed: list = field(default_factory=list)
    pseudos_by_atom: dict = DEFAULT_RING_CLOSURE_PSEUDOS_BY_ATOM
