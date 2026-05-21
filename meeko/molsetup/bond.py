"""Bond dataclass for MoleculeSetup."""

from dataclasses import dataclass, field
from typing import Any, Optional

from ..utils.jsonutils import BaseJSONParsable, tuple_to_string


DEFAULT_BOND_ROTATABLE = False
DEFAULT_BOND_BREAKABLE = False


@dataclass
class Bond(BaseJSONParsable):
    canon_id: tuple[int, int] = field(init=False)
    index1: int
    index2: int
    rotatable: bool = DEFAULT_BOND_ROTATABLE
    breakable: bool = DEFAULT_BOND_BREAKABLE

    def __post_init__(self):
        self.canon_id = self.get_bond_id(self.index1, self.index2)

    @classmethod
    def json_encoder(cls, obj: "Bond") -> Optional[dict[str, Any]]:
        return {
            "canon_id": tuple_to_string(obj.canon_id),
            "index1": obj.index1,
            "index2": obj.index2,
            "rotatable": obj.rotatable,
            "breakable": obj.breakable,
        }

    expected_json_keys = {"canon_id", "index1", "index2", "rotatable"}

    @classmethod
    def _decode_object(cls, obj: dict[str, Any]):
        return cls(
            obj["index1"],
            obj["index2"],
            obj["rotatable"],
            obj.get("breakable", DEFAULT_BOND_BREAKABLE),
        )

    @staticmethod
    def get_bond_id(idx1: int, idx2: int) -> tuple[int, int]:
        """Canonical (sorted) atom-index pair identifying a bond."""
        return (min(idx1, idx2), max(idx1, idx2))
