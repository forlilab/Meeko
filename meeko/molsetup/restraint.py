"""Restraint dataclass for MoleculeSetup."""

from dataclasses import dataclass
from typing import Any, Optional

from ..utils.jsonutils import BaseJSONParsable, tuple_to_string


@dataclass
class Restraint(BaseJSONParsable):
    atom_index: int
    target_coords: tuple[float, float, float]
    kcal_per_angstrom_square: float
    delay_angstroms: float

    @classmethod
    def json_encoder(cls, obj: "Restraint") -> Optional[dict[str, Any]]:
        return {
            "atom_index": obj.atom_index,
            "target_coords": tuple_to_string(obj.target_coords),
            "kcal_per_angstrom_square": obj.kcal_per_angstrom_square,
            "delay_angstroms": obj.delay_angstroms,
        }

    expected_json_keys = {
        "atom_index",
        "target_coords",
        "kcal_per_angstrom_square",
        "delay_angstroms",
    }

    @classmethod
    def _decode_object(cls, obj: dict[str, Any]):
        return cls(
            obj["atom_index"],
            tuple(obj["target_coords"]),
            obj["kcal_per_angstrom_square"],
            obj["delay_angstroms"],
        )

    def copy(self):
        return Restraint(
            self.atom_index,
            (self.target_coords[0], self.target_coords[1], self.target_coords[2]),
            self.kcal_per_angstrom_square,
            self.delay_angstroms,
        )
