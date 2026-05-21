"""``PrepConfig`` dataclass — typed configuration for ``MoleculePreparation``.

The previous shape was a 30-argument constructor on ``MoleculePreparation``.
This module gives those settings a typed home so they can be passed,
inspected, serialized, and validated as a single object.

``MoleculePreparation`` still accepts the same kwargs and remains the
entry point; internally it now stores a ``PrepConfig`` instance.
"""

import json
from dataclasses import MISSING, dataclass, field, fields, asdict
from typing import Any, Optional

import meeko.macrocycle


_ALLOWED_CHARGE_MODELS = ("espaloma", "gasteiger", "zero", "read", "nagl")


@dataclass
class PrepConfig:
    """Typed configuration for ``MoleculePreparation``.

    Every field corresponds 1:1 to a constructor kwarg of
    ``MoleculePreparation``; the defaults match exactly so existing
    callers keep their behavior.
    """

    merge_these_atom_types: tuple = ("H",)
    merge_rmin_half: bool = False
    hydrate: bool = False
    flexible_amides: bool = False
    rigid_macrocycles: bool = False
    untyped_macrocycles: bool = False
    min_ring_size: int = meeko.macrocycle.DEFAULT_MIN_RING_SIZE
    max_ring_size: int = meeko.macrocycle.DEFAULT_MAX_RING_SIZE
    keep_chorded_rings: bool = False
    keep_equivalent_rings: bool = False
    double_bond_penalty: float = meeko.macrocycle.DEFAULT_DOUBLE_BOND_PENALTY
    macrocycle_allow_A: bool = False
    rigidify_bonds_smarts: list = field(default_factory=list)
    rigidify_bonds_indices: list = field(default_factory=list)
    input_atom_params: Optional[dict] = None
    load_atom_params: Any = "ad4_types"
    add_atom_types: Optional[list] = None
    input_offatom_params: Optional[dict] = None
    load_offatom_params: Optional[Any] = None
    charge_model: str = "gasteiger"
    charge_atom_prop: Optional[str] = None
    dihedral_model: Optional[str] = None
    reactive_smarts: Optional[str] = None
    reactive_smarts_idx: Optional[int] = None
    add_index_map: bool = False
    remove_smiles: bool = False
    compute_charges: bool = False
    crippen: bool = False
    crippen_as_solpar: bool = False
    override_ad4sol_par_including_q: bool = False
    override_ad4sol_par_including_q_qasp: float = 0.0

    def __post_init__(self):
        if type(self.merge_these_atom_types) not in (list, set, tuple):
            raise ValueError(
                "you probably forgot '.from_config' in MoleculePreparation.from_config(mk_config)"
            )

        if self.charge_model not in _ALLOWED_CHARGE_MODELS:
            raise ValueError(
                "unrecognized charge_model: %s, allowed options are: %s"
                % (self.charge_model, list(_ALLOWED_CHARGE_MODELS))
            )

        if self.load_offatom_params is not None:
            raise NotImplementedError("load_offatom_params not implemented")

        if (self.reactive_smarts is None) != (self.reactive_smarts_idx is None):
            raise ValueError(
                "reactive_smarts and reactive_smarts_idx require each other"
            )

    # ----- factories -----

    @classmethod
    def from_dict(cls, data: dict) -> "PrepConfig":
        """Build a ``PrepConfig`` from a kwargs dict (mirrors
        ``MoleculePreparation.from_config``). Unknown keys raise.
        """
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise TypeError(
                f"Unexpected PrepConfig field(s): {sorted(unknown)}. "
                f"Known: {sorted(known)}"
            )
        return cls(**data)

    @classmethod
    def from_json_file(cls, filename) -> "PrepConfig":
        with open(filename) as f:
            return cls.from_dict(json.loads(f.read()))

    def to_dict(self) -> dict:
        return asdict(self)

    # ----- defaults helper (used by CLI) -----

    @classmethod
    def get_defaults_dict(cls) -> dict:
        """Return ``{field_name: default_value}`` for every field.

        Mirrors ``MoleculePreparation.get_defaults_dict()`` but reads from
        the dataclass instead of ``inspect.signature``.
        """
        defaults: dict = {}
        for f in fields(cls):
            if f.default is not MISSING:
                defaults[f.name] = f.default
            elif f.default_factory is not MISSING:
                defaults[f.name] = f.default_factory()
            else:
                defaults[f.name] = None
        return defaults
