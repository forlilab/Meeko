"""Typed container for a molecule's flexibility model.

Previously this state lived as an open-ended `dict` on
``MoleculeSetup.flexibility_model`` and the special tuple-keyed sub-dicts
were (de)serialized inline in ``MoleculeSetup.json_encoder`` /
``_decode_object``. The dict-key magic now lives here.

The class supports dict-style access (``model["key"]``, ``"key" in model``,
``bool(model)``) so existing call sites in ``flexibility.py``,
``preparation.py``, ``writer.py`` and ``polymer.py`` continue to work
unchanged.
"""

from dataclasses import dataclass, field, fields
from typing import Any, Optional

from ..utils.jsonutils import (
    convert_to_int_keyed_dict,
    string_to_tuple,
    tuple_to_string,
)


# Field names whose values are dicts keyed by ints (need int-key restoration
# when round-tripping through JSON, since JSON object keys are always strings).
_INT_KEYED_DICT_FIELDS = frozenset(
    {"rigid_body_graph", "rigid_body_members", "rigid_index_by_atom"}
)


@dataclass
class FlexibilityModel:
    """Rigid-body decomposition produced by ``flexibility.get_flexibility_model``.

    Required structural fields are populated during the graph walk; optional
    fields are filled in by downstream code (e.g. preparation.py sets
    ``torsions_org``; the macrocycle-break branch sets ``score``).
    """

    visited: list = field(default_factory=list)
    rigid_body_count: int = 0
    rigid_index_by_atom: dict = field(default_factory=dict)
    rigid_body_members: dict = field(default_factory=dict)
    rigid_body_connectivity: dict = field(default_factory=dict)
    rigid_body_graph: dict = field(default_factory=dict)

    root: Optional[int] = None
    score: Optional[float] = None
    torsions_org: Optional[int] = None

    # ------ dict-style access (backward compat) ------

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        if not hasattr(self, key):
            raise KeyError(
                f"FlexibilityModel has no field {key!r}. "
                f"Known fields: {[f.name for f in fields(self)]}"
            )
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) and getattr(self, key) is not None

    def __bool__(self) -> bool:
        # Truthy once the model has been populated, matching the legacy
        # `not setup.flexibility_model` test against an empty/None dict.
        return self.rigid_body_count > 0 or bool(self.rigid_body_members)

    def __iter__(self):
        # Iterate field names whose value is not None — mirrors dict-key
        # iteration where missing optional keys (e.g. ``score``) are skipped.
        return self.keys()

    def items(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            yield f.name, value

    def keys(self):
        for k, _ in self.items():
            yield k

    # ------ JSON interchange ------

    def encode(self) -> dict:
        """Serialize to a JSON-friendly dict.

        ``rigid_body_connectivity`` is keyed by tuples of ints, which JSON
        doesn't support natively, so the keys are stringified here.
        """
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            if f.name == "rigid_body_connectivity":
                out[f.name] = {tuple_to_string(k): v for k, v in value.items()}
            else:
                out[f.name] = value
        return out

    @classmethod
    def decode(cls, data) -> "FlexibilityModel":
        """Rebuild from the dict produced by ``encode``.

        Tolerates plain-dict input (legacy JSON written before this class
        existed) and FlexibilityModel-encoded input alike.
        """
        if data is None:
            return cls()
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            value = data[f.name]
            if f.name == "rigid_body_connectivity":
                value = {
                    string_to_tuple(k, int): tuple(string_to_tuple(v))
                    if isinstance(v, str)
                    else tuple(v)
                    for k, v in value.items()
                }
            elif f.name in _INT_KEYED_DICT_FIELDS:
                value = convert_to_int_keyed_dict(value)
            kwargs[f.name] = value
        return cls(**kwargs)
