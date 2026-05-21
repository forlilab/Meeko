"""JSON (de)serialization for MoleculeSetup and RDKitMoleculeSetup.

The two classes still expose ``json_encoder`` / ``_decode_object``
classmethods (the ``BaseJSONParsable`` protocol used by the rest of the
codebase), but the per-field encode/decode logic now lives here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from rdkit.Chem import rdMolInterchange

from ..utils.jsonutils import (
    convert_to_tuple_keyed_dict,
    rdkit_mol_from_json,
    string_to_tuple,
    tuple_to_string,
)
from .atom import Atom
from .bond import Bond
from .flex_model import FlexibilityModel
from .restraint import Restraint
from .ring import Ring, RingClosureInfo

if TYPE_CHECKING:
    from .setup import MoleculeSetup, RDKitMoleculeSetup


MOLSETUP_JSON_KEYS = {
    "name",
    "pseudoatom_count",
    "atoms",
    "bond_info",
    "rings",
    "ring_closure_info",
    "rotamers",
    "atom_params",
    "restraints",
    "flexibility_model",
}

RDKIT_MOLSETUP_EXTRA_KEYS = {
    "mol",
    "modified_atom_positions",
    "dihedral_interactions",
    "dihedral_partaking_atoms",
    "dihedral_labels",
    "atom_to_ring_id",
    "rmsd_symmetry_indices",
}


def encode_molecule_setup(obj: "MoleculeSetup") -> Optional[dict[str, Any]]:
    return {
        "name": obj.name,
        "pseudoatom_count": obj.pseudoatom_count,
        "atoms": [Atom.json_encoder(x) for x in obj.atoms],
        "bond_info": {
            tuple_to_string(k): Bond.json_encoder(v) for k, v in obj.bond_info.items()
        },
        "rings": {
            tuple_to_string(k): Ring.json_encoder(v) for k, v in obj.rings.items()
        },
        "ring_closure_info": obj.ring_closure_info.__dict__,
        "rotamers": [
            {tuple_to_string(k): v for k, v in rotamer.items()}
            for rotamer in obj.rotamers
        ],
        "atom_params": obj.atom_params,
        "restraints": [Restraint.json_encoder(x) for x in obj.restraints],
        "flexibility_model": obj.flexibility_model.encode()
        if isinstance(obj.flexibility_model, FlexibilityModel)
        else obj.flexibility_model,
    }


def decode_molecule_setup(cls, obj: dict[str, Any]) -> "MoleculeSetup":
    molsetup = cls(obj["name"])
    molsetup.pseudoatom_count = obj["pseudoatom_count"]
    molsetup.atoms = [Atom.from_dict(x) for x in obj["atoms"]]
    molsetup.bond_info = {
        string_to_tuple(k, int): Bond.from_dict(v)
        for k, v in obj["bond_info"].items()
    }
    molsetup.rings = {
        string_to_tuple(k, int): Ring.from_dict(v) for k, v in obj["rings"].items()
    }
    molsetup.ring_closure_info = RingClosureInfo(
        obj["ring_closure_info"]["bonds_removed"],
        obj["ring_closure_info"]["pseudos_by_atom"],
    )
    molsetup.rotamers = [
        convert_to_tuple_keyed_dict(rotamer, int) for rotamer in obj["rotamers"]
    ]
    molsetup.atom_params = obj["atom_params"]
    molsetup.restraints = [Restraint.from_dict(x) for x in obj["restraints"]]
    molsetup.flexibility_model = FlexibilityModel.decode(obj["flexibility_model"])
    return molsetup


def encode_rdkit_molecule_setup(obj: "RDKitMoleculeSetup") -> Optional[dict[str, Any]]:
    output_dict = encode_molecule_setup(obj)
    output_dict["mol"] = rdMolInterchange.MolToJSON(obj.mol)
    output_dict["modified_atom_positions"] = obj.modified_atom_positions
    output_dict["dihedral_interactions"] = obj.dihedral_interactions
    output_dict["dihedral_partaking_atoms"] = {
        tuple_to_string(k): v for k, v in obj.dihedral_partaking_atoms.items()
    }
    output_dict["dihedral_labels"] = {
        tuple_to_string(k): v for k, v in obj.dihedral_labels.items()
    }
    output_dict["atom_to_ring_id"] = obj.atom_to_ring_id
    output_dict["rmsd_symmetry_indices"] = obj.rmsd_symmetry_indices
    return output_dict


def decode_rdkit_molecule_setup(cls, obj: dict[str, Any]) -> "RDKitMoleculeSetup":
    # Import locally to avoid a circular import: setup.py imports io.py via
    # the class methods, but only at call time, not at class-creation time.
    from .setup import MoleculeSetup
    base_molsetup = MoleculeSetup.from_dict(obj)
    rdkit_molsetup = cls(source=base_molsetup)
    rdkit_molsetup.mol = rdkit_mol_from_json(obj["mol"])
    rdkit_molsetup.modified_atom_positions = list(map(int, obj["modified_atom_positions"]))
    rdkit_molsetup.dihedral_interactions = obj["dihedral_interactions"]
    rdkit_molsetup.dihedral_partaking_atoms = convert_to_tuple_keyed_dict(
        obj["dihedral_partaking_atoms"], int
    )
    rdkit_molsetup.dihedral_labels = convert_to_tuple_keyed_dict(
        obj["dihedral_labels"], int
    )
    rdkit_molsetup.atom_to_ring_id = {
        int(k): [string_to_tuple(t) for t in v]
        for k, v in obj["atom_to_ring_id"].items()
    }
    rdkit_molsetup.rmsd_symmetry_indices = list(
        map(string_to_tuple, obj["rmsd_symmetry_indices"])
    )
    return rdkit_molsetup
