import collections
import json
import meeko
import numpy as np
import pathlib
import pytest

from meeko import (
    Monomer,
    Polymer,
    MoleculePreparation,
    MoleculeSetup,
    RDKitMoleculeSetup,
    ResiduePadder,
    ResidueTemplate,
    ResidueChemTemplates,
    PDBQTWriterLegacy,
)

from meeko import polymer
from meeko.molsetup import Atom, Bond, Ring, RingClosureInfo, Restraint

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from meeko.utils.pdbutils import PDBAtomInfo

try:
    import openforcefields
    _got_openff = True
except ImportError as err:
    _got_openff = False

# from ..meeko.utils.pdbutils import PDBAtomInfo

pkgdir = pathlib.Path(meeko.__file__).parents[1]

# Test Data
ahhy_example = pkgdir / "test/polymer_data/AHHY.pdb"
just_one_ALA_missing = (
    pkgdir / "test/polymer_data/just-one-ALA-missing-CB.pdb"
)

# Polymer creation data
chem_templates = ResidueChemTemplates.create_from_defaults()
mk_prep = MoleculePreparation()


# region Fixtures
@pytest.fixture
def populated_polymer():
    file = open(ahhy_example)
    pdb_str = file.read()
    polymer = Polymer.from_pdb_string(
        pdb_str, chem_templates, mk_prep, blunt_ends=[("A:1", 0)]
    )
    return polymer


@pytest.fixture
def populated_polymer_missing():
    file = open(just_one_ALA_missing)
    pdb_str = file.read()
    polymer = Polymer.from_pdb_string(
        pdb_str,
        chem_templates,
        mk_prep,
        blunt_ends=[("A:1", 0), ("A:1", 2)],
        allow_bad_res=True,
    )
    return polymer

@pytest.fixture
def populated_residue_chem_templates():
    return ResidueChemTemplates.create_from_defaults()
# endregion

# region Standard Tests
EQUALITY_SKIP_FIELDS = { # Registry of attributes to skip per class
    RDKitMoleculeSetup: {"atom_true_count" },
    Monomer: {"template", "link_labels"},
}

def deep_assert_equal(decoded, original, path="root"):
    """Recursively compares two objects with support for type-aware handling and skip lists."""
    if type(decoded) != type(original):
        raise AssertionError(f"[{path}] Type mismatch: {type(decoded)} != {type(original)}")

    # Basic types
    if isinstance(decoded, (int, float, bool, str)):
        assert decoded == original, f"[{path}] Value mismatch: {decoded} != {original}"
        return

    # Dicts
    if isinstance(decoded, dict):
        assert decoded.keys() == original.keys(), f"[{path}] Dict keys mismatch"
        for key in decoded:
            deep_assert_equal(decoded[key], original[key], path=f"{path}.{key}")
        return

    # Lists or Tuples
    if isinstance(decoded, (list, tuple)):
        assert len(decoded) == len(original), f"[{path}] Length mismatch"
        for i, (d_item, o_item) in enumerate(zip(decoded, original)):
            deep_assert_equal(d_item, o_item, path=f"{path}[{i}]")
        return

    # Numpy arrays
    if isinstance(decoded, np.ndarray):
        assert np.allclose(decoded, original), f"[{path}] Numpy arrays not equal"
        return

    # RDKit Molecules
    if isinstance(decoded, Chem.Mol):
        decoded_smiles = Chem.MolToSmiles(decoded)
        original_smiles = Chem.MolToSmiles(original)
        if decoded_smiles != original_smiles:
            print(f"[DEBUG] Mol mismatch at {path}")
            print(f"Original: {original_smiles}")
            print(f"Decoded:  {decoded_smiles}")
            #import pdb; pdb.set_trace()  # Optional: step through interactively
        assert decoded_smiles == original_smiles, f"[{path}] Mol SMILES mismatch"

    # RDKit Reactions
    if isinstance(decoded, rdChemReactions.ChemicalReaction):
        assert rdChemReactions.ReactionToSmarts(decoded) == rdChemReactions.ReactionToSmarts(original), f"[{path}] Reaction SMARTS mismatch"
        return

    # Custom objects with attributes
    if hasattr(decoded, "__dict__"):
        cls = type(decoded)
        skip_attrs = EQUALITY_SKIP_FIELDS.get(cls, set())

        for attr in vars(original):
            if attr in skip_attrs:
                continue
            if not hasattr(decoded, attr):
                raise AssertionError(f"[{path}] Missing attribute: {attr}")
            decoded_val = getattr(decoded, attr)
            original_val = getattr(original, attr)
            deep_assert_equal(decoded_val, original_val, path=f"{path}.{attr}")
        return

    # Fallback
    assert decoded == original, f"[{path}] Fallback mismatch: {decoded} != {original}"

# region Standard Tests
def subobject_factory(cls, root):
    # Polymer-based hierarchy
    if isinstance(root, Polymer):
        if cls is Polymer:
            return [root]
        if cls is Monomer:
            return root.monomers.values()
        if cls is RDKitMoleculeSetup:
            return [m.molsetup for m in root.monomers.values()]

    # ResidueChemTemplates hierarchy
    if isinstance(root, ResidueChemTemplates):
        if cls is ResidueChemTemplates:
            return [root]
        if cls is ResidueTemplate:
            return root.residue_templates.values()
        if cls is ResiduePadder:
            return root.padders.values()

    # RDKitMoleculeSetup hierarchy
    if isinstance(root, RDKitMoleculeSetup):
        if cls is RDKitMoleculeSetup:
            return [root]
        if cls is Atom:
            return root.atoms
        if cls is Bond:
            return root.bond_info.values()
        if cls is Ring:
            return root.rings.values()
        if cls is Restraint:
            return root.restraints

    raise ValueError(f"Unexpected class or root: {cls}, {type(root)}")

@pytest.mark.parametrize("cls", [
    Polymer,
    Monomer,
    RDKitMoleculeSetup,
])
def test_json_roundtrip(cls, populated_polymer):
    for obj in subobject_factory(cls, populated_polymer):
        json_str = obj.to_json()
        decoded = cls.from_json(json_str)
        assert isinstance(decoded, cls)
        deep_assert_equal(decoded, obj)

@pytest.mark.parametrize("cls", [
    Polymer,
    Monomer,
    RDKitMoleculeSetup,
])
def test_json_roundtrip_missing(cls, populated_polymer_missing):
    for obj in subobject_factory(cls, populated_polymer_missing):
        if obj is None:
            continue
        json_str = obj.to_json()
        decoded = cls.from_json(json_str)
        assert isinstance(decoded, cls)
        deep_assert_equal(decoded, obj)

@pytest.mark.parametrize("cls", [
    ResidueChemTemplates,
    ResidueTemplate,
    ResiduePadder,
])
def test_json_rct(cls, populated_residue_chem_templates):
    for obj in subobject_factory(cls, populated_residue_chem_templates):
        json_str = obj.to_json()
        decoded = cls.from_json(json_str)
        assert isinstance(decoded, cls)
        deep_assert_equal(decoded, obj)

@pytest.mark.parametrize("cls", [
    RDKitMoleculeSetup,
    Atom,
    Bond,
    Ring,
    Restraint,
])
def test_json_molsetup(cls, populated_polymer):
    for molsetup in subobject_factory(RDKitMoleculeSetup, populated_polymer):
        for obj in subobject_factory(cls, molsetup):
            json_str = obj.to_json()
            decoded = cls.from_json(json_str)
            assert isinstance(decoded, cls)
            deep_assert_equal(decoded, obj)

# endregion

# region Other Tests
def test_load_reference_json():
    fn = str(pkgdir/"test"/"polymer_data"/"AHHY_reference_fewer_templates.json")
    with open(fn) as f:
        json_string = f.read()
    polymer = Polymer.from_json(json_string)
    assert len(polymer.get_valid_monomers()) == 4
    return


@pytest.mark.skipif(not _got_openff, reason="requires openff-forcefields")
def test_dihedral_equality():
    mk_prep = MoleculePreparation(
        merge_these_atom_types=(),
        dihedral_model="openff",
    )
    fn = str(pkgdir/"test"/"flexibility_data"/"non_sequential_atom_ordering_01.mol")
    mol = Chem.MolFromMolFile(fn, removeHs=False)
    starting_molsetup = mk_prep(mol)[0]
    json_str = starting_molsetup.to_json()
    decoded_molsetup = RDKitMoleculeSetup.from_json(json_str)
    deep_assert_equal(starting_molsetup, decoded_molsetup)
    return


def test_broken_bond(): 
    fn = str(pkgdir / "test" / "macrocycle_data" / "lorlatinib.mol")
    mol = Chem.MolFromMolFile(fn, removeHs=False)
    mk_prep_untyped = MoleculePreparation(untyped_macrocycles=True)
    starting_molsetup = mk_prep_untyped(mol)[0]
    decoded_molsetup = RDKitMoleculeSetup.from_json(starting_molsetup.to_json())
    count_rotatable = 0
    count_breakable = 0
    for bond_id, bond_info in decoded_molsetup.bond_info.items():
        count_rotatable += bond_info.rotatable
        count_breakable += bond_info.breakable
    assert count_rotatable == 10
    assert count_breakable == 1
# endregion
