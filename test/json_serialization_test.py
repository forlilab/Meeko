import numpy as np
import pathlib
import pytest
from rdkit import Chem
from rdkit.Chem import rdChemReactions
import meeko
from meeko import MoleculePreparation

# JSONParsable classes subject to serialization tests
from meeko import (
    Monomer,
    Polymer,
    RDKitMoleculeSetup,
    ResiduePadder,
    ResidueTemplate,
    ResidueChemTemplates,
)
from meeko.molsetup import Atom, Bond, Ring, Restraint

# Registry of class : set of attributes to skip for testing
EQUALITY_SKIP_FIELDS = { 
    RDKitMoleculeSetup: {"atom_true_count" },
    Monomer: {"template", "link_labels"},
}

# Optional dependency for test_dihedral_equality
try:
    import openforcefields
    _got_openff = True
except ImportError as err:
    _got_openff = False

# Test data: starting files for polymer creation
pkgdir = pathlib.Path(meeko.__file__).parents[1]
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
    """fixture for a populated polymer object"""
    with open(ahhy_example) as file:
        pdb_str = file.read()
    polymer = Polymer.from_pdb_string(
        pdb_str, chem_templates, mk_prep, blunt_ends=[("A:1", 0)]
    )
    return polymer

@pytest.fixture
def populated_polymer_missing():
    """fixture for a populated polymer object, with one residue missing"""
    with open(just_one_ALA_missing) as file: 
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
    """fixture for a populated ResidueChemTemplates object from default"""
    return ResidueChemTemplates.create_from_defaults()
# endregion


# region Helper Functions
def subobject_factory(cls, root):
    """
    Factory function to create subobjects based on the class and root object.

    Parameters
    ----------
    cls : type
        The class of the subobject to create.
    root : object
        The root object from which to create the subobject.
    
    Returns
    -------
    iterable
        An iterable of subobjects of the specified class.
    
    Raises
    ------
    ValueError
        If the class or root object is not recognized by given schema.
    """
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

def deep_assert_equal(decoded, original, path="root"):
    """Recursively compares two objects with support for type-aware handling and skip lists.
    
    Parameters
    ----------
    decoded : object
        The decoded object to compare.
    original : object
        The original object to compare against.
    path : str
        The current path in the object hierarchy for error reporting.
    
    Raises
    ------
    AssertionError
        If the objects are not equal or if there are type mismatches.
    """
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
        assert decoded_smiles == original_smiles, f"[{path}] Mol SMILES mismatch"
        return

    # RDKit Reactions
    if isinstance(decoded, rdChemReactions.ChemicalReaction):
        assert rdChemReactions.ReactionToSmarts(decoded) == rdChemReactions.ReactionToSmarts(original), f"[{path}] Reaction SMARTS mismatch"
        return

    # Custom objects with attributes
    if hasattr(decoded, "__dict__"):
        cls = type(decoded)
        skip_attrs = EQUALITY_SKIP_FIELDS.get(cls, set())

        # Check for extra attributes that are not in the original
        decoded_attr = set(dir(decoded))
        original_attr = set(dir(original))
        if decoded_attr - original_attr: 
            raise AssertionError(f"[{path}] Extra attributes in decoded object: {decoded_attr - original_attr}")

        for attr in original_attr:
            # skip private
            if attr.startswith("_"):
                continue
            # skip methods/functions/descriptors
            try:
                orig_val = getattr(original, attr)
            except Exception:
                continue  
            if callable(orig_val):
                continue
            # skip attributes if explicitly stated
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
    return
# endregion


# region Hierachical Tests

# iterate over nested classes in the Polymer hierarchy
@pytest.mark.parametrize("cls", [
    Polymer,
    Monomer,
    RDKitMoleculeSetup,
])
# check for seralization/deserialization and deep equality
def test_json_roundtrip(cls, populated_polymer):
    """Tests starting from a populated polymer object"""
    for obj in subobject_factory(cls, populated_polymer):
        json_str = obj.to_json()
        decoded = cls.from_json(json_str)
        assert isinstance(decoded, cls)
        deep_assert_equal(decoded, obj)

# same test for a polymer with missing residues
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

# same test but starting from the default ResidueChemTemplates object
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

# same test but iterating over the RDKitMoleculeSetup hierarchy
@pytest.mark.parametrize("cls", [
    RDKitMoleculeSetup,
    Atom,
    Bond,
    Ring,
    Restraint,
])
# the RDKitMoleculeSetup instances used for this test are created from the populated polymer
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
