import collections
import json
import meeko
import numpy
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
ahhy_v061_json = pkgdir / "test/polymer_data/AHHY-v0.6.1.json"
just_one_ALA_missing = (
    pkgdir / "test/polymer_data/just-one-ALA-missing-CB.pdb"
)

# Polymer creation data
chem_templates = ResidueChemTemplates.create_from_defaults()
mk_prep = MoleculePreparation()

def test_read_v061_polymer():
    with open(ahhy_v061_json) as f:
        json_str = f.read()
    polymer = Polymer.from_json(json_str)
    return

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
def populated_monomer(populated_polymer):
    polymer = populated_polymer
    return polymer.monomers["A:1"]


@pytest.fixture
def populated_rdkit_molsetup(populated_monomer):
    monomer = populated_monomer
    return monomer.molsetup


@pytest.fixture
def populated_residue_chem_templates(populated_polymer):
    polymer = populated_polymer
    return polymer.residue_chem_templates


@pytest.fixture
def populated_residue_template(populated_residue_chem_templates):
    res_chem_templates = populated_residue_chem_templates
    return res_chem_templates.residue_templates["G"]


@pytest.fixture
def populated_residue_padder(populated_residue_chem_templates):
    res_chem_templates = populated_residue_chem_templates
    return res_chem_templates.padders["5-prime"]


# endregion


# region Test Cases
def test_rdkit_molsetup_encoding_decoding(populated_rdkit_molsetup):
    """
    Takes a fully populated RDKitMoleculeSetup, checks that it can be serialized to JSON and deserialized back into an
    object without any errors, then checks that the deserialized object matches the starting object and that the
    attribute types, values, and structure of the deserialized object are as expected for an RDKitMoleculeSetup.

    Parameters
    ----------
    populated_rdkit_molsetup: RDKitMoleculeSetup
        Takes as input a populated RDKitMoleculeSetup object.

    Returns
    -------
    None
    """
    # TODO: Certain fields are empty in this example, and if we want to make sure that json is working in all scenarios
    # we will need to make other tests for those empty fields.
    # Encode and decode MoleculeSetup from json
    starting_molsetup = populated_rdkit_molsetup
    json_str = starting_molsetup.to_json() 
    decoded_molsetup = RDKitMoleculeSetup.from_json(json_str)

    # First asserts that all types are as expected
    assert isinstance(starting_molsetup, RDKitMoleculeSetup)
    assert isinstance(decoded_molsetup, RDKitMoleculeSetup)

    # Go through MoleculeSetup attributes and check that they are the expected type and match the MoleculeSetup object
    # before serialization.
    check_molsetup_equality(decoded_molsetup, starting_molsetup)
    return


def test_monomer_encoding_decoding(populated_monomer):
    """
    Takes a fully populated Monomer, checks that it can be serialized to JSON and deserialized back into an
    object without any errors, then checks that the deserialized object matches the starting object and that the
    attribute types, values, and structure of the deserialized object are as expected for an Monomer.

    Parameters
    ----------
    populated_monomer: Monomer
        Takes as input a populated Monomer object.

    Returns
    -------
    None
    """
    # Starts by getting a Monomer object, converting it to a json string, and then decoding the string into
    # a new Monomer object
    starting_monomer = populated_monomer
    json_str = starting_monomer.to_json()

    decoded_monomer = Monomer.from_json(json_str)

    # Asserts that the starting and ending objects have the expected Monomer type
    assert isinstance(starting_monomer, Monomer)
    assert isinstance(decoded_monomer, Monomer)

    check_monomer_equality(decoded_monomer, starting_monomer)
    return


def test_pdbqt_writing_from_decoded_polymer(populated_polymer):
    """
    Takes a fully populated Polymer, writes a PDBQT string from it, encodes and decodes it, writes
    another PDBQT string from the decoded polymer, and then checks that the PDBQT strings are identical.

    Parameters
    ----------
    populated_polymer: Polymer
        Takes as input a populated Polymer object.

    Returns
    -------
    None
    """

    starting_polymer = populated_polymer
    starting_pdbqt = PDBQTWriterLegacy.write_from_polymer(starting_polymer)
    json_str = starting_polymer.to_json()
    decoded_polymer = Polymer.from_json(json_str)
    decoded_pdbqt = PDBQTWriterLegacy.write_from_polymer(decoded_polymer) 
    assert decoded_pdbqt == starting_pdbqt
    return



def test_residue_template_encoding_decoding(populated_residue_template):
    """
    Takes a fully populated ResidueTemplate, checks that it can be serialized to JSON and deserialized back into an
    object without any errors, then checks that the deserialized object matches the starting object and that the
    attribute types, values, and structure of the deserialized object are as expected for an ResidueTemplate.

    Parameters
    ----------
    populated_residue_template: ResidueTemplate
        Takes as input a populated ResidueTemplate object.

    Returns
    -------
    None
    """
    # Starts by getting a ResidueTemplate object, converting it to a json string, and then decoding the string into
    # a new ResidueTemplate object
    starting_template = populated_residue_template
    json_str = starting_template.to_json()
    decoded_template = ResidueTemplate.from_json(json_str)

    # Asserts that the starting and ending objects have the expected ResidueTemplate type
    assert isinstance(starting_template, ResidueTemplate)
    assert isinstance(decoded_template, ResidueTemplate)

    # Checks that the two residue templates are equal
    check_residue_template_equality(decoded_template, starting_template)
    return


def test_residue_padder_encoding_decoding(populated_residue_padder):
    """
    Takes a fully populated ResiduePadder, checks that it can be serialized to JSON and deserialized back into an
    object without any errors, then checks that the deserialized object matches the starting object and that the
    attribute types, values, and structure of the deserialized object are as expected for an ResiduePadder.

    Parameters
    ----------
    populated_residue_padder: ResiduePadder
        Takes as input a populated ResiduePadder object.

    Returns
    -------
    None
    """
    # Starts by getting a ResiduePadder object, converting it to a json string, and then decoding the string into
    # a new ResiduePadder object
    starting_padder = populated_residue_padder
    json_str = starting_padder.to_json()
    decoded_padder = ResiduePadder.from_json(json_str)

    # Asserts that the starting and ending objects have the expected ResiduePadder type
    assert isinstance(starting_padder, ResiduePadder)
    assert isinstance(decoded_padder, ResiduePadder)

    # Checks that the two residue padders are equal
    check_residue_padder_equality(decoded_padder, starting_padder)
    return


def test_residue_chem_templates_encoding_decoding(populated_residue_chem_templates):
    """
    Takes a fully populated ResidueChemTemplates, checks that it can be serialized to JSON and deserialized back into an
    object without any errors, then checks that the deserialized object matches the starting object and that the
    attribute types, values, and structure of the deserialized object are as expected for an ResidueChemTemplates.

    Parameters
    ----------
    populated_residue_chem_templates: ResidueChemTemplates
        Takes as input a populated ResidueChemTemplates object.

    Returns
    -------
    None
    """
    # Starts by getting a ResidueChemTemplates object, converting it to a json string, and then decoding the string into
    # a new ResidueChemTemplates object
    starting_templates = populated_residue_chem_templates
    json_str = starting_templates.to_json()
    decoded_templates = ResidueChemTemplates.from_json(json_str)

    # Asserts that the starting and ending objects have the expected ResidueChemTemplates type
    assert isinstance(starting_templates, ResidueChemTemplates)
    assert isinstance(decoded_templates, ResidueChemTemplates)

    # Checks that the two chem templates are equal
    check_residue_chem_templates_equality(decoded_templates, starting_templates)
    return


def test_polymer_encoding_decoding(
    populated_polymer, populated_polymer_missing
):
    """
    Takes a fully populated Polymer, checks that it can be serialized to JSON and deserialized back into an
    object without any errors, then checks that the deserialized object matches the starting object and that the
    attribute types, values, and structure of the deserialized object are as expected for a Polymer.

    Parameters
    ----------
    populated_polymer: Polymer
        Takes as input a populated Polymer object.

    Returns
    -------
    None
    """
    # Starts by getting a Polymer object, converting it to a json string, and then decoding the string into
    # a new Polymer object
    polymers = (
        populated_polymer,
        populated_polymer_missing,
    )
    for starting_polymer in polymers:
        json_str = starting_polymer.to_json()
        decoded_polymer  = Polymer.from_json(json_str)

        # Asserts that the starting and ending objects have the expected Polymer type
        assert isinstance(starting_polymer, Polymer)
        assert isinstance(decoded_polymer, Polymer)

        # Checks that the two polymers are equal
        check_polymer_equality(decoded_polymer, starting_polymer)
    return


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
    check_molsetup_equality(starting_molsetup, decoded_molsetup)
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
    assert count_rotatable == 9
    assert count_breakable == 1

# endregion


# region Object Equality Checks
def check_molsetup_equality(decoded_obj: MoleculeSetup, starting_obj: MoleculeSetup):
    """
    Asserts that two MoleculeSetup objects are equal, and that the decoded_obj input has fields contain correctly typed
    data.

    Parameters
    ----------
    decoded_obj: MoleculeSetup
        A MoleculeSetup object that we want to check is correctly typed and contains the correct data.
    starting_obj: MoleculeSetup
        A MoleculeSetup object with the desired values to check the decoded object against.

    Returns
    -------
    None
    """

    # Checks if the MoleculeSetup is an RDKitMoleculeSetup, and if so also checks the RDKitMoleculeSetup attributes
    if isinstance(starting_obj, RDKitMoleculeSetup):
        assert isinstance(decoded_obj.mol, Chem.rdchem.Mol)
        pass

    # Going through and checking MoleculeSetup attributes
    assert decoded_obj.name == starting_obj.name
    assert isinstance(decoded_obj.pseudoatom_count, int)
    assert decoded_obj.pseudoatom_count == starting_obj.pseudoatom_count

    # Checking atoms
    atom_idx = 0
    assert len(decoded_obj.atoms) == len(starting_obj.atoms)
    for atom in decoded_obj.atoms:
        assert isinstance(atom, Atom)
        assert atom.index == atom_idx
        check_atom_equality(atom, starting_obj.atoms[atom_idx])
        atom_idx += 1

    # Checking bonds
    for bond_id in starting_obj.bond_info:
        assert isinstance(decoded_obj.bond_info[bond_id], Bond)
        assert bond_id in decoded_obj.bond_info
        check_bond_equality(
            decoded_obj.bond_info[bond_id], starting_obj.bond_info[bond_id]
        )

    # Checking rings
    for ring_id in starting_obj.rings:
        assert isinstance(decoded_obj.rings[ring_id], Ring)
        assert ring_id in decoded_obj.rings
        check_ring_equality(decoded_obj.rings[ring_id], starting_obj.rings[ring_id])
    assert isinstance(decoded_obj.ring_closure_info, RingClosureInfo)
    assert (
        decoded_obj.ring_closure_info.bonds_removed
        == starting_obj.ring_closure_info.bonds_removed
    )
    for key in starting_obj.ring_closure_info.pseudos_by_atom:
        assert key in decoded_obj.ring_closure_info.pseudos_by_atom
        assert (
            decoded_obj.ring_closure_info.pseudos_by_atom[key]
            == starting_obj.ring_closure_info.pseudos_by_atom[key]
        )

    # Checking other fields
    assert len(decoded_obj.rotamers) == len(starting_obj.rotamers)
    for idx, component_dict in enumerate(starting_obj.rotamers):
        decoded_dict = decoded_obj.rotamers[idx]
        for key in component_dict:
            assert key in decoded_dict
            assert decoded_dict[key] == component_dict[key]
    for key in starting_obj.atom_params:
        assert key in decoded_obj.atom_params
        assert decoded_obj.atom_params[key] == starting_obj.atom_params[key]
    assert len(decoded_obj.restraints) == len(starting_obj.restraints)
    for idx, restraint in starting_obj.restraints:
        assert isinstance(decoded_obj.restraints[idx], Restraint)
        check_restraint_equality(
            decoded_obj.restraints[idx], starting_obj.restraints[idx]
        )

    # dihedrals
    assert decoded_obj.dihedral_partaking_atoms == starting_obj.dihedral_partaking_atoms
    assert decoded_obj.dihedral_interactions == starting_obj.dihedral_interactions
    assert decoded_obj.dihedral_labels == starting_obj.dihedral_labels

    # Checking flexibility model
    for key in starting_obj.flexibility_model:
        assert key in decoded_obj.flexibility_model
        assert decoded_obj.flexibility_model[key] == starting_obj.flexibility_model[key]
    return


def check_atom_equality(decoded_obj: Atom, starting_obj: Atom):
    """
    Asserts that two Atom objects are equal, and that the decoded_obj input has fields contain correctly typed
    data.

    Parameters
    ----------
    decoded_obj: Atom
        An Atom object that we want to check is correctly typed and contains the correct data.
    starting_obj: Atom
        An Atom object with the desired values to check the decoded object against.

    Returns
    -------
    None
    """
    # np.array conversion checks
    assert isinstance(decoded_obj.coord, numpy.ndarray)

    # Checks for equality between decoded and original fields
    assert isinstance(decoded_obj.index, int)
    assert decoded_obj.index == starting_obj.index
    # Only checks pdb info if the starting object's pdbinfo was a string. Otherwise, the decoder is not going to convert
    # the pdbinfo field back to the PDBInfo type right now.
    if isinstance(starting_obj.pdbinfo, str):
        assert decoded_obj.pdbinfo == starting_obj.pdbinfo
    assert isinstance(decoded_obj.charge, float)
    assert decoded_obj.charge == starting_obj.charge
    for idx, val in enumerate(decoded_obj.coord):
        assert val == starting_obj.coord[idx]
    assert isinstance(decoded_obj.atomic_num, int)
    assert decoded_obj.atomic_num == starting_obj.atomic_num
    assert decoded_obj.atom_type == starting_obj.atom_type
    assert decoded_obj.graph == starting_obj.graph
    assert isinstance(decoded_obj.is_ignore, bool)
    assert decoded_obj.is_ignore == starting_obj.is_ignore
    assert isinstance(decoded_obj.is_dummy, bool)
    assert decoded_obj.is_dummy == starting_obj.is_dummy
    assert isinstance(decoded_obj.is_pseudo_atom, bool)
    assert decoded_obj.is_pseudo_atom == starting_obj.is_pseudo_atom
    return


def check_bond_equality(decoded_obj: Bond, starting_obj: Bond):
    """
    Asserts that two Bond objects are equal, and that the decoded_obj input has fields contain correctly typed
    data.

    Parameters
    ----------
    decoded_obj: Bond
        An Bond object that we want to check is correctly typed and contains the correct data.
    starting_obj: Bond
        An Bond object with the desired values to check the decoded object against.

    Returns
    -------
    None
    """
    assert isinstance(decoded_obj.canon_id, tuple)
    assert isinstance(decoded_obj.canon_id[0], int)
    assert isinstance(decoded_obj.canon_id[1], int)
    assert decoded_obj.canon_id == starting_obj.canon_id
    assert isinstance(decoded_obj.index1, int)
    assert decoded_obj.index1 == starting_obj.index1
    assert isinstance(decoded_obj.index2, int)
    assert decoded_obj.index2 == starting_obj.index2
    assert isinstance(decoded_obj.rotatable, bool)
    assert decoded_obj.rotatable == starting_obj.rotatable
    return


def check_ring_equality(decoded_obj: Ring, starting_obj: Ring):
    """
    Asserts that two Ring objects are equal, and that the decoded_obj input has fields contain correctly typed
    data.

    Parameters
    ----------
    decoded_obj: Ring
        An Ring object that we want to check is correctly typed and contains the correct data.
    starting_obj: Ring
        An Ring object with the desired values to check the decoded object against.

    Returns
    -------
    None
    """
    assert isinstance(decoded_obj.ring_id, tuple)
    assert decoded_obj.ring_id == starting_obj.ring_id
    return


def check_restraint_equality(decoded_obj: Restraint, starting_obj: Restraint):
    """
    Asserts that two Restraint objects are equal, and that the decoded_obj input has fields contain correctly typed
    data.

    Parameters
    ----------
    decoded_obj: Restraint
        An Restraint object that we want to check is correctly typed and contains the correct data.
    starting_obj: Restraint
        An Restraint object with the desired values to check the decoded object against.

    Returns
    -------
    None
    """
    assert isinstance(decoded_obj.atom_index, int)
    assert decoded_obj.atom_index == starting_obj.atom_index
    assert isinstance(decoded_obj.target_coords, tuple)
    assert decoded_obj.target_coords == starting_obj.target_coords
    assert isinstance(decoded_obj.kcal_per_angstrom_square, float)
    assert decoded_obj.kcal_per_angstrom_square == starting_obj.kcal_per_angstrom_square
    assert isinstance(decoded_obj.delay_angstroms, float)
    assert decoded_obj.delay_angstroms == starting_obj.delay_angstroms
    return


def check_monomer_equality(decoded_obj: Monomer, starting_obj: Monomer):
    """
    Asserts that two Monomer objects are equal, and that the decoded_obj input has fields contain correctly typed
    data.

    Parameters
    ----------
    decoded_obj: Monomer
        A Monomer object that we want to check is correctly typed and contains the correct data.
    starting_obj: Monomer
        A Monomer object with the desired values to check the decoded object against.

    Returns
    -------
    None
    """
    # Goes through the Monomer's fields and checks that they are the expected type and match the Monomer
    # object before serialization (that we have effectively rebuilt the Monomer)

    # RDKit Mols - Check whether we can test for equality with RDKit Mols
    # assert decoded_monomer.raw_rdkit_mol == starting_residue.raw_rdkit_mol
    assert type(decoded_obj.raw_rdkit_mol) == type(starting_obj.raw_rdkit_mol)
    if isinstance(decoded_obj.raw_rdkit_mol, Chem.rdchem.Mol):
        assert Chem.MolToSmiles(decoded_obj.raw_rdkit_mol) == Chem.MolToSmiles(
            starting_obj.raw_rdkit_mol
        )
    # assert decoded_monomer.rdkit_mol == starting_monomer.rdkit_mol
    assert type(decoded_obj.rdkit_mol) == type(starting_obj.rdkit_mol)
    if isinstance(decoded_obj.rdkit_mol, Chem.rdchem.Mol):
        assert Chem.MolToSmiles(decoded_obj.rdkit_mol) == Chem.MolToSmiles(
            starting_obj.rdkit_mol
        )
    # assert decoded_monomer.padded_mol == starting_monomer.padded_mol
    assert type(decoded_obj.padded_mol) == type(starting_obj.padded_mol)
    if isinstance(decoded_obj.padded_mol, Chem.rdchem.Mol):
        assert Chem.MolToSmiles(decoded_obj.padded_mol) == Chem.MolToSmiles(
            starting_obj.padded_mol
        )

    # MapIDX
    assert decoded_obj.mapidx_to_raw == starting_obj.mapidx_to_raw
    assert decoded_obj.mapidx_from_raw == starting_obj.mapidx_from_raw

    # Non-Bool vars
    assert decoded_obj.residue_template_key == starting_obj.residue_template_key
    assert decoded_obj.input_resname == starting_obj.input_resname
    assert decoded_obj.atom_names == starting_obj.atom_names
    assert type(decoded_obj.molsetup) == type(starting_obj.molsetup)
    if isinstance(decoded_obj.molsetup, RDKitMoleculeSetup):
        check_molsetup_equality(decoded_obj.molsetup, starting_obj.molsetup)

    # Bools
    assert decoded_obj.is_flexres_atom == starting_obj.is_flexres_atom
    assert decoded_obj.is_movable == starting_obj.is_movable
    return


def check_residue_chem_templates_equality(
    decoded_obj: ResidueChemTemplates, starting_obj: ResidueChemTemplates
):
    """
    Asserts that two ResidueChemTemplates objects are equal, and that the decoded_obj input has fields contain correctly
    typed data.

    Parameters
    ----------
    decoded_obj: ResidueChemTemplates
        A ResidueChemTemplates object that we want to check is correctly typed and contains the correct data.
    starting_obj: ResidueChemTemplates
        A ResidueChemTemplates object with the desired values to check the decoded object against.

    Returns
    -------
    None
    """
    # correct_val_type is used to check that all type conversions for nested data have happened correctly
    correct_val_type = True
    # Checks residue_templates by ensuring it has the same members as the starting object, that each value in the
    # dictionary is a ResidueTemplate object, and that each template is equal to its corresponding ResidueTemplate in
    # the starting object.
    assert decoded_obj.residue_templates.keys() == starting_obj.residue_templates.keys()
    for key in decoded_obj.residue_templates:
        correct_val_type = correct_val_type & isinstance(
            decoded_obj.residue_templates[key], ResidueTemplate
        )
        check_residue_template_equality(
            decoded_obj.residue_templates[key], starting_obj.residue_templates[key]
        )
    assert correct_val_type

    # Directly compares ambiguous values.
    assert decoded_obj.ambiguous == starting_obj.ambiguous

    # Checks padders by ensuring it has the same members as the starting object, that each value in the dictionary is a
    # ResiduePadder object, and that each padder is equal to its corresponding ResiduePadder in the starting object.
    assert decoded_obj.padders.keys() == starting_obj.padders.keys()
    for key in decoded_obj.padders:
        correct_val_type = correct_val_type & isinstance(
            decoded_obj.padders[key], ResiduePadder
        )
        check_residue_padder_equality(
            decoded_obj.padders[key], starting_obj.padders[key]
        )
    assert correct_val_type
    return


def check_residue_template_equality(
    decoded_obj: ResidueTemplate, starting_obj: ResidueTemplate
):
    """
    Asserts that two ResidueTemplate objects are equal, and that the decoded_obj input has fields contain correctly typed
    data.

    Parameters
    ----------
    decoded_obj: ResidueTemplate
        A ResidueTemplate object that we want to check is correctly typed and contains the correct data.
    starting_obj: ResidueTemplate
        A ResidueTemplate object with the desired values to check the decoded object against.

    Returns
    -------
    None
    """
    # Goes through the ResidueTemplate's fields and checks that they have the expected type and that they match the
    # ResidueTemplate object before serialization
    assert isinstance(decoded_obj.mol, Chem.rdchem.Mol)

    assert decoded_obj.link_labels == starting_obj.link_labels
    assert decoded_obj.atom_names == starting_obj.atom_names
    return


def check_residue_padder_equality(
    decoded_obj: ResiduePadder, starting_obj: ResiduePadder
):
    """
    Asserts that two ResiduePadder objects are equal, and that the decoded_obj input has fields contain correctly typed
    data.

    Parameters
    ----------
    decoded_obj: ResiduePadder
        A ResiduePadder object that we want to check is correctly typed and contains the correct data.
    starting_obj: ResiduePadder
        A ResiduePadder object with the desired values to check the decoded object against.

    Returns
    -------
    None
    """
    assert isinstance(decoded_obj.rxn, rdChemReactions.ChemicalReaction)
    decoded_obj_rxn_smarts = rdChemReactions.ReactionToSmarts(decoded_obj.rxn)
    starting_obj_rxn_smarts = rdChemReactions.ReactionToSmarts(starting_obj.rxn)
    assert decoded_obj_rxn_smarts == starting_obj_rxn_smarts

    assert (
        decoded_obj.adjacent_smartsmol_mapidx == starting_obj.adjacent_smartsmol_mapidx
    )

    decoded_adj = decoded_obj.adjacent_smartsmol
    starting_adj = starting_obj.adjacent_smartsmol
    assert isinstance(decoded_adj, Chem.rdchem.Mol) or decoded_adj is None
    if decoded_adj is None:
        assert decoded_adj == starting_adj
    else:
        decoded_adj_smarts = Chem.MolToSmarts(decoded_adj)
        starting_adj_smarts = Chem.MolToSmarts(starting_adj)
        assert decoded_adj_smarts == starting_adj_smarts
    return


def check_polymer_equality(
    decoded_obj: Polymer, starting_obj: Polymer
):
    """
    Asserts that two Polymer objects are equal, and that the decoded_obj input has fields contain correctly
    typed data.

    Parameters
    ----------
    decoded_obj: Polymer
        A Polymer object that we want to check is correctly typed and contains the correct data.
    starting_obj: Polymer
        A Polymer object with the desired values to check the decoded object against.

    Returns
    -------
    None
    """
    # correct_val_type is used to check that all type conversions for nested data have happened correctly
    correct_val_type = True
    # Checks residue_chem_templates equality
    check_residue_chem_templates_equality(
        decoded_obj.residue_chem_templates, starting_obj.residue_chem_templates
    )

    # Loops through residues, checks that the decoded and starting obj share the same set of keys, that all the residues
    # are represented as Monomer objects, and that the decoding and starting obj Monomers are equal.
    assert decoded_obj.monomers.keys() == starting_obj.monomers.keys()
    for key in decoded_obj.monomers:
        correct_val_type = correct_val_type & isinstance(
            decoded_obj.monomers[key], Monomer
        )
        check_monomer_equality(decoded_obj.monomers[key], starting_obj.monomers[key])
    assert correct_val_type

    # Checks log equality
    assert decoded_obj.log == starting_obj.log
    return

# endregion
