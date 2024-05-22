import collections
import json
import meeko
import numpy
import pathlib
import pytest

from meeko import (
    ChorizoResidue,
    ChorizoResidueEncoder,
    LinkedRDKitChorizo,
    LinkedRDKitChorizoEncoder,
    MoleculePreparation,
    MoleculeSetup,
    MoleculeSetupEncoder,
    RDKitMoleculeSetup,
    ResiduePadder,
    ResiduePadderEncoder,
    ResidueTemplate,
    ResidueTemplateEncoder,
    ResidueChemTemplates,
    ResidueChemTemplatesEncoder,
    Restraint,
)

from meeko import linked_rdkit_chorizo

from rdkit import Chem
from rdkit.Chem import rdChemReactions

# from ..meeko.utils.pdbutils import PDBAtomInfo

pkgdir = pathlib.Path(meeko.__file__).parents[1]
meekodir = pathlib.Path(meeko.__file__).parents[0]

# Test Data
ahhy_example = pkgdir / "example/chorizo/AHHY.pdb"

# Chorizo creation data
with open(meekodir / "data" / "residue_chem_templates.json") as f:
    t = json.load(f)
chem_templates = ResidueChemTemplates.from_dict(t)
mk_prep = MoleculePreparation()


# region Fixtures


@pytest.fixture
def populated_rdkit_chorizo():
    file = open(ahhy_example)
    pdb_str = file.read()
    chorizo = LinkedRDKitChorizo.from_pdb_string(pdb_str, chem_templates, mk_prep)
    return chorizo


@pytest.fixture
def populated_rdkit_chorizo_residue(populated_rdkit_chorizo):
    chorizo = populated_rdkit_chorizo
    return chorizo.residues["A:1"]


@pytest.fixture
def populated_rdkit_molsetup(populated_rdkit_chorizo_residue):
    residue = populated_rdkit_chorizo_residue
    return residue.molsetup


@pytest.fixture
def populated_residue_chem_templates(populated_rdkit_chorizo):
    chorizo = populated_rdkit_chorizo
    return chorizo.residue_chem_templates


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
    # TODO: Certain fields are empty in this example, and if we want to make sure that json is working in all scenarios
    # we will need to make other tests for those empty fields.
    # Encode and decode MoleculeSetup from json
    starting_molsetup = populated_rdkit_molsetup
    json_str = json.dumps(starting_molsetup, cls=MoleculeSetupEncoder)
    decoded_molsetup = json.loads(
        json_str, object_hook=MoleculeSetup.molsetup_json_decoder
    )

    # First asserts that all types are as expected
    assert isinstance(starting_molsetup, RDKitMoleculeSetup)
    assert isinstance(decoded_molsetup, RDKitMoleculeSetup)

    # Go through MoleculeSetup attributes and check that they are the expected type and match the MoleculeSetup object
    # before serialization.
    check_molsetup_equality(decoded_molsetup, starting_molsetup)
    return


def test_chorizo_residue_encoding_decoding(populated_rdkit_chorizo_residue):
    # Starts by getting a chorizo residue object, converting it to a json string, and then decoding the string into
    # a new chorizo residue object
    starting_residue = populated_rdkit_chorizo_residue
    json_str = json.dumps(starting_residue, cls=ChorizoResidueEncoder)
    decoded_residue = json.loads(
        json_str, object_hook=linked_rdkit_chorizo.chorizo_residue_json_decoder
    )

    # Asserts that the starting and ending objects have the expected ChorizoResidue type
    assert isinstance(starting_residue, ChorizoResidue)
    assert isinstance(decoded_residue, ChorizoResidue)

    check_residue_equality(decoded_residue, starting_residue)
    return


def test_residue_template_encoding_decoding(populated_residue_template):
    # Starts by getting a ResidueTemplate object, converting it to a json string, and then decoding the string into
    # a new ResidueTemplate object
    starting_template = populated_residue_template
    json_str = json.dumps(starting_template, cls=ResidueTemplateEncoder)
    decoded_template = json.loads(
        json_str, object_hook=linked_rdkit_chorizo.residue_template_json_decoder
    )

    # Asserts that the starting and ending objects have the expected ResidueTemplate type
    assert isinstance(starting_template, ResidueTemplate)
    assert isinstance(decoded_template, ResidueTemplate)

    # Checks that the two residue templates are equal
    check_residue_template_equality(decoded_template, starting_template)
    return


def test_residue_padder_encoding_decoding(populated_residue_padder):
    # Starts by getting a ResiduePadder object, converting it to a json string, and then decoding the string into
    # a new ResiduePadder object
    starting_padder = populated_residue_padder
    json_str = json.dumps(starting_padder, cls=ResiduePadderEncoder)
    decoded_padder = json.loads(
        json_str, object_hook=linked_rdkit_chorizo.residue_padder_json_decoder
    )

    # Asserts that the starting and ending objects have the expected ResiduePadder type
    assert isinstance(starting_padder, ResiduePadder)
    assert isinstance(decoded_padder, ResiduePadder)

    # Checks that the two residue padders are equal
    check_residue_padder_equality(decoded_padder, starting_padder)
    return


def test_residue_chem_templates_encoding_decoding(populated_residue_chem_templates):
    # Starts by getting a ResidueChemTemplates object, converting it to a json string, and then decoding the string into
    # a new ResidueChemTemplates object
    starting_templates = populated_residue_chem_templates
    json_str = json.dumps(starting_templates, cls=ResidueChemTemplatesEncoder)
    decoded_templates = json.loads(
        json_str, object_hook=linked_rdkit_chorizo.residue_chem_templates_json_decoder
    )

    # Asserts that the starting and ending objects have the expected ResidueChemTemplates type
    assert isinstance(starting_templates, ResidueChemTemplates)
    assert isinstance(decoded_templates, ResidueChemTemplates)

    # Checks that the two chem templates are equal
    check_residue_chem_templates_equality(decoded_templates, starting_templates)
    return


def test_linked_rdkit_chorizo_encoding_decoding(populated_rdkit_chorizo):
    # Starts by getting a LinkedRDKitChorizo object, converting it to a json string, and then decoding the string into
    # a new LinkedRDKitChorizo object
    starting_chorizo = populated_rdkit_chorizo
    json_str = json.dumps(starting_chorizo, cls=LinkedRDKitChorizoEncoder)
    decoded_chorizo = json.loads(
        json_str, object_hook=linked_rdkit_chorizo.linked_rdkit_chorizo_json_decoder
    )

    # Asserts that the starting and ending objects have the expected LinkedRDKitChorizo type
    assert isinstance(starting_chorizo, LinkedRDKitChorizo)
    assert isinstance(decoded_chorizo, LinkedRDKitChorizo)

    # Checks that the two chorizos are equal
    check_linked_rdkit_chorizo_equality(decoded_chorizo, starting_chorizo)
    return


# endregion


# region Object Equality Checks
def check_molsetup_equality(decoded_obj: MoleculeSetup, starting_obj: MoleculeSetup):

    # Bool used while looping through values to check whether all values in a data structure have the expected type
    correct_val_type = True

    # First checks the attributes that we are not doing any type conversion on

    # Next checks attributes that needed minimal type conversion

    # Finally goes through attributes that needed complex interventions

    assert decoded_obj.atom_pseudo == starting_obj.atom_pseudo  # EMPTY
    assert isinstance(decoded_obj.coord, collections.OrderedDict)
    assert decoded_obj.coord.keys() == starting_obj.coord.keys()
    for key in decoded_obj.coord:
        correct_val_type = correct_val_type & isinstance(
            decoded_obj.coord[key], numpy.ndarray
        )
        assert set(decoded_obj.coord[key]) == set(starting_obj.coord[key])
    assert correct_val_type
    assert decoded_obj.charge == starting_obj.charge
    assert decoded_obj.pdbinfo == starting_obj.pdbinfo
    correct_val_type = True
    # for key in decoded_molsetup.pdbinfo:
    #    correct_val_type = correct_val_type & isinstance(decoded_molsetup.pdbinfo[key], PDBAtomInfo)
    assert correct_val_type
    assert decoded_obj.atom_type == starting_obj.atom_type
    assert (
        decoded_obj.atom_params == starting_obj.atom_params
    )  # WILL THERE BE CONVERSIONS NEEDED FOR PARAMS?
    assert (
        decoded_obj.dihedral_interactions == starting_obj.dihedral_interactions
    )  # EMPTY
    assert (
        decoded_obj.dihedral_partaking_atoms == starting_obj.dihedral_partaking_atoms
    )  # EMPTY
    assert decoded_obj.dihedral_labels == starting_obj.dihedral_labels  # EMPTY
    assert decoded_obj.atom_ignore == starting_obj.atom_ignore
    assert decoded_obj.chiral == starting_obj.chiral
    assert decoded_obj.atom_true_count == starting_obj.atom_true_count
    assert decoded_obj.graph == starting_obj.graph
    # Assert that the starting object's bond attribute was not compromised in the serialization process:
    assert isinstance(list(starting_obj.bond.keys())[0], tuple)
    # Assert that the final object's bond attribute had the expected key type:
    assert isinstance(list(decoded_obj.bond.keys())[0], tuple)
    assert decoded_obj.bond == starting_obj.bond
    assert decoded_obj.element == starting_obj.element
    assert decoded_obj.interaction_vector == starting_obj.interaction_vector  # EMPTY
    # Assert that both the starting and final objects have keys that are the expected type
    if "rigid_body_connectivity" in starting_obj.flexibility_model:
        assert isinstance(
            list(starting_obj.flexibility_model["rigid_body_connectivity"].keys())[0],
            tuple,
        )
        assert isinstance(
            list(decoded_obj.flexibility_model["rigid_body_connectivity"].keys())[0],
            tuple,
        )
    assert decoded_obj.flexibility_model == starting_obj.flexibility_model
    assert decoded_obj.ring_closure_info == starting_obj.ring_closure_info  # EMPTY
    assert decoded_obj.restraints == starting_obj.restraints  # EMPTY
    assert decoded_obj.is_sidechain == starting_obj.is_sidechain
    assert decoded_obj.rmsd_symmetry_indices == starting_obj.rmsd_symmetry_indices
    # Assert that the starting object's bond attribute was not compromised in the serialization process:
    assert isinstance(list(starting_obj.bond.keys())[0], tuple)
    # Assert that the final object's bond attribute had the expected key type:
    assert isinstance(list(decoded_obj.bond.keys())[0], tuple)
    assert decoded_obj.rings == starting_obj.rings
    assert decoded_obj.rings_aromatic == starting_obj.rings_aromatic
    assert decoded_obj.atom_to_ring_id == starting_obj.atom_to_ring_id
    assert decoded_obj.ring_corners == starting_obj.ring_corners  # EMPTY
    assert decoded_obj.name == starting_obj.name  # EMPTY
    assert decoded_obj.rotamers == starting_obj.rotamers  # EMPTY
    return


def check_residue_equality(decoded_obj: ChorizoResidue, starting_obj: ChorizoResidue):
    # Goes through the Chorizo Residue's fields and checks that they are the expected type and match the ChorizoResidue
    # object before serialization (that we have effectively rebuilt the ChorizoResidue)

    # RDKit Mols - Check whether we can test for equality with RDKit Mols
    # assert decoded_residue.raw_rdkit_mol == starting_residue.raw_rdkit_mol
    assert isinstance(decoded_obj.raw_rdkit_mol, Chem.rdchem.Mol)
    # assert decoded_residue.rdkit_mol == starting_residue.rdkit_mol
    assert isinstance(decoded_obj.rdkit_mol, Chem.rdchem.Mol)
    # assert decoded_residue.padded_mol == starting_residue.padded_mol
    assert isinstance(decoded_obj.padded_mol, Chem.rdchem.Mol)

    # MapIDX
    assert decoded_obj.mapidx_to_raw == starting_obj.mapidx_to_raw
    assert decoded_obj.mapidx_from_raw == starting_obj.mapidx_from_raw

    # Non-Bool vars
    assert decoded_obj.residue_template_key == starting_obj.residue_template_key
    assert decoded_obj.input_resname == starting_obj.input_resname
    assert decoded_obj.atom_names == starting_obj.atom_names
    check_molsetup_equality(decoded_obj.molsetup, starting_obj.molsetup)
    assert isinstance(decoded_obj.molsetup, RDKitMoleculeSetup)

    # Bools
    assert decoded_obj.is_flexres_atom == starting_obj.is_flexres_atom
    assert decoded_obj.is_movable == starting_obj.is_movable
    assert decoded_obj.user_deleted == starting_obj.user_deleted
    return


def check_residue_chem_templates_equality(
    decoded_obj: ResidueChemTemplates, starting_obj: ResidueChemTemplates
):
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
    starting_obj
        A ResiduePadder object with the desired values to check the decoded object against.
    Returns
    -------
    None

    """
    assert isinstance(decoded_obj.rxn, rdChemReactions.ChemicalReaction)
    decoded_obj_rxn_smarts = rdChemReactions.ReactionToSmarts(decoded_obj.rxn)
    starting_obj_rxn_smarts = rdChemReactions.ReactionToSmarts(starting_obj.rxn)
    assert decoded_obj_rxn_smarts == starting_obj_rxn_smarts

    assert isinstance(decoded_obj.adjacent_smartsmol, Chem.rdchem.Mol)
    assert (
        decoded_obj.adjacent_smartsmol_mapidx == starting_obj.adjacent_smartsmol_mapidx
    )
    return


def check_linked_rdkit_chorizo_equality(
    decoded_obj: LinkedRDKitChorizo, starting_obj: LinkedRDKitChorizo
):
    # correct_val_type is used to check that all type conversions for nested data have happened correctly
    correct_val_type = True
    # Checks residue_chem_templates equality
    check_residue_chem_templates_equality(decoded_obj.residue_chem_templates, starting_obj.residue_chem_templates)

    # Loops through residues, checks that the decoded and starting obj share the same set of keys, that all the residues
    # are represented as ChorizoResidue objects, and that the decoding and starting obj ChorizoResidues are equal.
    assert decoded_obj.residues.keys() == starting_obj.residues.keys()
    for key in decoded_obj.residues:
        correct_val_type = correct_val_type & isinstance(
            decoded_obj.residues[key], ChorizoResidue
        )
        check_residue_equality(decoded_obj.residues[key], starting_obj.residues[key])
    assert correct_val_type

    # Checks log equality
    assert decoded_obj.log == starting_obj.log
    return


# endregion
