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
    ResidueChemTemplates,
    Restraint,
)

from rdkit import Chem

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


# Fixtures
# @pytest.fixture
def populated_rdkit_molsetup():
    file = open(ahhy_example)
    pdb_str = file.read()
    chorizo = LinkedRDKitChorizo.from_pdb_string(pdb_str, chem_templates, mk_prep)
    residue = chorizo.residues["A:1"]
    return residue.molsetup


def populated_rdkit_chorizo_residue():
    file = open(ahhy_example)
    pdb_str = file.read()
    chorizo = LinkedRDKitChorizo.from_pdb_string(pdb_str, chem_templates, mk_prep)
    return chorizo.residues["A:1"]


# Test Cases
def test_rdkit_molsetup_encoding_decoding():
    # TODO: Certain fields are empty in this example, and if we want to make sure that json is working in all scenarios
    # we will need to make other tests for those empty fields.
    # Encode and decode molsetup from json
    starting_molsetup = populated_rdkit_molsetup()
    json_str = json.dumps(starting_molsetup, cls=MoleculeSetupEncoder)
    decoded_molsetup = json.loads(
        json_str, object_hook=MoleculeSetup.molsetup_json_decoder
    )

    # First asserts that all types are as expected
    assert isinstance(starting_molsetup, RDKitMoleculeSetup)
    assert isinstance(decoded_molsetup, RDKitMoleculeSetup)

    # Go through molsetup attributes and check that they are the expected type and match the molsetup object
    # before serialization.
    check_molsetup_equality(decoded_molsetup, starting_molsetup)
    return


def test_chorizo_residue_encoding_decoding():
    # Starts by getting a chorizo residue object, converting it to a json string, and then decoding the string into
    # a new chorizo residue object
    starting_residue = populated_rdkit_chorizo_residue()
    json_str = json.dumps(starting_residue, cls=ChorizoResidueEncoder)
    decoded_residue = json.loads(
        json_str, object_hook=ChorizoResidue.chorizo_residue_json_decoder
    )

    # Asserts that the starting and ending objects have the expected ChorizoResidue type
    assert isinstance(starting_residue, ChorizoResidue)
    assert isinstance(decoded_residue, ChorizoResidue)

    check_residue_equality(decoded_residue, starting_residue)
    return


def check_molsetup_equality(
    decoded_molsetup: MoleculeSetup, starting_molsetup: MoleculeSetup
):

    # Bool used while looping through values to check whether all values in a data structure have the expected type
    correct_val_type = True

    # First checks the attributes that we are not doing any type conversion on

    # Next checks attributes that needed minimal type conversion

    # Finally goes through attributes that needed complex interventions

    assert decoded_molsetup.atom_pseudo == starting_molsetup.atom_pseudo  # EMPTY
    assert isinstance(decoded_molsetup.coord, collections.OrderedDict)
    assert decoded_molsetup.coord.keys() == starting_molsetup.coord.keys()
    for key in decoded_molsetup.coord:
        correct_val_type = correct_val_type & isinstance(
            decoded_molsetup.coord[key], numpy.ndarray
        )
        assert set(decoded_molsetup.coord[key]) == set(starting_molsetup.coord[key])
    assert correct_val_type
    assert decoded_molsetup.charge == starting_molsetup.charge
    assert decoded_molsetup.pdbinfo == starting_molsetup.pdbinfo
    correct_val_type = True
    # for key in decoded_molsetup.pdbinfo:
    #    correct_val_type = correct_val_type & isinstance(decoded_molsetup.pdbinfo[key], PDBAtomInfo)
    assert correct_val_type
    assert decoded_molsetup.atom_type == starting_molsetup.atom_type
    assert (
        decoded_molsetup.atom_params == starting_molsetup.atom_params
    )  # WILL THERE BE CONVERSIONS NEEDED FOR PARAMS?
    assert (
        decoded_molsetup.dihedral_interactions
        == starting_molsetup.dihedral_interactions
    )  # EMPTY
    assert (
        decoded_molsetup.dihedral_partaking_atoms
        == starting_molsetup.dihedral_partaking_atoms
    )  # EMPTY
    assert (
        decoded_molsetup.dihedral_labels == starting_molsetup.dihedral_labels
    )  # EMPTY
    assert decoded_molsetup.atom_ignore == starting_molsetup.atom_ignore
    assert decoded_molsetup.chiral == starting_molsetup.chiral
    assert decoded_molsetup.atom_true_count == starting_molsetup.atom_true_count
    assert decoded_molsetup.graph == starting_molsetup.graph
    # Assert that the starting object's bond attribute was not compromised in the serialization process:
    assert isinstance(list(starting_molsetup.bond.keys())[0], tuple)
    # Assert that the final object's bond attribute had the expected key type:
    assert isinstance(list(decoded_molsetup.bond.keys())[0], tuple)
    assert decoded_molsetup.bond == starting_molsetup.bond
    assert decoded_molsetup.element == starting_molsetup.element
    assert (
        decoded_molsetup.interaction_vector == starting_molsetup.interaction_vector
    )  # EMPTY
    # Assert that both the starting and final objects have keys that are the expected type
    if "rigid_body_connectivity" in starting_molsetup.flexibility_model:
        assert isinstance(
            list(starting_molsetup.flexibility_model["rigid_body_connectivity"].keys())[
                0
            ],
            tuple,
        )
        assert isinstance(
            list(decoded_molsetup.flexibility_model["rigid_body_connectivity"].keys())[
                0
            ],
            tuple,
        )
    assert decoded_molsetup.flexibility_model == starting_molsetup.flexibility_model
    assert (
        decoded_molsetup.ring_closure_info == starting_molsetup.ring_closure_info
    )  # EMPTY
    assert decoded_molsetup.restraints == starting_molsetup.restraints  # EMPTY
    assert decoded_molsetup.is_sidechain == starting_molsetup.is_sidechain
    assert (
        decoded_molsetup.rmsd_symmetry_indices
        == starting_molsetup.rmsd_symmetry_indices
    )
    # Assert that the starting object's bond attribute was not compromised in the serialization process:
    assert isinstance(list(starting_molsetup.bond.keys())[0], tuple)
    # Assert that the final object's bond attribute had the expected key type:
    assert isinstance(list(decoded_molsetup.bond.keys())[0], tuple)
    assert decoded_molsetup.rings == starting_molsetup.rings
    assert decoded_molsetup.rings_aromatic == starting_molsetup.rings_aromatic
    assert decoded_molsetup.atom_to_ring_id == starting_molsetup.atom_to_ring_id
    assert decoded_molsetup.ring_corners == starting_molsetup.ring_corners  # EMPTY
    assert decoded_molsetup.name == starting_molsetup.name  # EMPTY
    assert decoded_molsetup.rotamers == starting_molsetup.rotamers  # EMPTY
    return


def check_residue_equality(
    decoded_residue: ChorizoResidue, starting_residue: ChorizoResidue
):
    # Goes through the Chorizo Residue's fields and checks that they are the expected type and match the ChorizoResidue
    # object before serialization (that we have effectively rebuilt the ChorizoResidue)

    # RDKit Mols - Check whether we can test for equality with RDKit Mols
    # assert decoded_residue.raw_rdkit_mol == starting_residue.raw_rdkit_mol
    assert isinstance(decoded_residue.raw_rdkit_mol, Chem.rdchem.Mol)
    # assert decoded_residue.rdkit_mol == starting_residue.rdkit_mol
    assert isinstance(decoded_residue.rdkit_mol, Chem.rdchem.Mol)
    # assert decoded_residue.padded_mol == starting_residue.padded_mol
    assert isinstance(decoded_residue.padded_mol, Chem.rdchem.Mol)

    # MapIDX
    assert decoded_residue.mapidx_to_raw == starting_residue.mapidx_to_raw
    assert decoded_residue.mapidx_from_raw == starting_residue.mapidx_from_raw

    # Non-Bool vars
    assert decoded_residue.residue_template_key == starting_residue.residue_template_key
    assert decoded_residue.input_resname == starting_residue.input_resname
    assert decoded_residue.atom_names == starting_residue.atom_names
    check_molsetup_equality(decoded_residue.molsetup, starting_residue.molsetup)
    assert isinstance(decoded_residue.molsetup, RDKitMoleculeSetup)

    # Bools
    assert decoded_residue.is_flexres_atom == starting_residue.is_flexres_atom
    assert decoded_residue.is_movable == starting_residue.is_movable
    assert decoded_residue.user_deleted == starting_residue.user_deleted
