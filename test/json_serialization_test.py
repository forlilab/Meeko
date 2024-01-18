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
    MoleculePreparation,
    MoleculeSetup,
    MoleculeSetupEncoder,
    RDKitMoleculeSetup,
    ResidueAdditionalConnection,
)

#from ..meeko.utils.pdbutils import PDBAtomInfo

pkgdir = pathlib.Path(meeko.__file__).parents[1]

# Test Data
ahhy_example = pkgdir / "example/chorizo/AHHY.pdb"


# Fixtures
#@pytest.fixture
def populated_rdkit_molsetup():
    file = open(ahhy_example)
    pdb_str = file.read()
    chorizo = LinkedRDKitChorizo(pdb_str)
    molecule_prep = MoleculePreparation()
    # To add RDKit Mol to molecule setup
    chorizo.flexibilize_protein_sidechain("A:TYR:4", molecule_prep)
    residue = chorizo.residues["A:TYR:4"]
    return residue.molsetup


# Test Cases
def test_rdkit_molsetup_encoding_decoding():
    # TODO: Certain fields are empty in this example, and if we want to make sure that json is working in all scenarios
    # we will need to make other tests for those empty fields.
    # Encode and decode molsetup from json
    starting_molsetup = populated_rdkit_molsetup()
    json_str = json.dumps(starting_molsetup, cls=MoleculeSetupEncoder)
    decoded_molsetup = json.loads(json_str, object_hook=MoleculeSetup.molsetup_json_decoder)

    # First asserts that all types are as expected
    assert isinstance(starting_molsetup, RDKitMoleculeSetup)
    assert isinstance(decoded_molsetup, RDKitMoleculeSetup)

    # First checks the attributes that we are not doing any type conversion on

    # Next checks attributes that needed minimal type conversion

    # Finally goes through attributes that needed complex interventions

    # Bool used while looping through values to check whether all values in a data structure have the expected type
    correct_val_type = True
    # Go through molsetup attributes and check that they are the expected type and match the molsetup object
    # before serialization.
    assert decoded_molsetup.atom_pseudo == starting_molsetup.atom_pseudo  # EMPTY
    assert isinstance(decoded_molsetup.coord, collections.OrderedDict)
    assert decoded_molsetup.coord.keys() == starting_molsetup.coord.keys()
    for key in decoded_molsetup.coord:
        correct_val_type = correct_val_type & isinstance(decoded_molsetup.coord[key], numpy.ndarray)
        assert set(decoded_molsetup.coord[key]) == set(starting_molsetup.coord[key])
    assert correct_val_type
    assert decoded_molsetup.charge == starting_molsetup.charge
    assert decoded_molsetup.pdbinfo == starting_molsetup.pdbinfo
    correct_val_type = True
    # for key in decoded_molsetup.pdbinfo:
    #    correct_val_type = correct_val_type & isinstance(decoded_molsetup.pdbinfo[key], PDBAtomInfo)
    assert correct_val_type
    assert decoded_molsetup.atom_type == starting_molsetup.atom_type
    assert decoded_molsetup.atom_params == starting_molsetup.atom_params  # WILL THERE BE CONVERSIONS NEEDED FOR PARAMS?
    assert decoded_molsetup.dihedral_interactions == starting_molsetup.dihedral_interactions  # EMPTY
    assert decoded_molsetup.dihedral_partaking_atoms == starting_molsetup.dihedral_partaking_atoms  # EMPTY
    assert decoded_molsetup.dihedral_labels == starting_molsetup.dihedral_labels  # EMPTY
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
    assert decoded_molsetup.interaction_vector == starting_molsetup.interaction_vector  # EMPTY
    # Assert that both the starting and final objects have keys that are the expected type
    if 'rigid_body_connectivity' in starting_molsetup.flexibility_model:
        assert isinstance(list(starting_molsetup.flexibility_model['rigid_body_connectivity'].keys())[0], tuple)
        assert isinstance(list(decoded_molsetup.flexibility_model['rigid_body_connectivity'].keys())[0], tuple)
    assert decoded_molsetup.flexibility_model == starting_molsetup.flexibility_model
    assert decoded_molsetup.ring_closure_info == starting_molsetup.ring_closure_info # EMPTY
    assert decoded_molsetup.restraints == starting_molsetup.restraints # EMPTY
    assert decoded_molsetup.is_sidechain == starting_molsetup.is_sidechain
    assert decoded_molsetup.rmsd_symmetry_indices == starting_molsetup.rmsd_symmetry_indices
    # Assert that the starting object's bond attribute was not compromised in the serialization process:
    assert isinstance(list(starting_molsetup.bond.keys())[0], tuple)
    # Assert that the final object's bond attribute had the expected key type:
    assert isinstance(list(decoded_molsetup.bond.keys())[0], tuple)
    assert decoded_molsetup.rings == starting_molsetup.rings
    assert decoded_molsetup.rings_aromatic == starting_molsetup.rings_aromatic
    assert decoded_molsetup.atom_to_ring_id == starting_molsetup.atom_to_ring_id
    assert decoded_molsetup.ring_corners == starting_molsetup.ring_corners # EMPTY
    assert decoded_molsetup.name == starting_molsetup.name # EMPTY
    assert decoded_molsetup.rotamers == starting_molsetup.rotamers # empty
