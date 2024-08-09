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
    PDBQTWriterLegacy,
)

from meeko import linked_rdkit_chorizo
from meeko.molsetup import Atom, Bond, Ring, RingClosureInfo, Restraint

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from meeko.utils.pdbutils import PDBAtomInfo

# from ..meeko.utils.pdbutils import PDBAtomInfo

pkgdir = pathlib.Path(meeko.__file__).parents[1]
meekodir = pathlib.Path(meeko.__file__).parents[0]

# Test Data
ahhy_example = pkgdir / "test/linked_rdkit_chorizo_data/AHHY.pdb"
just_one_ALA_missing = (
    pkgdir / "test/linked_rdkit_chorizo_data/just-one-ALA-missing-CB.pdb"
)

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
    chorizo = LinkedRDKitChorizo.from_pdb_string(
        pdb_str, chem_templates, mk_prep, blunt_ends=[("A:1", 0)]
    )
    return chorizo


@pytest.fixture
def populated_rdkit_chorizo_missing():
    file = open(just_one_ALA_missing)
    pdb_str = file.read()
    chorizo = LinkedRDKitChorizo.from_pdb_string(
        pdb_str,
        chem_templates,
        mk_prep,
        blunt_ends=[("A:1", 0), ("A:1", 2)],
        allow_bad_res=True,
    )
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
    json_str = json.dumps(starting_molsetup, cls=MoleculeSetupEncoder)
    decoded_molsetup = json.loads(json_str, object_hook=RDKitMoleculeSetup.from_json)

    # First asserts that all types are as expected
    assert isinstance(starting_molsetup, RDKitMoleculeSetup)
    assert isinstance(decoded_molsetup, RDKitMoleculeSetup)

    # Go through MoleculeSetup attributes and check that they are the expected type and match the MoleculeSetup object
    # before serialization.
    check_molsetup_equality(decoded_molsetup, starting_molsetup)
    return


def test_chorizo_residue_encoding_decoding(populated_rdkit_chorizo_residue):
    """
    Takes a fully populated ChorizoResidue, checks that it can be serialized to JSON and deserialized back into an
    object without any errors, then checks that the deserialized object matches the starting object and that the
    attribute types, values, and structure of the deserialized object are as expected for an ChorizoResidue.

    Parameters
    ----------
    populated_rdkit_chorizo_residue: ChorizoResidue
        Takes as input a populated ChorizoResidue object.

    Returns
    -------
    None
    """
    # Starts by getting a ChorizoResidue object, converting it to a json string, and then decoding the string into
    # a new ChorizoResidue object
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


def test_pdbqt_writing_from_decoded_chorizo(populated_rdkit_chorizo):
    """
    Takes a fully populated ChorizoResidue, writes a PDBQT string from it, encodes and decodes it, writes
    another PDBQT string from the decoded chorizo, and then checks that the PDBQT strings are identical.

    Parameters
    ----------
    populated_rdkit_chorizo: LinkedRDKitChorizo
        Takes as input a populated LinkedRDKitChorizo object.

    Returns
    -------
    None
    """

    starting_chorizo = populated_rdkit_chorizo
    starting_pdbqt = PDBQTWriterLegacy.write_from_linked_rdkit_chorizo(starting_chorizo)
    json_str = json.dumps(starting_chorizo, cls=LinkedRDKitChorizoEncoder)
    decoded_chorizo = json.loads(
        json_str, object_hook=linked_rdkit_chorizo.linked_rdkit_chorizo_json_decoder
    )
    decoded_pdbqt = PDBQTWriterLegacy.write_from_linked_rdkit_chorizo(decoded_chorizo) 
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


def test_linked_rdkit_chorizo_encoding_decoding(
    populated_rdkit_chorizo, populated_rdkit_chorizo_missing
):
    """
    Takes a fully populated LinkedRDKitChorizo, checks that it can be serialized to JSON and deserialized back into an
    object without any errors, then checks that the deserialized object matches the starting object and that the
    attribute types, values, and structure of the deserialized object are as expected for an LinkedRDKitChorizo.

    Parameters
    ----------
    populated_rdkit_chorizo: LinkedRDKitChorizo
        Takes as input a populated LinkedRDKitChorizo object.

    Returns
    -------
    None
    """
    # Starts by getting a LinkedRDKitChorizo object, converting it to a json string, and then decoding the string into
    # a new LinkedRDKitChorizo object
    chorizos = (
        populated_rdkit_chorizo,
        populated_rdkit_chorizo_missing,
    )
    for starting_chorizo in chorizos:
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
    assert isinstance(decoded_obj.is_sidechain, bool)
    assert decoded_obj.is_sidechain == starting_obj.is_sidechain
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
    correct_val_type = True
    # np.array conversion checks
    assert isinstance(decoded_obj.coord, numpy.ndarray)
    for i_vec in decoded_obj.interaction_vectors:
        correct_val_type = correct_val_type and isinstance(i_vec, numpy.ndarray)
    assert correct_val_type

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
    assert isinstance(decoded_obj.corner_flip, bool)
    assert decoded_obj.corner_flip == starting_obj.corner_flip
    assert len(decoded_obj.graph) == len(starting_obj.graph)
    for idx, val in enumerate(starting_obj.graph):
        assert decoded_obj.graph[idx] == val
    assert isinstance(decoded_obj.is_aromatic, bool)
    assert decoded_obj.is_aromatic == starting_obj.is_aromatic
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


def check_residue_equality(decoded_obj: ChorizoResidue, starting_obj: ChorizoResidue):
    """
    Asserts that two ChorizoResidue objects are equal, and that the decoded_obj input has fields contain correctly typed
    data.

    Parameters
    ----------
    decoded_obj: ChorizoResidue
        A ChorizoResidue object that we want to check is correctly typed and contains the correct data.
    starting_obj: ChorizoResidue
        A ChorizoResidue object with the desired values to check the decoded object against.

    Returns
    -------
    None
    """
    # Goes through the Chorizo Residue's fields and checks that they are the expected type and match the ChorizoResidue
    # object before serialization (that we have effectively rebuilt the ChorizoResidue)

    # RDKit Mols - Check whether we can test for equality with RDKit Mols
    # assert decoded_residue.raw_rdkit_mol == starting_residue.raw_rdkit_mol
    assert type(decoded_obj.raw_rdkit_mol) == type(starting_obj.raw_rdkit_mol)
    if isinstance(decoded_obj.raw_rdkit_mol, Chem.rdchem.Mol):
        assert Chem.MolToSmiles(decoded_obj.raw_rdkit_mol) == Chem.MolToSmiles(
            starting_obj.raw_rdkit_mol
        )
    # assert decoded_residue.rdkit_mol == starting_residue.rdkit_mol
    assert type(decoded_obj.rdkit_mol) == type(starting_obj.rdkit_mol)
    if isinstance(decoded_obj.rdkit_mol, Chem.rdchem.Mol):
        assert Chem.MolToSmiles(decoded_obj.rdkit_mol) == Chem.MolToSmiles(
            starting_obj.rdkit_mol
        )
    # assert decoded_residue.padded_mol == starting_residue.padded_mol
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
    assert decoded_obj.user_deleted == starting_obj.user_deleted
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

    assert isinstance(decoded_obj.adjacent_smartsmol, Chem.rdchem.Mol)
    assert (
        decoded_obj.adjacent_smartsmol_mapidx == starting_obj.adjacent_smartsmol_mapidx
    )
    return


def check_linked_rdkit_chorizo_equality(
    decoded_obj: LinkedRDKitChorizo, starting_obj: LinkedRDKitChorizo
):
    """
    Asserts that two LinkedRDKitChorizo objects are equal, and that the decoded_obj input has fields contain correctly
    typed data.

    Parameters
    ----------
    decoded_obj: LinkedRDKitChorizo
        A LinkedRDKitChorizo object that we want to check is correctly typed and contains the correct data.
    starting_obj: LinkedRDKitChorizo
        A LinkedRDKitChorizo object with the desired values to check the decoded object against.

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
