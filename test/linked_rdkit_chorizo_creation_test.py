from rdkit import Chem
from meeko import LinkedRDKitChorizo
from meeko import PDBQTWriterLegacy
from meeko import MoleculePreparation
from meeko import ResidueChemTemplates

import json
import pathlib
import pytest
import os

import meeko

pkgdir = pathlib.Path(meeko.__file__).parents[1]
meekodir = pathlib.Path(meeko.__file__).parents[0]

ahhy_example = pkgdir / "test/linked_rdkit_chorizo_data/AHHY.pdb"
just_one_ALA_missing = (
    pkgdir / "test/linked_rdkit_chorizo_data/just-one-ALA-missing-CB.pdb"
)
just_one_ALA = pkgdir / "test/linked_rdkit_chorizo_data/just-one-ALA.pdb"
just_three_residues = pkgdir / "test/linked_rdkit_chorizo_data/just-three-residues.pdb"
disulfide_bridge = pkgdir / "test/linked_rdkit_chorizo_data/just_a_disulfide_bridge.pdb"
loop_with_disulfide = pkgdir / "test/linked_rdkit_chorizo_data/loop_with_disulfide.pdb"
insertion_code = pkgdir / "test/linked_rdkit_chorizo_data/1igy_B_82-83_has-icode.pdb"


# TODO: add checks for untested chorizo fields (e.g. input options not indicated here)

with open(meekodir / "data" / "residue_chem_templates.json") as f:
    t = json.load(f)
chem_templates = ResidueChemTemplates.from_dict(t)
mk_prep = MoleculePreparation(
    merge_these_atom_types=["H"],
    charge_model="gasteiger",
    load_atom_params="ad4_types",
)


def check_charge(residue, expected_charge, tolerance=0.002):
    charge = 0
    for atom in residue.molsetup.atoms:
        if not atom.is_ignore:
            charge += atom.charge
    assert abs(charge - expected_charge) < tolerance


def run_padding_checks(residue):
    assert len(residue.molsetup_mapidx) == residue.rdkit_mol.GetNumAtoms()
    # check index mapping between padded and rea molecule
    for i, j in residue.molsetup_mapidx.items():
        padding_z = residue.padded_mol.GetAtomWithIdx(i).GetAtomicNum()
        real_z = residue.rdkit_mol.GetAtomWithIdx(j).GetAtomicNum()
        assert padding_z == real_z
    # check padding atoms are ignored
    for i in range(residue.padded_mol.GetNumAtoms()):
        if i not in residue.molsetup_mapidx:  # is padding atom
            assert residue.molsetup.atoms[i].is_ignore


def test_flexres_pdbqt():
    with open(loop_with_disulfide) as f:
        pdb_string = f.read()
    set_templates = {
        ":6": "CYX",
        ":17": "CYX",
    }  # TODO remove this to test use of bonds to set templates
    chorizo = LinkedRDKitChorizo.from_pdb_string(
        pdb_string,
        chem_templates,
        mk_prep,
        set_templates,
        blunt_ends=[(":5", 0), (":18", 2)],
    )
    chorizo.flexibilize_sidechain(":11", mk_prep)
    rigid, flex_dict = PDBQTWriterLegacy.write_from_linked_rdkit_chorizo(chorizo)
    nr_rigid_atoms = len(rigid.splitlines())
    assert nr_rigid_atoms == 124
    nr_flex_atoms = 0
    for line in flex_dict[":11"].splitlines():
        nr_flex_atoms += int(line.startswith("ATOM"))
    assert nr_flex_atoms == 9


def test_AHHY_all_static_residues():
    f = open(ahhy_example, "r")
    pdb_string = f.read()
    chorizo = LinkedRDKitChorizo.from_pdb_string(
        pdb_string,
        chem_templates,
        mk_prep,
        blunt_ends=[("A:1", 0)],
    )
    # Asserts that the residues have been imported in a way that makes sense, and that all the
    # private functions we expect to have run as expected.
    assert len(chorizo.residues) == 4
    assert len(chorizo.get_ignored_residues()) == 0
    assert len(chorizo.get_valid_residues()) == 4
    assert len(chorizo.get_user_deleted_residues()) == 0
    assert chorizo.residues["A:1"].residue_template_key == "ALA"
    assert chorizo.residues["A:2"].residue_template_key == "HID"
    assert chorizo.residues["A:3"].residue_template_key == "HIE"
    assert chorizo.residues["A:4"].residue_template_key == "CTYR"

    check_charge(chorizo.residues["A:1"], 0.0)
    check_charge(chorizo.residues["A:2"], 0.0)
    check_charge(chorizo.residues["A:3"], 0.0)
    check_charge(chorizo.residues["A:4"], -1.0)

    pdbqt_strings = PDBQTWriterLegacy.write_string_from_linked_rdkit_chorizo(chorizo)
    rigid_part, movable_part = pdbqt_strings

    # remove newline chars because Windows/Unix differ
    rigid_part = "".join(rigid_part.splitlines())

    assert len(rigid_part) == 3555
    assert len(movable_part) == 0


def test_AHHY_padding():
    with open(ahhy_example, "r") as f:
        pdb_string = f.read()
    chorizo = LinkedRDKitChorizo.from_pdb_string(
        pdb_string,
        chem_templates,
        mk_prep,
        blunt_ends=[("A:1", 0)],
    )
    assert len(chorizo.residues) == 4
    assert len(chorizo.get_ignored_residues()) == 0

    for residue_id in ["A:1", "A:2", "A:3", "A:4"]:
        residue = chorizo.residues[residue_id]
        run_padding_checks(residue)


def test_just_three_padded_mol():
    with open(just_three_residues, "r") as f:
        pdb_string = f.read()
    set_template = {":15": "NMET"}
    chorizo = LinkedRDKitChorizo.from_pdb_string(
        pdb_string,
        chem_templates,
        mk_prep,
        set_template=set_template,
        blunt_ends=[(":17", 17)],
    )
    assert len(chorizo.residues) == 3
    assert len(chorizo.get_ignored_residues()) == 0
    assert len(chorizo.get_valid_residues()) == 3
    assert len(chorizo.get_user_deleted_residues()) == 0

    assert chorizo.residues[":15"].residue_template_key == "NMET"
    assert chorizo.residues[":16"].residue_template_key == "SER"
    assert chorizo.residues[":17"].residue_template_key == "LEU"
    check_charge(chorizo.residues[":15"], 1.0)
    check_charge(chorizo.residues[":16"], 0.0)
    check_charge(chorizo.residues[":17"], 0.0)

    for residue_id in [":15", ":16", ":17"]:
        residue = chorizo.residues[residue_id]
        run_padding_checks(residue)

    pdbqt_strings = PDBQTWriterLegacy.write_string_from_linked_rdkit_chorizo(chorizo)
    rigid_part, movable_part = pdbqt_strings
    # remove newline chars because Windows/Unix differ
    rigid_part = "".join(rigid_part.splitlines())
    assert len(rigid_part) == 2212
    assert len(movable_part) == 0


def test_AHHY_mutate_residues():
    # We want both histidines to be "HIP" and to delete the tyrosine
    set_template = {
        "A:2": "HIP",
        "A:3": "HIP",
    }
    delete_residues = ("A:4",)
    with open(ahhy_example, "r") as f:
        pdb_string = f.read()
    chorizo = LinkedRDKitChorizo.from_pdb_string(
        pdb_string,
        chem_templates,
        mk_prep,
        residues_to_delete=delete_residues,
        set_template=set_template,
        blunt_ends=[("A:1", 0)],
    )
    assert len(chorizo.residues) == 4
    assert len(chorizo.get_ignored_residues()) == 0
    assert len(chorizo.get_valid_residues()) == 3
    assert len(chorizo.get_user_deleted_residues()) == 1
    assert chorizo.residues["A:4"].user_deleted

    assert chorizo.residues["A:1"].residue_template_key == "ALA"
    assert chorizo.residues["A:2"].residue_template_key == "HIP"
    assert chorizo.residues["A:3"].residue_template_key == "HIP"

    check_charge(chorizo.residues["A:1"], 0.0)
    check_charge(chorizo.residues["A:2"], 1.0)
    check_charge(chorizo.residues["A:3"], 1.0)

    pdbqt_strings = PDBQTWriterLegacy.write_string_from_linked_rdkit_chorizo(chorizo)
    rigid_part, movable_part = pdbqt_strings
    # remove newline chars because Windows/Unix differ
    rigid_part = "".join(rigid_part.splitlines())
    assert len(rigid_part) == 2528
    assert len(movable_part) == 0


def test_residue_missing_atoms():
    with open(just_one_ALA_missing, "r") as f:
        pdb_string = f.read()

    chorizo = LinkedRDKitChorizo.from_pdb_string(
        pdb_string,
        chem_templates,
        mk_prep,
        allow_bad_res=True,
        blunt_ends=[("A:1", 0), ("A:1", 2)],
    )
    assert len(chorizo.get_valid_residues()) == 0
    assert len(chorizo.get_user_deleted_residues()) == 0
    assert len(chorizo.residues) == 1
    assert len(chorizo.get_ignored_residues()) == 1

    with pytest.raises(RuntimeError):
        chorizo = LinkedRDKitChorizo.from_pdb_string(
            pdb_string,
            chem_templates,
            mk_prep,
            allow_bad_res=False,
            blunt_ends=[("A:1", 0), ("A:1", 2)],
        )
    return


def test_AHHY_mk_prep_and_export():
    with open(ahhy_example, "r") as f:
        pdb_text = f.read()
    mk_prep2 = MoleculePreparation(
        add_atom_types=[{"smarts": "[CH2,CH3]", "new_param": 42.0}]
    )
    chorizo = LinkedRDKitChorizo.from_pdb_string(
        pdb_text,
        chem_templates,
        mk_prep2,
        blunt_ends=[("A:1", 0)],
    )
    ap, xyz = chorizo.export_static_atom_params()
    # all parameters musthave same size
    assert len(set([len(values) for (key, values) in ap.items()])) == 1
    assert "new_param" in ap


def test_disulfides():
    with open(disulfide_bridge, "r") as f:
        pdb_text = f.read()
    # auto disulfide detection is enabled by default
    chorizo_disulfide = LinkedRDKitChorizo.from_pdb_string(
        pdb_text,
        chem_templates,
        mk_prep,
        blunt_ends=[("B:22", 0), ("B:22", 2), ("B:95", 0), ("B:95", 2)],
    )
    # the disulfide bond is detected, and it expects two paddings,
    # but forcing CYS not CYX disables the padding, so error expected
    with pytest.raises(RuntimeError):
        chorizo_thiols = LinkedRDKitChorizo.from_pdb_string(
            pdb_text,
            chem_templates,
            mk_prep,
            set_template={"B:22": "CYS"},
            blunt_ends=[("B:22", 0), ("B:22", 2), ("B:95", 0), ("B:95", 2)],
        )

    # remove bond and expect CYS between residues (currently, max one bond between each pair of residues)
    chorizo_thiols = LinkedRDKitChorizo.from_pdb_string(
        pdb_text,
        chem_templates,
        mk_prep,
        bonds_to_delete=[("B:22", "B:95")],
        blunt_ends=[("B:22", 0), ("B:22", 2), ("B:95", 0), ("B:95", 2)],
    )

    # check residue names
    assert chorizo_disulfide.residues["B:22"].residue_template_key == "CYX"
    assert chorizo_disulfide.residues["B:95"].residue_template_key == "CYX"
    assert chorizo_thiols.residues["B:22"].residue_template_key == "CYS"
    assert chorizo_thiols.residues["B:95"].residue_template_key == "CYS"


def test_insertion_code():
    with open(insertion_code, "r") as f:
        pdb_text = f.read()
    chorizo = LinkedRDKitChorizo.from_pdb_string(
        pdb_text,
        chem_templates,
        mk_prep,
        blunt_ends=[("B:82", 0), ("B:83", 2)],
    )

    expected_res = set(("B:82", "B:82A", "B:82B", "B:82C", "B:83"))
    res = set(chorizo.residues)
    assert res == expected_res


def test_write_pdb_1igy():
    with open(insertion_code, "r") as f:
        pdb_text = f.read()
    chorizo = LinkedRDKitChorizo.from_pdb_string(
        pdb_text,
        chem_templates,
        mk_prep,
        blunt_ends=[("B:82", 0), ("B:83", 2)],
    )
    pdbstr = chorizo.to_pdb()

    # input 1igy has some hydrogens, here we are making sure
    # that the position of one of them didn't change
    expected = "  -7.232 -23.058 -15.763"
    found = False
    for line in pdbstr.splitlines():
        if line[30:54] == expected:
            found = True
            break
    assert found


def test_write_pdb_AHHY():
    with open(ahhy_example, "r") as f:
        pdb_text = f.read()
    chorizo = LinkedRDKitChorizo.from_pdb_string(
        pdb_text,
        chem_templates,
        mk_prep,
        blunt_ends=[("A:1", 0)],
    )
    newpdbstr = chorizo.to_pdb()
    # AHHy doesn't have hydrogens. If hydrogens get mangled xyz=(0, 0, 0) when
    # added by RDKit, we will probably not be able to match templates anymore.
    # and recreating the chorizo from newpdbstr will very likely fail
    chorizo = LinkedRDKitChorizo.from_pdb_string(
        newpdbstr,
        chem_templates,
        mk_prep,
        blunt_ends=[("A:1", 0)],
    )
