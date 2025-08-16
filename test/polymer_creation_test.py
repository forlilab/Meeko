import json
import pathlib
import pytest

from meeko import Polymer
from meeko import PDBQTWriterLegacy
from meeko import MoleculePreparation
from meeko import ResidueChemTemplates
import meeko

from rdkit import Chem
import numpy as np


pkgdir = pathlib.Path(meeko.__file__).parents[1]

ahhy_example = pkgdir / "test/polymer_data/AHHY.pdb"
pqr_example = pkgdir / "test/polymer_data/1FAS_dry.pqr"
nphe_ser_example = pkgdir / "test/polymer_data/NPHE_SER.pdb"
just_one_ALA_missing = (
    pkgdir / "test/polymer_data/just-one-ALA-missing-CB.pdb"
)
just_one_ALA = pkgdir / "test/polymer_data/just-one-ALA.pdb"
just_three_residues = pkgdir / "test/polymer_data/just-three-residues.pdb"
disulfide_bridge = pkgdir / "test/polymer_data/just_a_disulfide_bridge.pdb"
loop_with_disulfide = pkgdir / "test/polymer_data/loop_with_disulfide.pdb"
insertion_code = pkgdir / "test/polymer_data/1igy_B_82-83_has-icode.pdb"
non_sequential_res = pkgdir / "test/polymer_data/non-sequential-res.pdb"
has_altloc = pkgdir / "test/polymer_data/has-altloc.pdb"
has_lys = pkgdir / "test/polymer_data/has-lys.pdb"
has_lyn = pkgdir / "test/polymer_data/has-lyn.pdb"
has_lys_resname_lyn = pkgdir / "test/polymer_data/has-lys-resname-lyn.pdb"
disulfide_adjacent = pkgdir / "test/polymer_data/disulfide_bridge_in_adjacent_residues.pdb"
nglu = pkgdir / "test/polymer_data/nglu.pdb"


# TODO: add checks for untested polymer fields (e.g. input options not indicated here)

chem_templates = ResidueChemTemplates.create_from_defaults()
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
    polymer = Polymer.from_pdb_string(
        pdb_string,
        chem_templates,
        mk_prep,
        set_templates,
        blunt_ends=[(":5", 0), (":18", 2)],
    )
    res11 = polymer.monomers[":11"]
    assert sum(res11.is_flexres_atom) == 0
    polymer.flexibilize_sidechain(":11", mk_prep)
    assert sum(res11.is_flexres_atom) == 9
    rigid, flex_dict = PDBQTWriterLegacy.write_from_polymer(polymer)
    nr_rigid_atoms = len(rigid.splitlines())
    assert nr_rigid_atoms == 124
    nr_flex_atoms = 0
    for line in flex_dict[":11"].splitlines():
        nr_flex_atoms += int(line.startswith("ATOM"))
    assert nr_flex_atoms == 9


def test_AHHY_all_static_residues():
    f = open(ahhy_example, "r")
    pdb_string = f.read()
    polymer = Polymer.from_pdb_string(
        pdb_string,
        chem_templates,
        mk_prep,
        blunt_ends=[("A:1", 0)],
    )
    # Asserts that the residues have been imported in a way that makes sense, and that all the
    # private functions we expect to have run as expected.
    assert len(polymer.monomers) == 4
    assert len(polymer.get_ignored_monomers()) == 0
    assert len(polymer.get_valid_monomers()) == 4
    assert polymer.monomers["A:1"].residue_template_key == "ALA"
    assert polymer.monomers["A:2"].residue_template_key == "HID"
    assert polymer.monomers["A:3"].residue_template_key == "HIE"
    assert polymer.monomers["A:4"].residue_template_key == "CTYR"

    check_charge(polymer.monomers["A:1"], 0.0)
    check_charge(polymer.monomers["A:2"], 0.0)
    check_charge(polymer.monomers["A:3"], 0.0)
    check_charge(polymer.monomers["A:4"], -1.0)

    pdbqt_strings = PDBQTWriterLegacy.write_string_from_polymer(polymer)
    rigid_part, movable_part = pdbqt_strings

    # remove newline chars because Windows/Unix differ
    rigid_part = "".join(rigid_part.splitlines())

    assert len(rigid_part) == 3555
    assert len(movable_part) == 0

def test_AHHY_flex_residues():
    f = open(ahhy_example, "r")
    pdb_string = f.read()
    polymer = Polymer.from_pdb_string(
        pdb_string,
        chem_templates,
        mk_prep,
    )
    polymer.flexibilize_sidechain("A:2", mk_prep)
    # Asserts that the residues have been imported in a way that makes sense, and that all the
    # private functions we expect to have run as expected.
    assert len(polymer.monomers) == 4
    assert len(polymer.get_ignored_monomers()) == 0
    assert len(polymer.get_valid_monomers()) == 4
    assert polymer.monomers["A:1"].residue_template_key == "ALA"
    assert polymer.monomers["A:2"].residue_template_key == "HID"
    assert polymer.monomers["A:3"].residue_template_key == "HIE"
    assert polymer.monomers["A:4"].residue_template_key == "CTYR"

    check_charge(polymer.monomers["A:1"], 0.0)
    check_charge(polymer.monomers["A:2"], 0.0)
    check_charge(polymer.monomers["A:3"], 0.0)
    check_charge(polymer.monomers["A:4"], -1.0)

    pdbqt_strings = PDBQTWriterLegacy.write_string_from_polymer(polymer)
    rigid_part, movable_part = pdbqt_strings

    # remove newline chars because Windows/Unix differ
    rigid_part = "".join(rigid_part.splitlines())

    assert len(rigid_part) == 2923
    assert len(movable_part) == 809

    # and now with a fully rigid sidechain, to make sure it goes in rigid
    rigid_prep = MoleculePreparation(
        rigidify_bonds_smarts=["[*]~[*]"],
        rigidify_bonds_indices=[(0, 1)],
    )
    polymer.monomers["A:2"].parameterize(rigid_prep, "A:2")
    polymer.flexibilize_sidechain("A:2", rigid_prep)
    pdbqt_strings = PDBQTWriterLegacy.write_string_from_polymer(polymer)
    rigid_part, movable_part = pdbqt_strings

    # remove newline chars because Windows/Unix differ
    rigid_part = "".join(rigid_part.splitlines())

    assert len(rigid_part) == 3555
    assert len(movable_part) == 0


def test_AHHY_flexibilize_then_parameterize():
    f = open(ahhy_example, "r")
    pdb_string = f.read()
    polymer = Polymer.from_pdb_string(
        pdb_string,
        chem_templates,
        mk_prep,
    )
    polymer.flexibilize_sidechain("A:2", mk_prep)
    m = polymer.monomers["A:2"]
    nr_rot_bonds = sum([b.rotatable for _, b in m.molsetup.bond_info.items()])
    assert nr_rot_bonds == 2
    # now parameterize and check we still have 2 rotatable bonds
    # backbone may have become flexible
    m.parameterize(mk_prep, "A:2")
    nr_rot_bonds = sum([b.rotatable for _, b in m.molsetup.bond_info.items()])
    assert nr_rot_bonds == 2



def test_protonated_Nterm_residue():
    f = open(nphe_ser_example, "r")
    pdb_string = f.read()
    polymer = Polymer.from_pdb_string(
        pdb_string,
        chem_templates,
        mk_prep,
        blunt_ends=[("A:2", 0)],
    )
    # Asserts that the residues have been imported in a way that makes sense, and that all the
    # private functions we expect to have run as expected.
    assert len(polymer.monomers) == 2
    assert len(polymer.get_ignored_monomers()) == 0
    assert len(polymer.get_valid_monomers()) == 2
    assert polymer.monomers["A:1"].residue_template_key == "NPHE"
    assert polymer.monomers["A:2"].residue_template_key == "SER"

    check_charge(polymer.monomers["A:1"], 1.0)
    check_charge(polymer.monomers["A:2"], 0.0)


def test_AHHY_padding():
    with open(ahhy_example, "r") as f:
        pdb_string = f.read()
    polymer = Polymer.from_pdb_string(
        pdb_string,
        chem_templates,
        mk_prep,
        blunt_ends=[("A:1", 0)],
    )
    assert len(polymer.monomers) == 4
    assert len(polymer.get_ignored_monomers()) == 0

    for residue_id in ["A:1", "A:2", "A:3", "A:4"]:
        residue = polymer.monomers[residue_id]
        run_padding_checks(residue)


def test_just_three_padded_mol():
    with open(just_three_residues, "r") as f:
        pdb_string = f.read()
    set_template = {":15": "NMET"}
    polymer = Polymer.from_pdb_string(
        pdb_string,
        chem_templates,
        mk_prep,
        set_template=set_template,
        blunt_ends=[(":17", 17)],
    )
    assert len(polymer.monomers) == 3
    assert len(polymer.get_ignored_monomers()) == 0
    assert len(polymer.get_valid_monomers()) == 3

    assert polymer.monomers[":15"].residue_template_key == "NMET"
    assert polymer.monomers[":16"].residue_template_key == "SER"
    assert polymer.monomers[":17"].residue_template_key == "LEU"
    check_charge(polymer.monomers[":15"], 1.0)
    check_charge(polymer.monomers[":16"], 0.0)
    check_charge(polymer.monomers[":17"], 0.0)

    for residue_id in [":15", ":16", ":17"]:
        residue = polymer.monomers[residue_id]
        run_padding_checks(residue)

    pdbqt_strings = PDBQTWriterLegacy.write_string_from_polymer(polymer)
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
    polymer = Polymer.from_pdb_string(
        pdb_string,
        chem_templates,
        mk_prep,
        residues_to_delete=delete_residues,
        set_template=set_template,
        blunt_ends=[("A:1", 0)],
    )
    assert len(polymer.monomers) == 3
    assert len(polymer.get_ignored_monomers()) == 0
    assert len(polymer.get_valid_monomers()) == 3

    assert polymer.monomers["A:1"].residue_template_key == "ALA"
    assert polymer.monomers["A:2"].residue_template_key == "HIP"
    assert polymer.monomers["A:3"].residue_template_key == "HIP"

    check_charge(polymer.monomers["A:1"], 0.0)
    check_charge(polymer.monomers["A:2"], 1.0)
    check_charge(polymer.monomers["A:3"], 1.0)

    pdbqt_strings = PDBQTWriterLegacy.write_string_from_polymer(polymer)
    rigid_part, movable_part = pdbqt_strings
    # remove newline chars because Windows/Unix differ
    rigid_part = "".join(rigid_part.splitlines())
    assert len(rigid_part) == 2528
    assert len(movable_part) == 0


def test_residue_missing_atoms():
    with open(just_one_ALA_missing, "r") as f:
        pdb_string = f.read()

    polymer = Polymer.from_pdb_string(
        pdb_string,
        chem_templates,
        mk_prep,
        allow_bad_res=True,
        blunt_ends=[("A:1", 0), ("A:1", 2)],
    )
    assert len(polymer.get_valid_monomers()) == 0
    assert len(polymer.monomers) == 1
    assert len(polymer.get_ignored_monomers()) == 1

    with pytest.raises(RuntimeError):
        polymer = Polymer.from_pdb_string(
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
    polymer = Polymer.from_pdb_string(
        pdb_text,
        chem_templates,
        mk_prep2,
        blunt_ends=[("A:1", 0)],
    )
    ap, xyz = polymer.export_static_atom_params()
    # all parameters musthave same size
    assert len(set([len(values) for (key, values) in ap.items()])) == 1
    assert "new_param" in ap


def test_disulfides():
    with open(disulfide_bridge, "r") as f:
        pdb_text = f.read()
    # auto disulfide detection is enabled by default
    polymer_disulfide = Polymer.from_pdb_string(
        pdb_text,
        chem_templates,
        mk_prep,
        blunt_ends=[("B:22", 0), ("B:22", 2), ("B:95", 0), ("B:95", 2)],
    )
    # the disulfide bond is detected, and it expects two paddings,
    # but forcing CYS not CYX disables the padding, so error expected
    with pytest.raises(RuntimeError):
        polymer_thiols = Polymer.from_pdb_string(
            pdb_text,
            chem_templates,
            mk_prep,
            set_template={"B:22": "CYS"},
            blunt_ends=[("B:22", 0), ("B:22", 2), ("B:95", 0), ("B:95", 2)],
        )

    # remove bond and expect CYS between residues
    # currently, all bonds between a pair of residues will be removed
    polymer_thiols = Polymer.from_pdb_string(
        pdb_text,
        chem_templates,
        mk_prep,
        bonds_to_delete=[("B:22", "B:95")],
        blunt_ends=[("B:22", 0), ("B:22", 2), ("B:95", 0), ("B:95", 2)],
    )

    # check residue names
    assert polymer_disulfide.monomers["B:22"].residue_template_key == "CYX"
    assert polymer_disulfide.monomers["B:95"].residue_template_key == "CYX"
    assert polymer_thiols.monomers["B:22"].residue_template_key == "CYS"
    assert polymer_thiols.monomers["B:95"].residue_template_key == "CYS"


def test_insertion_code():
    with open(insertion_code, "r") as f:
        pdb_text = f.read()
    polymer = Polymer.from_pdb_string(
        pdb_text,
        chem_templates,
        mk_prep,
        blunt_ends=[("B:82", 0), ("B:83", 2)],
    )

    expected_res = set(("B:82", "B:82A", "B:82B", "B:82C", "B:83"))
    res = set(polymer.monomers)
    assert res == expected_res


def test_write_pdb_1igy():
    with open(insertion_code, "r") as f:
        pdb_text = f.read()
    polymer = Polymer.from_pdb_string(
        pdb_text,
        chem_templates,
        mk_prep,
        blunt_ends=[("B:82", 0), ("B:83", 2)],
    )
    pdbstr = polymer.to_pdb()

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
    polymer = Polymer.from_pdb_string(
        pdb_text,
        chem_templates,
        mk_prep,
        blunt_ends=[("A:1", 0)],
    )
    newpdbstr = polymer.to_pdb()
    # AHHy doesn't have hydrogens. If hydrogens get mangled xyz=(0, 0, 0) when
    # added by RDKit, we will probably not be able to match templates anymore.
    # and recreating the polymer from newpdbstr will very likely fail
    polymer = Polymer.from_pdb_string(
        newpdbstr,
        chem_templates,
        mk_prep,
        blunt_ends=[("A:1", 0)],
    )

def test_write_pdbqt_from_pqr():
    with open(pqr_example, "r") as f:
        pqr_string = f.read()
    mk_prep_for_pqr = MoleculePreparation(
        charge_model="read", 
        charge_atom_prop="PQRCharge"
    )
    polymer = Polymer.from_pqr_string(
        pqr_string, 
        chem_templates,
        mk_prep_for_pqr
    )
    pdbqt_rigid = PDBQTWriterLegacy.write_from_polymer(polymer)[0].split("\n")
    expected_lines = """ATOM      1  C   THR     1      43.983  16.642   1.087  1.00  0.00     0.550 C 
ATOM      2  O   THR     1      44.150  17.855   0.925  1.00  0.00    -0.550 OA
ATOM      3  CA  THR     1      44.862  15.936   2.105  1.00  0.00     0.330 C 
ATOM      4  N   THR     1      46.148  16.581   2.104  1.00  0.00    -0.320 N 
ATOM      5  CB  THR     1      44.293  16.088   3.528  1.00  0.00     0.000 C 
ATOM      6 CG2  THR     1      43.175  15.110   3.826  1.00  0.00     0.000 C 
ATOM      7 OG1  THR     1      45.409  15.915   4.403  1.00  0.00    -0.490 OA
ATOM      8 HG1  THR     1      45.246  15.149   5.034  1.00  0.00     0.490 HD
ATOM      9  H1  THR     1      46.041  17.581   2.102  1.00  0.00     0.330 HD
ATOM     10  H2  THR     1      46.674  16.320   2.920  1.00  0.00     0.330 HD
ATOM     11  H3  THR     1      46.675  16.317   1.289  1.00  0.00     0.330 HD
""".splitlines()
    for i, line in enumerate(expected_lines): 
        if pdbqt_rigid[i][:len(line)] != line: 
            print(pdbqt_rigid[i][:len(line)])
            print(line)
            assert False


def test_non_seq_res():
    """the residue atoms are interrupted (not in contiguous lines)
        which should cause the parser to throw an error. Here we
        check the an error is thrown.
    """
    with open(non_sequential_res, "r") as f:
        pdb_text = f.read()
    with pytest.raises(ValueError) as err_msg:
        polymer = Polymer.from_pdb_string(
            pdb_text,
            chem_templates,
            mk_prep,
        )
    assert str(err_msg.value).startswith("interrupted")

def test_altloc():
    with open(has_altloc, "r") as f:
        pdb_text = f.read()
    with pytest.raises(RuntimeError) as err_msg:
        polymer = Polymer.from_pdb_string(
            pdb_text,
            chem_templates,
            mk_prep,
        )
    assert "altloc" in str(err_msg.value).lower()

    polymer = Polymer.from_pdb_string(
        pdb_text,
        chem_templates,
        mk_prep,
        default_altloc="B",
    )
    res = polymer.monomers["A:264"]
    xyz = res.rdkit_mol.GetConformer().GetPositions()
    for atom in res.rdkit_mol.GetAtoms():
        index = atom.GetIdx()
        name = res.atom_names[index]
        if name == "OG":
            break
    assert abs(xyz[index][0] - 11.220) < 0.001

    polymer = Polymer.from_pdb_string(
        pdb_text,
        chem_templates,
        mk_prep,
        default_altloc="B",
        wanted_altloc={"A:264": "A"}
    )
    res = polymer.monomers["A:264"]
    xyz = res.rdkit_mol.GetConformer().GetPositions()
    for atom in res.rdkit_mol.GetAtoms():
        index = atom.GetIdx()
        name = res.atom_names[index]
        if name == "OG":
            break
    assert abs(xyz[index][0] - 12.346) < 0.001

def test_set_template_LYN():
    """the input is fully protonated NH3+"""
    with open(loop_with_disulfide) as f:
        pdb_string = f.read()
    polymer = Polymer.from_pdb_string(
        pdb_string,
        chem_templates,
        mk_prep,
        set_template={":16": "LYN"},
    )
    res16 = polymer.monomers[":16"]
    res17 = polymer.monomers[":17"]
    assert res17.residue_template_key == "CYX"
    assert res16.residue_template_key == "LYN"
    chrg16 = sum([a.charge for a in res16.molsetup.atoms if not a.is_ignore])
    assert abs(chrg16) < 1e-6

def test_weird_zero_coord():
    with open(has_lys) as f:
        pdbstr = f.read()
    polymer = Polymer.from_pdb_string(pdbstr, chem_templates, mk_prep)
    for _, res in polymer.monomers.items():
        positions = res.rdkit_mol.GetConformer().GetPositions()
        for atom in res.molsetup.atoms:
            # there was a bug in which the C-term CYS of has_lys would be
            # be assigned the erroneous CCYS template,and the extra oxygen
            # would get coordinates set to zero.
            assert np.min(np.sum(positions**2, 1)) > 1e-6

def test_auto_LYN():
    with open(has_lyn) as f:
        pdbstr = f.read()
    polymer = Polymer.from_pdb_string(pdbstr, chem_templates, mk_prep)
    assert polymer.monomers[":15"].residue_template_key == "LEU"
    assert polymer.monomers[":16"].residue_template_key == "LYN"
    assert polymer.monomers[":17"].residue_template_key == "CYX-"
    with open(has_lys) as f:
        pdbstr = f.read()
    polymer = Polymer.from_pdb_string(pdbstr, chem_templates, mk_prep)
    assert polymer.monomers[":16"].residue_template_key == "LYS"
    assert polymer.monomers[":17"].residue_template_key == "CYX-"
    polymer = Polymer.from_pdb_string(pdbstr, chem_templates, mk_prep, set_template={":16": "LYN"})
    assert polymer.monomers[":16"].residue_template_key == "LYN"
    assert polymer.monomers[":17"].residue_template_key == "CYX-"
    with open(has_lys_resname_lyn) as f:
        pdbstr = f.read()
    with pytest.raises(RuntimeError) as err_msg:
        polymer = Polymer.from_pdb_string(pdbstr, chem_templates, mk_prep)

def test_disulfide_adjacent():
    """ disulfide bridge in adjacent residues broke a version of the code
        that assumed only one bond between each pair of residues
    """
    with open(disulfide_adjacent, "r") as f:
        pdb_text = f.read()
    polymer = Polymer.from_pdb_string(
        pdb_text,
        chem_templates,
        mk_prep,
    )

def test_stitch_polymer():
    with open(disulfide_adjacent, "r") as f:
        pdb_text = f.read()
    polymer = Polymer.from_pdb_string(
        pdb_text,
        chem_templates,
        mk_prep,
    )
    adjacent_disulfide = Chem.MolFromSmarts("S1CCNCCCS1")
    disulfide_then_proline = Chem.MolFromSmarts("CSSCCC(=O)N1CCCC1")
    stitched_mol = polymer.stitch()
    assert stitched_mol.HasSubstructMatch(adjacent_disulfide)
    assert stitched_mol.HasSubstructMatch(disulfide_then_proline)
    # after serialization
    polymer = Polymer.from_json(polymer.to_json())
    stitched_mol = polymer.stitch()
    assert stitched_mol.HasSubstructMatch(adjacent_disulfide)
    assert stitched_mol.HasSubstructMatch(disulfide_then_proline)

def test_nglu():
    with open(nglu, "r") as f:
        pdb_text = f.read()
    polymer = Polymer.from_pdb_string(
        pdb_text,
        chem_templates,
        mk_prep,
    )
    assert ":1" in polymer.get_valid_monomers()
