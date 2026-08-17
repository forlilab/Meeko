import pathlib
import json
import pytest
import meeko
from meeko.chemtempgen import ChemicalComponent
from meeko.chemtempgen import export_chem_templates_to_json
from meeko.chemtempgen import build_noncovalent_CC
from meeko.chemtempgen import build_linked_CCs
from meeko.chemtempgen import formal_charge_from_cif_value

pkgdir = pathlib.Path(meeko.__file__).parents[1]
default_template_file = pkgdir / "meeko/data/residue_chem_templates.json"
nakb_template_file = pkgdir / "meeko/data/NAKB_templates.json"

def template_equality_check(ref_template_file: str, basename: str, 
                            suffix: str, cc_instance: ChemicalComponent) -> bool:
    """
    Check if the JSON representation of a residue template matches the default data.

    Parameters
    ----------
    ref_template_file : str
        The reference template file to compare against
    basename : str
        The residue name to check
    suffix : str
        The suffix to append to the residue name
    cc_instance : ChemicalComponent
        The generated chemical component instance

    Returns
    -------
    bool
        True if the exported JSON matches the default template
    """

    # Find the expected template from default data file for the given residue name + suffix
    with open(ref_template_file, "r") as f:
        ref_templates = json.load(f)
    expected = ref_templates["residue_templates"][basename + suffix]

    # Get the comparable JSON representation of the made residue template in test
    result_json = export_chem_templates_to_json([cc_instance])
    parsed_result = json.loads(result_json)

    print(f"Expected: {expected}")
    print(f"Parsed: {parsed_result['residue_templates'][basename]}")

    return parsed_result["residue_templates"][basename] == expected

def test_build_noncovalent_CC():
    basename = "WMG"  # free ligand from CCD
    cc = build_noncovalent_CC(basename)

    assert cc is not None
    assert isinstance(cc, ChemicalComponent)

    assert template_equality_check(default_template_file, basename, "_fl-ccd", cc)

def test_add_variants():
    basename = "AMP" 
    cc_list = build_linked_CCs(basename)

    for cc in cc_list:
        assert cc is not None
        assert isinstance(cc, ChemicalComponent)

        assert template_equality_check(default_template_file, cc.resname, "-ccd", cc)


# --- Regression tests for issue #491 -----------------------------------------
# mmCIF uses '?' (value unknown) and '.' (value not applicable) as null markers.
# CCD templates fetched for unknown residues (e.g. HEM) can carry one of these
# in the _chem_comp_atom.charge column, which used to crash from_cif() with
# "ValueError: invalid literal for int() with base 10: '?'" and abort the whole
# receptor preparation. These tests are network-free and pin the fix in place.

def _write_single_atom_cif(directory: pathlib.Path, charge_token: str) -> str:
    """Write a minimal CCD-style CIF for a single-Fe residue with the given
    charge token (faithful to HEM's heme iron) into ``directory`` and return
    its path. ``directory`` is a pytest tmp_path, so the file is auto-cleaned."""
    cif = (
        "data_FEQ\n"
        "_chem_comp.id FEQ\n"
        "loop_\n"
        "_chem_comp_atom.comp_id\n"
        "_chem_comp_atom.atom_id\n"
        "_chem_comp_atom.type_symbol\n"
        "_chem_comp_atom.charge\n"
        "_chem_comp_atom.pdbx_leaving_atom_flag\n"
        f"FEQ FE FE {charge_token} N\n"
    )
    path = pathlib.Path(directory) / "single_atom.cif"
    path.write_text(cif)
    return str(path)


def test_formal_charge_from_cif_value():
    # null markers and blanks are unspecified -> neutral (0)
    assert formal_charge_from_cif_value("?") == 0
    assert formal_charge_from_cif_value(".") == 0
    assert formal_charge_from_cif_value("") == 0
    assert formal_charge_from_cif_value("  ? ") == 0
    assert formal_charge_from_cif_value(None) == 0
    # valid integer charges are preserved
    assert formal_charge_from_cif_value("0") == 0
    assert formal_charge_from_cif_value("1") == 1
    assert formal_charge_from_cif_value("-1") == -1
    assert formal_charge_from_cif_value("2") == 2
    # a genuinely malformed (non-null-marker) charge still raises, preserving
    # the original fail-loud behavior for corrupt CIFs
    with pytest.raises(ValueError):
        formal_charge_from_cif_value("n/a")


def test_from_cif_unknown_charge_marker_does_not_crash(tmp_path):
    # Regression: a '?' in the charge column must not raise (issue #491).
    cif_path = _write_single_atom_cif(tmp_path, "?")
    cc = ChemicalComponent.from_cif(cif_path, "FEQ")
    assert cc is not None
    assert isinstance(cc, ChemicalComponent)
    # unspecified charge -> neutral formal charge
    assert cc.rdkit_mol.GetAtomWithIdx(0).GetFormalCharge() == 0


def test_from_cif_valid_integer_charge_preserved(tmp_path):
    # The fix must not change behavior for a real integer charge.
    cif_path = _write_single_atom_cif(tmp_path, "2")
    cc = ChemicalComponent.from_cif(cif_path, "FEQ")
    assert cc is not None
    assert cc.rdkit_mol.GetAtomWithIdx(0).GetFormalCharge() == 2

