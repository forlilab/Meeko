import pathlib
import json
import meeko
from meeko.chemtempgen import ChemicalComponent
from meeko.chemtempgen import export_chem_templates_to_json
from meeko.chemtempgen import build_noncovalent_CC
from meeko.chemtempgen import build_linked_CCs

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

