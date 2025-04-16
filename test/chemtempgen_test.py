import pathlib
import json
import meeko
from meeko import ResidueChemTemplates
from meeko.chemtempgen import export_chem_templates_to_json, build_noncovalent_CC, ChemicalComponent

pkgdir = pathlib.Path(meeko.__file__).parents[1]
default_template_file = pkgdir / "meeko/data/residue_chem_templates.json"

def template_equality_check(basename: str, suffix: str, cc_instance: ChemicalComponent) -> bool:
    """
    Check if the JSON representation of a residue template matches the default data.

    Parameters
    ----------
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
    with open(default_template_file, "r") as f:
        default_templates = json.load(f)
    expected = default_templates["residue_templates"][basename + suffix]

    # Get the comparable JSON representation of the made residue template in test
    result_json = export_chem_templates_to_json([cc_instance])
    parsed_result = json.loads(result_json)

    return parsed_result["residue_templates"][basename] == expected

def test_build_noncovalent_CC():
    basename = "WMG"  # free ligand from CCD
    cc = build_noncovalent_CC(basename)

    assert cc is not None
    assert isinstance(cc, ChemicalComponent)

    assert template_equality_check(basename, "_fl-ccd", cc)
