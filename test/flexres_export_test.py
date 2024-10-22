import json
import pathlib
from meeko import utils
from meeko import LinkedRDKitChorizo
from meeko import MoleculePreparation
from meeko import ResidueChemTemplates
from meeko import PDBQTMolecule
from meeko import export_pdb_updated_flexres
import meeko

pkgdir = pathlib.Path(meeko.__file__).parents[1]
just_three_residues = pkgdir / "test/linked_rdkit_chorizo_data/just-three-residues.pdb"
j3r_docked = pkgdir / "test/linked_rdkit_chorizo_data/just-three-residues_vina_flexres.pdbqt"
j3r_idx_docked = pkgdir / "test/linked_rdkit_chorizo_data/just-three-residues_vina_flexres_idxmap.pdbqt"

meekodir = pathlib.Path(meeko.__file__).parents[0]

with open(meekodir / "data" / "residue_chem_templates.json") as f:
    t = json.load(f)
chem_templates = ResidueChemTemplates.from_dict(t)
mk_prep = MoleculePreparation(
    merge_these_atom_types=["H"],
    charge_model="gasteiger",
    load_atom_params="ad4_types",
)

def test_begin_res_parsing():
    assert utils.parse_begin_res("SE8 C 23") == "C:23" 
    assert utils.parse_begin_res("SE8  23") == ":23"
    assert utils.parse_begin_res("SE8  23A") == ":23A"
    assert utils.parse_begin_res("SER A1234A") == "A:1234A"
    assert utils.parse_begin_res("    A 999") == "A:999"
    assert utils.parse_begin_res("  1") == ":1"
    assert utils.parse_begin_res(" A1234A") == "A:1234A"
    assert utils.parse_begin_res(" B1234") == "B:1234"
    assert utils.parse_begin_res("S  23A") == "S:23A"

def get_x_from_pdb_str(pdbstr, wanted_chain, wanted_resnum, wanted_name):
    for line in pdbstr.splitlines():
        if not line.startswith("ATOM") and not line.startswith("HETATM"):
            continue
        resn = int(line[22:26])
        name = line[12:16].strip()
        c = line[21].strip()
        if resn == wanted_resnum and name == wanted_name and c == wanted_chain:
            return float(line[30:38])
    return None

def test_export_sidechains_no_idxmap():
    with open(just_three_residues) as f:
        pdb_string = f.read()
    chorizo = LinkedRDKitChorizo.from_pdb_string(
        pdb_string,
        chem_templates,
        mk_prep,
    )
    # docking had these two flexible
    chorizo.flexibilize_sidechain(":15", mk_prep)
    chorizo.flexibilize_sidechain(":16", mk_prep)
    
    with open(j3r_docked) as f:
        docked_pdbqt_string = f.read()
    pdbqt_mol = PDBQTMolecule(docked_pdbqt_string, skip_typing=True)
    pdbqt_mol._current_pose = 0
    pdb_string = export_pdb_updated_flexres(chorizo, pdbqt_mol)
    x = get_x_from_pdb_str(pdb_string, "", 15, "SD")
    assert abs(x - 10.529) < 0.0001
    pdbqt_mol._current_pose = 1
    pdb_string = export_pdb_updated_flexres(chorizo, pdbqt_mol)
    x = get_x_from_pdb_str(pdb_string, "", 16, "HG")
    assert abs(x - 18.724) < 0.0001

    # rebuilding chorizo can fail if structure is mangled
    chorizo = LinkedRDKitChorizo.from_pdb_string(
        pdb_string,
        chem_templates,
        mk_prep,
    )
    chorizo.flexibilize_sidechain(":15", mk_prep)
    chorizo.flexibilize_sidechain(":16", mk_prep)
    assert len(chorizo.get_valid_residues()) == 3

    # with INDEX MAP in flexres PDBQT
    with open(j3r_idx_docked) as f:
        docked_pdbqt_string = f.read()
    pdbqt_mol = PDBQTMolecule(docked_pdbqt_string, skip_typing=True)
    pdbqt_mol._current_pose = 0
    pdb_string = export_pdb_updated_flexres(chorizo, pdbqt_mol)
    x = get_x_from_pdb_str(pdb_string, "", 15, "SD")
    assert abs(x - 10.577) < 0.0001
    pdbqt_mol._current_pose = 1
    pdb_string = export_pdb_updated_flexres(chorizo, pdbqt_mol)
    x = get_x_from_pdb_str(pdb_string, "", 16, "HG")
    assert abs(x - 18.690) < 0.0001

    # rebuilding chorizo can fail if structure is mangled
    chorizo = LinkedRDKitChorizo.from_pdb_string(
        pdb_string,
        chem_templates,
        mk_prep,
    )
    chorizo.flexibilize_sidechain(":16", mk_prep)
    assert len(chorizo.get_valid_residues()) == 3
