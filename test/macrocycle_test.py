from meeko import MoleculePreparation
from meeko import PDBQTMolecule
from meeko import PDBQTWriterLegacy
from rdkit import Chem
import pathlib

num_cycle_breaks = {
    "tetrahedron1": 3,
    "tetrahedron2": 3,
    "vancomycin":   3,
    "macrocycle2":   1,
    "macrocycle3":   1,
    "macrocycle4":   2,
    "macrocycle5":   1,
}
workdir = pathlib.Path(__file__).parents[0]
filenames = {name: str(workdir / "macrocycle_data" / ("%s.sdf" % name)) for name in num_cycle_breaks}
mk_prep = MoleculePreparation()

def get_macrocycle_atom_types(pdbqt_string):
    macrocycle_carbon = ['CG0', 'CG1', 'CG2', 'CG3', 'CG4', 'CG5', 'CG6', 'CG7', 'CG8', 'CG9']
    macrocycle_pseudo = [ 'G0',  'G1',  'G2',  'G3',  'G4',  'G5',  'G6',  'G7',  'G8',  'G9']
    cg_atoms = []
    g_atoms = []
    lines = pdbqt_string.split('\n')
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atom_type = line[77:].strip()
            if atom_type in macrocycle_carbon:
                cg_atoms.append(atom_type)
            elif atom_type in macrocycle_pseudo:
                g_atoms.append(atom_type)
    return cg_atoms, g_atoms

def test_external_ring_closure():
    mol_fn = workdir / "macrocycle_data" /"open-ring-3D-graph-intact_small.mol"
    mol = Chem.MolFromMolFile(str(mol_fn), removeHs=False)
    delete_bonds = [(2, 3)]
    glue_pseudos = {2: (-999.9, 0, 0), 3: (42.0, 0, 0.)}
    setups = mk_prep.prepare(
                    mol,
                    delete_ring_bonds=delete_bonds,
                    glue_pseudo_atoms=glue_pseudos)
    assert(len(setups) == 1)
    pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setups[0])
    assert setups[0].ring_closure_info.bonds_removed == [(2, 3)]
    assert 2 in setups[0].ring_closure_info.pseudos_by_atom
    assert 3 in setups[0].ring_closure_info.pseudos_by_atom
    cg_atoms, g_atoms = get_macrocycle_atom_types(pdbqt_string)
    assert(len(cg_atoms) == len(g_atoms))
    assert(len(cg_atoms) == 2 * len(set(cg_atoms)))
    assert(len(g_atoms) == 2 * len(set(g_atoms)))
    assert(len(set(g_atoms)) == 1)
    p = PDBQTMolecule(pdbqt_string)
    glue_x_coords = []
    for atom in p.atoms():
        if atom["atom_type"].startswith("G"):
            glue_x_coords.append(atom["xyz"][0])
    for atom_index, (x, y, z) in glue_pseudos.items():
        mindist = 99999.9
        for xcoord in glue_x_coords:
            mindist = min(mindist, abs(x - xcoord))
        assert(mindist < 1e-3) # loose matching

def test_aromatic_rings_unbroken():
    write_pdbqt = PDBQTWriterLegacy.write_string
    mol = Chem.MolFromMolFile(filenames["macrocycle3"], removeHs=False)
    mk_prep = MoleculePreparation(min_ring_size=5)
    setups = mk_prep(mol)
    assert len(setups) == 1
    setup = setups[0]
    nr_rot = sum([b.rotatable for _, b in setup.bond_info.items()])
    assert nr_rot == 9
    pdbqt_string, is_ok, error_msg = write_pdbqt(setup)
    assert pdbqt_string.count("ENDBRANCH") == 8
    # now we will check that a 6-member ring that is not aromatic does break
    # when we set the min ring size to six or less
    mol2 = Chem.MolFromMolFile(filenames["macrocycle5"], removeHs=False)
    setups_broken = mk_prep(mol2)
    mk_prep = MoleculePreparation(min_ring_size=7) 
    setups_unbroken = mk_prep(mol2)
    assert len(setups_broken) == 1
    assert len(setups_unbroken) == 1
    pdbqt_broken, is_ok, err = write_pdbqt(setups_broken[0])
    pdbqt_unbroken, is_ok, err = write_pdbqt(setups_unbroken[0])
    assert pdbqt_broken.count("ENDBRANCH") > pdbqt_unbroken.count("ENDBRANCH")


def run(molname):
    filename = filenames[molname]
    mol = Chem.MolFromMolFile(filename, removeHs=False)
    setups = mk_prep.prepare(mol)
    pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setups[0])
    cg_atoms, g_atoms = get_macrocycle_atom_types(pdbqt_string)
    assert(len(cg_atoms) == len(g_atoms))
    assert(len(cg_atoms) == 2 * len(set(cg_atoms)))
    assert(len(g_atoms) == 2 * len(set(g_atoms)))
    assert(len(set(g_atoms)) == num_cycle_breaks[molname])

def test_all():
    for molname in num_cycle_breaks:
        run(molname)

def test_untyped_macrocycle():
    fn = str(workdir / "macrocycle_data" / "lorlatinib.mol")
    mol = Chem.MolFromMolFile(fn, removeHs=False)

    # type based, can only break C-C bonds, but we have none
    mk_prep_typed = MoleculePreparation()
    molsetup_typed = mk_prep_typed(mol)[0]
    count_rotatable = 0
    count_breakable = 0
    for bond_id, bond_info in molsetup_typed.bond_info.items():
        count_rotatable += bond_info.rotatable
        count_breakable += bond_info.breakable
    assert count_rotatable == 1
    assert count_breakable == 0

    mk_prep_untyped = MoleculePreparation(untyped_macrocycles=True)
    molsetup_untyped = mk_prep_untyped(mol)[0]
    count_rotatable = 0
    count_breakable = 0
    for bond_id, bond_info in molsetup_untyped.bond_info.items():
        count_rotatable += bond_info.rotatable
        count_breakable += bond_info.breakable
    assert count_rotatable == 9
    assert count_breakable == 1
