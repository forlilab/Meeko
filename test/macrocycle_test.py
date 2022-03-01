from meeko import MoleculePreparation
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
workdir = pathlib.Path(__file__)
filenames = {name: str(workdir.parents[0] / "macrocycle_data" / ("%s.sdf" % name)) for name in num_cycle_breaks}
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

def run(molname):
    filename = filenames[molname]
    mol = Chem.MolFromMolFile(filename)
    mk_prep.prepare(mol)
    pdbqt_string = mk_prep.write_pdbqt_string()
    cg_atoms, g_atoms = get_macrocycle_atom_types(pdbqt_string)
    assert(len(cg_atoms) == len(g_atoms))
    assert(len(cg_atoms) == 2 * len(set(cg_atoms)))
    assert(len(g_atoms) == 2 * len(set(g_atoms)))
    assert(len(set(g_atoms)) == num_cycle_breaks[molname])

def test_all():
    for molname in num_cycle_breaks:
        run(molname)
