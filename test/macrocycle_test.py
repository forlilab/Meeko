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
