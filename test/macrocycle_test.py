from meeko import MoleculePreparation
from meeko import PDBQTMolecule
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
    mk_prep.prepare(
            mol,
            delete_ring_bonds=delete_bonds,
            glue_pseudo_atoms=glue_pseudos)
    pdbqt_string = mk_prep.write_pdbqt_string()
    assert(mk_prep.setup.ring_closure_info["bonds_removed"] == [(2, 3)])
    assert(2 in mk_prep.setup.ring_closure_info["pseudos_by_atom"])
    assert(3 in mk_prep.setup.ring_closure_info["pseudos_by_atom"])
    cg_atoms, g_atoms = get_macrocycle_atom_types(pdbqt_string)
    assert(len(cg_atoms) == len(g_atoms))
    assert(len(cg_atoms) == 2 * len(set(cg_atoms)))
    assert(len(g_atoms) == 2 * len(set(g_atoms)))
    assert(len(set(g_atoms)) == 1)
        
    p = PDBQTMolecule(pdbqt_string)
    imap_ = p._pose_data['smiles_index_map']
    n = int(len(imap_) / 2)
    imap = {}
    for i in range(n):
        imap[imap_[2*i] - 1] = imap_[2*i+1] - 1
    pseudo_by_atom = {}
    for idx in mk_prep.setup.atom_pseudo:
        pseudo_by_atom[mk_prep.setup.get_neigh(idx)[0]] = idx
    for atom_index, (x, y, z) in glue_pseudos.items():
        pseudo_index = pseudo_by_atom[atom_index]
        pseudo_index_pdbqt = mk_prep._writer._numbering[pseudo_index] - 1
        xcoord = p._positions[0][pseudo_index_pdbqt, 0]
        assert(abs(xcoord - x) < 1e-3)

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
