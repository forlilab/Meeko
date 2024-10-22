from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D
from .utils.utils import parse_begin_res
from .utils.utils import mini_periodic_table


mini_periodic_table = {v: k for k, v in mini_periodic_table.items()}

def sidechain_to_mol(pdbqt_atoms):
    positions = []
    mol = Chem.EditableMol(Chem.Mol())
    mol.BeginBatchEdit()
    for row in pdbqt_atoms:
        # the first character of the PDB atom name seems to be the most
        # reliable way to get the element from PDBQT files written by
        # meeko or mgltools
        if len(row["name"]) > 1 and row["name"][0].isdecimal():
            element = row["name"][1]
        elif len(row["name"]) > 0:
            element = row["name"][0] 
        atomic_nr = mini_periodic_table[element]
        atom = Chem.Atom(atomic_nr)
        mol.AddAtom(atom)
        x, y, z = row["xyz"]
        positions.append(Point3D(float(x), float(y), float(z)))
    mol.CommitBatchEdit()
    mol = mol.GetMol()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, position in enumerate(positions):
        conformer.SetAtomPosition(index, position)
    mol.AddConformer(conformer, assignId=True)
    rdDetermineBonds.DetermineConnectivity(mol)
    Chem.SanitizeMol(mol)
    return mol

def export_pdb_updated_flexres(chorizo, pdbqt_mol):
    flexres_id = pdbqt_mol._pose_data["mol_index_to_flexible_residue"]
    new_positions = {}
    for mol_idx, atom_idxs in pdbqt_mol._atom_annotations["mol_index"].items():
        if flexres_id[mol_idx] is not None:
            res_id = parse_begin_res(flexres_id[mol_idx])
            atoms = pdbqt_mol.atoms(atom_idxs)
            molsetup_to_pdbqt = pdbqt_mol._pose_data["index_map"][mol_idx]
            molsetup_to_template = chorizo.residues[res_id].molsetup_mapidx
            template_to_molsetup = {j: i for i, j in molsetup_to_template.items()}

            # use index_map stored in PDBQT REMARK
            if molsetup_to_pdbqt:
                template_to_pdbqt = {}
                for i, j in molsetup_to_pdbqt.items():
                    template_to_pdbqt[molsetup_to_template[i - 1]] = j - 1

            # use chorizo template to match sidechain
            else:
                mol = sidechain_to_mol(atoms)
                key = chorizo.residues[res_id].residue_template_key
                template = chorizo.residue_chem_templates.residue_templates[key]
                _, template_to_pdbqt = template.match(mol)

            sidechain_positions = {}
            molsetup_matched = set()
            for i, j in template_to_pdbqt.items():
                molsetup_matched.add(template_to_molsetup[i])
                sidechain_positions[i] = tuple(atoms["xyz"][j])
            if len(molsetup_matched) != len(template_to_pdbqt):
                raise RuntimeError(f"{len(molsetup_matched)} {len(template_to_pdbqt)=}")
            is_flexres_atom = chorizo.residues[res_id].is_flexres_atom 
            hit_count = sum([is_flexres_atom[i] for i in molsetup_matched])
            if hit_count != len(molsetup_matched):
                raise RuntimeError(f"{hit_count=} {len(molsetup_matched)=}")
            if hit_count != sum(is_flexres_atom):
                raise RuntimeError(f"{hit_count=} {sum(is_flexres_atom)=}")
            new_positions[res_id] = sidechain_positions
    pdbstr = chorizo.to_pdb(new_positions)
    return pdbstr
