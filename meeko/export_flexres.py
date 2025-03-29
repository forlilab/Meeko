from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D
from .utils.utils import parse_begin_res
from .utils.utils import mini_periodic_table
from .rdkit_mol_create import RDKitMolCreate
from .polymer import Polymer


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

def export_pdb_updated_flexres(polymer, pdbqt_mol):
    flexres_id = pdbqt_mol._pose_data["mol_index_to_flexible_residue"]
    new_positions = {}
    for mol_idx, atom_idxs in pdbqt_mol._atom_annotations["mol_index"].items():
        if flexres_id[mol_idx] is not None:
            res_id = parse_begin_res(flexres_id[mol_idx])
            atoms = pdbqt_mol.atoms(atom_idxs)
            molsetup_to_pdbqt = pdbqt_mol._pose_data["index_map"][mol_idx]
            
            molsetup_to_template = polymer.monomers[res_id].molsetup_mapidx
            template_to_molsetup = {j: i for i, j in molsetup_to_template.items()}
            
            # use smiles and smiles_index_map in PDBQT REMARK
            # smiles will be None if it's a typical flexres
            # but there will be a valid Smiles string if it's a covalent flexres
            if pdbqt_mol._pose_data["smiles"][mol_idx] is not None: 

                orig_resmol = Chem.Mol(polymer.monomers[res_id].rdkit_mol)
                orig_atomnames = polymer.monomers[res_id].atom_names
                
                backbone_SMARTS = "[NX3]([H])[CX4]([H])[CX3](=O)"
                expected_names = ('N', 'H', 'CA', 'HA', 'C', 'O')

                backbone_qmol = Chem.MolFromSmarts(backbone_SMARTS)
                backbone_matches = orig_resmol.GetSubstructMatches(backbone_qmol)

                if not backbone_matches: 
                    raise RuntimeError(f"Could not find standard backbone structures in receptor {res_id}. ")
                backbone_expected = tuple(orig_atomnames.index(atom_name) for atom_name in expected_names)

                if backbone_expected not in backbone_matches: 
                    raise RuntimeError(f"Could not confirm residue's backbone by atom names in receptor {res_id}. ")
                
                not_backbone = [atom.GetIdx() for atom in orig_resmol.GetAtoms() if atom.GetIdx() not in backbone_expected]
                orig_backbone = Chem.RWMol(orig_resmol)
                for idx in sorted(not_backbone, reverse=True):
                    orig_backbone.RemoveAtom(idx)
                orig_backbone = orig_backbone.GetMol()
              
                covres_mol = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol, 
                                                           only_cluster_leads=False, 
                                                           keep_flexres=True)[mol_idx]
                
                covres_conformer = covres_mol.GetConformer()
                CA_index = None
                target_coord = orig_backbone.GetConformer().GetPositions()[expected_names.index('CA')]
                tolerance = 5e-3
                for atom in covres_mol.GetAtoms():
                    idx = atom.GetIdx()
                    pos = covres_conformer.GetAtomPosition(idx)
                    if (abs(pos.x - target_coord[0]) <= tolerance and
                        abs(pos.y - target_coord[1]) <= tolerance and
                        abs(pos.z - target_coord[2]) <= tolerance):
                        CA_index = idx
                        break
                if CA_index is None:
                    raise RuntimeError(f"Could not determine CA by coordinates in docked pose of {res_id}. ")
                
                CA_in_covres = covres_mol.GetAtomWithIdx(CA_index)
                bonds_to_recover = []
                for bond in CA_in_covres.GetBonds():
                    neighbor_idx = bond.GetOtherAtomIdx(CA_index)
                    bond_type = bond.GetBondType()
                    bonds_to_recover.append((neighbor_idx, bond_type))

                combined_res = Chem.RWMol(Chem.CombineMols(covres_mol, orig_backbone))
                for neighbor_idx, bond_type in bonds_to_recover: 
                    new_CA_idx = covres_mol.GetNumAtoms() + expected_names.index('CA')
                    combined_res.AddBond(new_CA_idx, neighbor_idx, bond_type)

                CA_h_idx = [nei.GetIdx() for nei in combined_res.GetAtomWithIdx(CA_index).GetNeighbors() if nei.GetAtomicNum()==1]
                for atom_idx in sorted(CA_h_idx + [CA_index], reverse=True): 
                    combined_res.RemoveAtom(atom_idx)
                combined_res = combined_res.GetMol()

                polymer.monomers[res_id].rdkit_mol = combined_res
                polymer.monomers[res_id].atom_names = [atom.GetSymbol() for atom in combined_res.GetAtoms()]
                new_positions[res_id] = {idx: coord for idx,coord in enumerate(combined_res.GetConformer().GetPositions())}

            else: # use templates
            
                # use index_map stored in PDBQT REMARK
                if molsetup_to_pdbqt:
                    template_to_pdbqt = {}
                    for i, j in molsetup_to_pdbqt.items():
                        template_to_pdbqt[molsetup_to_template[i - 1]] = j - 1


                # use polymer template to match sidechain
                else:
                    mol = sidechain_to_mol(atoms)
                    key = polymer.monomers[res_id].residue_template_key
                    template = polymer.residue_chem_templates.residue_templates[key]
                    _, template_to_pdbqt = template.match(mol)

                sidechain_positions = {}
                molsetup_matched = set()
                for i, j in template_to_pdbqt.items():
                    molsetup_matched.add(template_to_molsetup[i])
                    sidechain_positions[i] = tuple(atoms["xyz"][j])
                if len(molsetup_matched) != len(template_to_pdbqt):
                    raise RuntimeError(f"{len(molsetup_matched)} {len(template_to_pdbqt)=}")
                is_flexres_atom = polymer.monomers[res_id].is_flexres_atom 
                hit_count = sum([is_flexres_atom[i] for i in molsetup_matched])
                if hit_count != len(molsetup_matched):
                    raise RuntimeError(f"{hit_count=} {len(molsetup_matched)=}")
                if hit_count != sum(is_flexres_atom):
                    raise RuntimeError(f"{hit_count=} {sum(is_flexres_atom)=}")
                # new_positions[res_id] = sidechain_positions

                # remove root atom(s) (often C-alpha) and first atom after bond
                flex_model = polymer.monomers[res_id].molsetup.flexibility_model
                root_body_idx = flex_model["root"]
                graph = flex_model["rigid_body_graph"]
                conn = flex_model["rigid_body_connectivity"]
                rigid_index_by_atom = flex_model["rigid_index_by_atom"]
                first_after_root = set()
                for other_body_idx in graph[root_body_idx]:
                    first_after_root.add(conn[(root_body_idx, other_body_idx)][1])
                to_pop = set()
                for index in sidechain_positions:
                    index_molsetup = template_to_molsetup[index]
                    if (
                        rigid_index_by_atom[index_molsetup] == root_body_idx or 
                        index_molsetup in first_after_root
                    ): 
                        to_pop.add(index) 
                for index in to_pop:
                    sidechain_positions.pop(index)
                new_positions[res_id] = sidechain_positions
    
    pdbstr = polymer.to_pdb(new_positions)
    return pdbstr

def pdb_updated_flexres_from_rdkit(polymer:Polymer, flexres_rdkit_mols:dict):
    """Take dict of flexible residue RDKit molecules and update the polymer PDB with their positions

    Args:
        polymer (Polymer): receptor Polymer to edit
        flexres_rdkit (dict): Dict of RDKit molecule objects of updated flexible residue positions. Key is residue ID number

    Returns:
        str: PDB string for receptor with updated flexres positons
    """
    # TODO: make this a helper function callable by the function above?

    new_positions = {}
    for res_id, mol in flexres_rdkit_mols.items():
        print(Chem.MolToSmiles(mol))
        #mol = Chem.RemoveHs(mol)
        # get templates for matching indices of rdkit mol to monomer in polymer
        key = polymer.monomers[res_id].residue_template_key
        template = polymer.residue_chem_templates.residue_templates[key]
        _, template_to_pdbqt = template.match(mol)
        molsetup_to_template = polymer.monomers[res_id].molsetup_mapidx
        template_to_molsetup = {j: i for i, j in molsetup_to_template.items()}

        sidechain_positions = {}
        molsetup_matched = set()
        matched_but_ignored = 0
        for i, j in template_to_pdbqt.items():
            index_molsetup = template_to_molsetup[i]
            if polymer.monomers[res_id].molsetup.atoms[index_molsetup].is_ignore:  # do not include ignored atoms (non-polar hydrogens)
                matched_but_ignored += 1
            else:
                molsetup_matched.add(template_to_molsetup[i])
                sidechain_positions[i] = mol.GetConformer().GetAtomPosition(j)
        if len(molsetup_matched) != len(template_to_pdbqt) - matched_but_ignored:
            raise RuntimeError(f"{len(molsetup_matched)} {len(template_to_pdbqt)=}")
        is_flexres_atom = polymer.monomers[res_id].is_flexres_atom
        hit_count = sum([is_flexres_atom[i] for i in molsetup_matched])
        if hit_count != len(molsetup_matched):
            raise RuntimeError(f"{hit_count=} {len(molsetup_matched)=}")
        if hit_count != sum(is_flexres_atom):
            raise RuntimeError(f"{hit_count=} {sum(is_flexres_atom)=}")
        # new_positions[res_id] = sidechain_positions

        # remove root atom(s) (often C-alpha) and first atom after bond
        flex_model = polymer.monomers[res_id].molsetup.flexibility_model
        root_body_idx = flex_model["root"]
        graph = flex_model["rigid_body_graph"]
        conn = flex_model["rigid_body_connectivity"]
        rigid_index_by_atom = flex_model["rigid_index_by_atom"]
        first_after_root = set()
        for other_body_idx in graph[root_body_idx]:
            first_after_root.add(conn[(root_body_idx, other_body_idx)][1])
        to_pop = set()
        for index in sidechain_positions:
            print(index)
            index_molsetup = template_to_molsetup[index]
            if (
                rigid_index_by_atom[index_molsetup] == root_body_idx or 
                index_molsetup in first_after_root
            ): 
                to_pop.add(index) 
        for index in to_pop:
            sidechain_positions.pop(index)
        new_positions[res_id] = sidechain_positions

    pdbstr = polymer.to_pdb(new_positions=new_positions)
    return pdbstr
