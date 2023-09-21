from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem.AllChem import EmbedMolecule, AssignBondOrdersFromTemplate
import json

def mapping_by_mcs(mol, ref):
    mcs_result = rdFMCS.FindMCS([mol,ref])
    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)

    mol_idxs = mol.GetSubstructMatch(mcs_mol)
    ref_idxs = ref.GetSubstructMatch(mcs_mol)

    atom_map = {i:j for (i,j) in zip(mol_idxs, ref_idxs)}
    return atom_map

def reassign_formal_charge(mol, ref, mapping):
    #TODO this could be optimized
    #TODO ref charges could be precalculated to speed up large structures
    mol_charged_atoms = []
    for idx, atom in enumerate(mol.GetAtoms()):
        if atom.GetFormalCharge() != 0:
            mol_charged_atoms.append(idx)
            
    ref_charged_atoms = []
    for idx, atom in enumerate(ref.GetAtoms()):
        if atom.GetFormalCharge() != 0:
            ref_charged_atoms.append(idx)
    
    for (k,v) in mapping.items():
        if k in ref_charged_atoms or v in mol_charged_atoms:
            mol.GetAtomWithIdx(v).SetFormalCharge(ref.GetAtomWithIdx(k).GetFormalCharge())

    return mol

def reassign_bond_orders(mol, ref, mapping):
    #TODO this could be optimized
    for i in mapping.keys():
        for ref_bond in ref.GetAtomWithIdx(i).GetBonds():
            j = ref_bond.GetOtherAtomIdx(i)
            if j in mapping.keys():
                mol_bond = mol.GetBondBetweenAtoms(mapping[i], mapping[j])
                mol_bond.SetBondType(ref_bond.GetBondType())
    return mol

def h_coord_from_dipeptide(pdb1, pdb2):
    mol = Chem.MolFromPDBBlock(pdb1+pdb2)
    mol_h = Chem.AddHs(mol, addCoords=True)
    ps = Chem.SmilesParserParams()
    ps.removeHs = False
    template = Chem.MolFromSmiles('C(=O)C([H])N([H])C(=O)C([H])N',ps)
    h_idx = 5
    atom_map = mapping_by_mcs(template, mol_h)
    
    return mol_h.GetConformer().GetAtomPosition(atom_map[h_idx])

class RDKITReceptor:
    def __init__(self, pdb_path, json_path, res_swaps):
        self._pdb_to_resblocks(pdb_path)
        self._rename_residues(res_swaps)
        self._load_params(json_path)
        self.atoms = []
        self.removed_residues = []
        
    def _load_params(self,json_path):
        #name changes will go in main file, temp fix
        desired_props = {
                "atom_names": ('atom_name',str),
                "atom_types": ('atom_type',str),
                "ad4_epsii": ('ad4_epsii',float),
                "ad4_rii": ('ad4_rii',float),
                "ad4_sol_vol": ('ad4_sol_vol',float),
                "ad4_sol_par": ("ad4_sol_par",float),
                "ad4_hb_rij": ("ad4_hb_rij",float),
                "ad4_hb_epsij": ("ad4_hb_epsij",float),
                "vina_ri": ("vina_ri",float),
                "vina_donor": ("vina_donor",bool),
                "vina_acceptor": ("vina_acceptor",bool),
                "vina_hydrophobic": ("vina_hydrophobic",bool),
                "gasteiger": ("gasteiger",float)
                }
        
        with open(json_path,'r') as fin:
            self.params = json.load(fin)

        ps = Chem.SmilesParserParams()
        ps.removeHs = False

        self.res_templates = {}
        for resn in self.params:
            resmol = Chem.MolFromSmiles(self.params[resn]['smiles'], ps)
            for idx, atom in enumerate(resmol.GetAtoms()):
                for prop, (propname, type) in desired_props.items():
                    if type == bool:
                        atom.SetBoolProp(propname, bool(self.params[resn][prop][idx]))
                    if type == float:
                        if self.params[resn][prop][idx]:
                            atom.SetDoubleProp(propname, float(self.params[resn][prop][idx]))
                    if type == str:
                        atom.SetProp(propname, self.params[resn][prop][idx])
            self.res_templates[resn] = resmol
    
    def _pdb_to_resblocks(self,pdb_path):
        #TODO detect chain breaks
        self.residues = {}
        with open(pdb_path, 'r') as fin:
            current_res = None
            for line in fin:
                if line.startswith('TER'):
                    self.residues[current_res]['next res'] = None
                    current_res = None
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    resname = line[17:20].strip()
                    resid = int(line[22:26].strip())
                    chainid = line[21].strip()
                    full_resid = ':'.join([chainid, resname, str(resid)])

                    if full_resid == current_res:
                        self.residues[full_resid]['pdb block'] += line
                    else:
                        if current_res:
                            self.residues[current_res]['next res'] = full_resid
                        self.residues[full_resid] = {}
                        self.residues[full_resid]['pdb block'] = line
                        self.residues[full_resid]['previous res'] = current_res
                        current_res = full_resid
    
    def _rename_residues(self, resdict):
        for res in resdict:
            old_resn = res.split(':')[1]
            new_resn = resdict[res].split(':')[1]
            self.residues[resdict[res]] = self.residues.pop(res)
            # self.residues[resdict[res]]['pdb block'] = self.residues[resdict[res]]['pdb block'].replace(old_resn, new_resn)
            previous_res = self.residues[resdict[res]]['previous res']
            if previous_res:
                self.residues[previous_res]['next res'] = resdict[res]
            next_res = self.residues[resdict[res]]['next res']
            if next_res:
                self.residues[next_res]['previous res'] = resdict[res]

    def parameterize_residues(self):
        for res in self.residues:
            exclude_idxs = []
            err = ''

            pdbmol = Chem.MolFromPDBBlock(self.residues[res]['pdb block'], removeHs=False)
            
            # Check if parameters are available for a residue and rename if terminal
            resn = res.split(':')[1]
            if resn not in self.res_templates:
                self.residues.pop(res)
                self.removed_residues.append(res)
                return 0
            if self.residues[res]['next res'] == None:
                resn = 'C' + resn
            if self.residues[res]['previous res'] == None:
                resn = 'N' + resn

            # TODO add to preprocessing to save time
            # Create mol object and map between the pdb and residue template
            resmol = Chem.Mol(self.res_templates[resn])
            n_atoms = len(resmol.GetAtoms())
            atom_map = mapping_by_mcs(resmol, pdbmol)

            # Transfer coordinates and info for any matched atoms
            #TODO time these functions
            #TODO maybe embed in preprocessing depending on time
            #EmbedMolecule(resmol)
            Chem.rdDepictor.Compute2DCoords(resmol)
            resmol.GetConformer().Set3D(True)
            for idx in atom_map.keys():
                pdb_idx = atom_map[idx]
                pdb_atom = pdbmol.GetAtomWithIdx(pdb_idx)
                pdb_coord = pdbmol.GetConformer().GetAtomPosition(pdb_idx)
                resmol.GetConformer().SetAtomPosition(idx, pdb_coord)

                resinfo = pdb_atom.GetPDBResidueInfo()
                resmol.GetAtomWithIdx(idx).SetDoubleProp('occupancy', resinfo.GetOccupancy())
                resmol.GetAtomWithIdx(idx).SetDoubleProp('temp_factor', resinfo.GetTempFactor())

            missing_atoms = {resmol.GetAtomWithIdx(i).GetProp('atom_name'):i for i in range(n_atoms) if i not in atom_map.keys()}

            # Handle case of missing backbone amide H
            if 'H' in missing_atoms:
                h_pos = h_coord_from_dipeptide(self.residues[res]['pdb block'], 
                                                self.residues[self.residues[res]['previous res']]['pdb block'])
                resmol.GetConformer().SetAtomPosition(missing_atoms['H'], h_pos)
                resmol.GetAtomWithIdx(missing_atoms['H']).SetBoolProp('computed', True)
                exclude_idxs.append(missing_atoms['H'])
                missing_atoms.pop('H')
                
            
            missing_atom_elements = set([atom[0] for atom in missing_atoms.keys()])
            if len(missing_atom_elements) > 0:
                if missing_atom_elements != set('H'):
                    err += f'{res=} {missing_atoms=}\n'
                
                resmol_h = Chem.RemoveHs(resmol)
                resmol_h = Chem.AddHs(resmol_h, addCoords=True)
                h_map = mapping_by_mcs(resmol, resmol_h)
                for atom in list(missing_atoms.keys()):
                    if atom.startswith('H'):
                        h_idx = missing_atoms[atom]
                        #TODO magic function from Diogo to add H atom
                        #TODO bring in parameterless H atom for chain breaks
                        resmol.GetConformer().SetAtomPosition(h_idx, resmol_h.GetConformer().GetAtomPosition(h_map[h_idx]))
                        resmol.GetAtomWithIdx(h_idx).SetBoolProp('computed', True)
                        missing_atoms.pop(atom)

            if len(missing_atoms) > 0:
                err += f'Could not add {res=} {missing_atoms=}'
            
            self.residues[res]['resmol'] = resmol

            if err:
                print(err)
                with Chem.SDWriter(f'{resn}{res.split(":")[2]}.sdf') as w:
                    w.write(pdbmol)
                    w.write(resmol)

    def write_pdb(self, outpath):
        # TODO currently does not contain residue information
        with open(outpath, 'w') as fout:
            for res in self.residues:
                try:
                    pdb_block = Chem.MolToPDBBlock(self.residues[res]['resmol'])
                except:
                    print(res)
                for line in pdb_block.split('\n'):
                    if line.startswith('HETATM'):
                        fout.write(line+'\n')