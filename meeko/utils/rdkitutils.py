from rdkit import Chem
from rdkit.Chem import rdChemReactions
from .utils import mini_periodic_table
from .pdbutils import PDBAtomInfo
from rdkit.Geometry import Point3D
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import rdPartialCharges

periodic_table = Chem.GetPeriodicTable()


"""
create new RDKIT residue

mi  =  Chem.AtomPDBResidueInfo()
mi.SetResidueName('MOL')
mi.SetResidueNumber(1)
mi.SetOccupancy(0.0)
mi.SetTempFactor(0.0)

source: https://sourceforge.net/p/rdkit/mailman/message/36404394/
"""

def set_h_isotope_atom_coords(mol: Chem.Mol, conf: Chem.Conformer) -> dict[int, Point3D]: 

    """
    use AddHs() and RemoveHs() to generate H isotopes coordinates
    in a mol copy as if they were regular Hs
    returns a dict of H isotope atom idx and assigned position as Point3D
    """

    # check if molecule has H isotopes
    def is_h_isotope(atom: Chem.Atom) -> bool:
        return atom.GetAtomicNum() == 1 and atom.GetIsotope() > 0
    def has_h_isotopes(mol: Chem.Mol) -> bool:
        for atom in mol.GetAtoms():
            if is_h_isotope(atom):
                return True
        return False
    
    if not has_h_isotopes(mol): 
        return {}
    
    # create a nested dictionary for index mapping, isotope type and chirality flag
    def get_h_isotope_data(mol: Chem.Mol) -> dict: 
        """
        example output: 
        isotope_data = {
            'H_isotope_idx': {  
                1: {'parent': 0, 'Isotope': 2}, 
                2: {'parent': 0, 'Isotope': 3}, 
                3: {'parent': 4, 'Isotope': 3}, 
            },
            'parent_idx': {  
                0: {'kids': [1, 2], 'CIPCode': 'R'}, 
                4: {'kids': [3], 'CIPCode': None}, 
            }
        }
        """
        
        # initialize 
        isotope_data = {
            'H_isotope_idx': {}, 
            'parent_idx': {}
        }
        H_isotope_idxs = [
            atom.GetIdx() for atom in mol.GetAtoms() 
            if is_h_isotope(atom)
        ]
        parent_idxs = []
        
        # populate H_isotope section
        for idx in H_isotope_idxs: 
            atom = mol.GetAtomWithIdx(idx)
            parent_idx = [nei.GetIdx() for nei in atom.GetNeighbors()][0]
            isotope_data['H_isotope_idx'][idx] = {
                'parent': parent_idx, 'Isotope': atom.GetIsotope()
            }
            if parent_idx not in parent_idxs:
                parent_idxs.append(parent_idx)
        
        # re-compute CIP labels in a copy
        mol_copy = Chem.Mol(mol)
        Chem.rdCIPLabeler.AssignCIPLabels(mol_copy)
        
        # populate parent section
        for idx in parent_idxs: 
            atom = mol.GetAtomWithIdx(idx)
            kids = [nei.GetIdx() for nei in atom.GetNeighbors() if is_h_isotope(nei)]
            cip_code = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None
            isotope_data['parent_idx'][idx] = {
                'kids': kids, 'CIPCode': cip_code
            }

        return isotope_data
    
    isotope_data = get_h_isotope_data(mol)
    H_isotope_idxs = list(isotope_data["H_isotope_idx"].keys())
    parent_idxs = list(isotope_data["parent_idx"].keys())

    # in an editable copy, remove isotopes and convert to explicit Hs
    editable_mol = Chem.RWMol(mol)
    idxs_to_remove = sorted(H_isotope_idxs, reverse=True)
    for idx in idxs_to_remove:
        parent_atom = editable_mol.GetAtomWithIdx(
            isotope_data['H_isotope_idx'][idx]['parent']
        )
        editable_mol.RemoveAtom(idx)
        parent_atom.SetNumExplicitHs(parent_atom.GetNumExplicitHs() + 1)
    copy_mol = editable_mol.GetMol()

    # track idx shift after isotope removal
    # mapping original idx to shifted idx
    atom_idx_trace = {}
    shifted_conf = Chem.Conformer(copy_mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()): 
        if i not in idxs_to_remove: 
            shift = len([idx for idx in idxs_to_remove if idx < i])
            atom_idx_trace[i] = i - shift
            # make a conformer copy with shifted idx
            atom_idx_shifted,  atom_idx_original = (atom_idx_trace[i], i)
            shifted_conf.SetAtomPosition(atom_idx_shifted, 
                                        conf.GetAtomPosition(atom_idx_original))
    
    # reset conformers, setup new conformer in mol copy, add Hs
    copy_mol.RemoveAllConformers()
    copy_mol.AddConformer(shifted_conf)
    copy_mol = Chem.AddHs(copy_mol, addCoords=True)

    # reorder atoms, make sure to have regular Hs at the isotopes positions
    # as placeholders in order to restore the original ordering of all 
    # heavy atoms AND explicit H isotopes
    placeholders = {}
    for parent_idx in isotope_data["parent_idx"]: 
        # get current (shifted) idx of parent atom that has isotope kids
        parent_idx_shifted = atom_idx_trace[parent_idx]
        # get parent atom
        parent_atom = copy_mol.GetAtomWithIdx(parent_idx_shifted)
        # get isotope kids' original idxs
        isotope_kid_idxs = isotope_data["parent_idx"][parent_idx]["kids"]
        # get just enough current (shifted) idxs of avail kids of the parent atom
        avail_kids_idxs_shifted = [kid.GetIdx() for kid in parent_atom.GetNeighbors() 
                                   if kid.GetAtomicNum()==1][:len(isotope_kid_idxs)]
        # re-order move these regular Hs
        for i in range(len(isotope_kid_idxs)): 
            # for the absent isotope (key)
            # move avail regular H (value) to the position as a placeholder
            placeholders[isotope_kid_idxs[i]] = avail_kids_idxs_shifted[i]
    # merge the expected ordering placeholders into the tracked idx of heavy atoms
    # so we get the mapping to restore the original ordering of all heavy atoms
    # AND explicit H isotopes 
    atom_idx_trace = placeholders | atom_idx_trace
    # prepare a list of wanted order
    wanted_order = [v for k, v in sorted(atom_idx_trace.items())]
    # extend the list with no additional moves of un-used hydrogens
    for i in range(copy_mol.GetNumAtoms()): 
        if i not in wanted_order: 
            wanted_order.append(i)
    # re-order and overwrite the copy_mol
    copy_mol = Chem.RenumberAtoms(copy_mol, wanted_order)
    copy_conf = copy_mol.GetConformer()

    # assign H isotope coordinates from the regularized equivalent H
    assigned_isotope_pos = {}
    for parent_idx in parent_idxs:

        # get parent atom in mol copy
        parent_atom = copy_mol.GetAtomWithIdx(parent_idx)
        # get avail Hs of parent atom
        kids_all = [nei for nei in parent_atom.GetNeighbors() if nei.GetAtomicNum() == 1]
        # get regular Hs idx
        kids_idxs_all = [nei.GetIdx() for nei in kids_all]
        
        # initial assignment, in an uncontrolled order
        for ik, kid in enumerate(isotope_data['parent_idx'][parent_idx]['kids']):
            copy_mol.GetAtomWithIdx(kids_idxs_all[ik]).SetIsotope(
                isotope_data["H_isotope_idx"][kid]['Isotope']
            )
            assigned_isotope_pos[kid] = copy_conf.GetAtomPosition(kids_idxs_all[ik])
        
        # make correction to recover CIPCode (chirality)
        expected_cip_code = isotope_data['parent_idx'][parent_idx]['CIPCode']
        if len(kids_all) >1 and expected_cip_code is not None: 
            # evalaute current CIPCode from 3D
            Chem.AssignStereochemistryFrom3D(copy_mol)
            current_cip_code = parent_atom.GetProp('_CIPCode') if parent_atom.HasProp('_CIPCode') else None
            if current_cip_code==expected_cip_code: 
                continue
            # get a list of isotope types on current parent
            kids_types = [nei.GetIsotope() for nei in kids_all]
            # in case the chirality can't be inverted due to unsolvable problems
            if len(set(kids_types)) <= 1: # not enough isotopes
                raise RuntimeError(
                    f"Unable to recover original chirality by manipulating H sotope positions: \n"
                    f"Atom # ({parent_idx}) \n"
                    f"Current CIPCode ({current_cip_code}) \n" 
                    f"Expected CIPCode: ({expected_cip_code})\n"
                    "Its chirality might have changed due to re-arrangements of heavy atoms, "
                    "or become ambiguous to Chem.rdmolops.AssignStereochemistry due to strained geometry. "
                )
            # swap assignment of max and min isotope, re-evaluate CIPCode
            # find the max and min isotope type values
            max_isotope_type, min_isotope_type = (max(kids_types), min(kids_types))
            # get kiw (kid's index with) max and min isotope types
            kiw_max_isotope = kids_idxs_all[kids_types.index(max(kids_types))]
            kiw_min_isotope = kids_idxs_all[kids_types.index(min(kids_types))]
            copy_mol.GetAtomWithIdx(kiw_max_isotope).SetIsotope(min_isotope_type)
            copy_mol.GetAtomWithIdx(kiw_min_isotope).SetIsotope(max_isotope_type)
            # re-evaluate current CIPCode from 3D
            Chem.AssignStereochemistryFrom3D(copy_mol)
            current_cip_code = parent_atom.GetProp('_CIPCode') if parent_atom.HasProp('_CIPCode') else None
            # a single swap of max and min isotopes normally inverts the chirality, in case not
            if current_cip_code!=expected_cip_code: 
                raise RuntimeError(
                    "Failed to recover original chirality after attempts to manipulate H sotope positions: \n"
                    f"Atom # ({parent_idx}) \n"
                    f"Current CIPCode ({current_cip_code}) \n" 
                    f"Expected CIPCode: ({expected_cip_code})\n"
                    "Its chirality might have changed due to re-arrangements of heavy atoms, "
                    "or become ambiguous to Chem.rdmolops.AssignStereochemistry due to strained geometry. "
                )
            # apply the swap 
            max_pos = copy_conf.GetAtomPosition(kiw_max_isotope)
            min_pos = copy_conf.GetAtomPosition(kiw_min_isotope)
            assigned_isotope_pos[kiw_max_isotope] = min_pos
            # only assign positions if min isotope type is greater than 0 (not regular Hs)
            if min_isotope_type>0: 
                assigned_isotope_pos[kiw_min_isotope] = max_pos

    return assigned_isotope_pos

        
def getPdbInfoNoNull(atom):
    """extract information for populating an ATOM/HETATM line
    in the PDB"""
    minfo = atom.GetMonomerInfo()  # same as GetPDBResidueInfo
    if minfo is None:
        atomic_number = atom.GetAtomicNum()
        if atomic_number == 0:
            name = "%-2s" % "*"
        else:
            name = "%-2s" % mini_periodic_table[atomic_number]
        chain = " "
        resNum = 1
        icode = ""
        resName = "UNL"
    else:
        name = minfo.GetName()
        chain = minfo.GetChainId()
        resNum = minfo.GetResidueNumber()
        icode = minfo.GetInsertionCode()
        resName = minfo.GetResidueName()
    return PDBAtomInfo(
        name=name, resName=resName, resNum=resNum, icode=icode, chain=chain
    )


class Mol2MolSupplier:
    """RDKit Mol2 molecule supplier.
    Parameters
        sanitize: perform RDKit sanitization of Mol2 molecule"""

    def __init__(
        self, filename, sanitize=True, removeHs=False, cleanupSubstructures=True
    ):
        self.fp = open(filename, "r")
        self._opts = {
            "sanitize": sanitize,
            "removeHs": removeHs,
            "cleanupSubstructures": cleanupSubstructures,
        }
        self.buff = []

    def __iter__(self):
        return self

    def __next__(self):
        """iterator step"""
        while True:
            line = self.fp.readline()
            # empty line
            if not line:
                if len(self.buff):
                    # buffer full, returning last molecule
                    mol = Chem.MolFromMol2Block("".join(self.buff), **self._opts)
                    self.buff = []
                    return mol
                # buffer empty, stopping the iteration
                self.fp.close()
                raise StopIteration
            if "@<TRIPOS>MOLECULE" in line:
                # first molecule parsed
                if len(self.buff) == 0:
                    self.buff.append(line)
                else:
                    # found the next molecule, breaking to return the complete one
                    break
            else:
                # adding another line in the current molecule
                self.buff.append(line)
        # found a complete molecule, returning it
        mol = Chem.MolFromMol2Block("".join(self.buff), **self._opts)
        self.buff = [line]
        return mol

class AtomField:
    """Stores data parsed from PDB or mmCIF"""

    def __init__(
        self,
        atomname: str,
        altloc: str,
        resname: str,
        chain: str,
        resnum: int,
        icode: str,
        x: float,
        y: float,
        z: float,
        element: str,
    ):
        self.atomname = atomname
        self.altloc = altloc
        self.resname = resname
        self.chain = chain
        self.resnum = resnum
        self.icode = icode
        self.x = x
        self.y = y
        self.z = z
        if len(element) > 1:
            element = f"{element[0].upper()}{element[1].lower()}"
        else:
            element = f"{element.upper()}"
        self.atomic_nr = periodic_table.GetAtomicNumber(element)


def _build_rdkit_mol_for_altloc(atom_fields_list, wanted_altloc:str=None):
    mol = Chem.EditableMol(Chem.Mol())
    mol.BeginBatchEdit() 
    positions = []
    idx_to_rdkit = {}
    for index_list, atom in enumerate(atom_fields_list):
        if wanted_altloc is not None:
            if atom.altloc and atom.altloc != wanted_altloc:
                # if atom.altloc is "" we still want to consider this atom
                continue
        rdkit_atom = Chem.Atom(atom.atomic_nr)
        positions.append(Point3D(atom.x, atom.y, atom.z))
        res_info = Chem.AtomPDBResidueInfo()
        res_info.SetName(atom.atomname)
        res_info.SetResidueName(atom.resname)
        res_info.SetResidueNumber(atom.resnum)
        res_info.SetChainId(atom.chain)
        res_info.SetInsertionCode(atom.icode)
        rdkit_atom.SetPDBResidueInfo(res_info)
        index_rdkit = mol.AddAtom(rdkit_atom)
        idx_to_rdkit[index_list] = index_rdkit
    mol.CommitBatchEdit()
    mol = mol.GetMol()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, position in enumerate(positions):
        conformer.SetAtomPosition(index, position)
    mol.AddConformer(conformer, assignId=True)
    return mol, idx_to_rdkit
        

def build_one_rdkit_mol_per_altloc(atom_fields_list):
    """ if no altlocs, the only key in the output dict is None
        if altlocs exist, None is not a key: the keys are the altloc IDs
    """
    altlocs = set([atom.altloc for atom in atom_fields_list if atom.altloc])
    rdkit_mol_dict = {}
    if not altlocs:
        altlocs = {None}
    for altloc in altlocs:
        mol, idx_to_rdkit = _build_rdkit_mol_for_altloc(atom_fields_list, altloc)
        rdkit_mol_dict[altloc] = (mol, idx_to_rdkit)
    return rdkit_mol_dict


def _aux_altloc_mol_build(atom_field_list, requested_altloc, default_altloc):
    missed_altloc = False
    needed_altloc = False
    mols_dict = build_one_rdkit_mol_per_altloc(atom_field_list) 
    has_altloc = None not in mols_dict
    if has_altloc and requested_altloc is None and default_altloc is None:
        pdbmol = None
        missed_altloc = False 
        needed_altloc = True
    elif requested_altloc and requested_altloc in mols_dict:
        pdbmol, idx_to_rdkit = mols_dict[requested_altloc]
    elif requested_altloc and requested_altloc not in mols_dict:
        pdbmol = None
        missed_altloc = True
        needed_altloc = False
    elif default_altloc and default_altloc in mols_dict:
        pdbmol, idx_to_rdkit = mols_dict[default_altloc]
    elif has_altloc and default_altloc not in mols_dict:
        pdbmol = None
        missed_altloc = True
        needed_altloc = False
    elif not has_altloc and requested_altloc is None:
        pdbmol, idx_to_rdkit = mols_dict[None]
    else:
        raise RuntimeError("programming bug, please post full error on github")
    if pdbmol is None: 
        idx_to_rdkit = None
        return pdbmol, idx_to_rdkit, missed_altloc, needed_altloc
    else:
        rdDetermineBonds.DetermineConnectivity(pdbmol)
        for atom in pdbmol.GetAtoms():
            if atom.GetAtomicNum() == 7 and len(atom.GetNeighbors()) == 4:
                atom.SetFormalCharge(1)
        _ = Chem.SanitizeMol(pdbmol)

    return pdbmol, idx_to_rdkit, missed_altloc, needed_altloc

def react_and_map(reactants: tuple[Chem.Mol], rxn: rdChemReactions.ChemicalReaction):
    """
    Run a reaction and keep track of atom indices from reactants to products.
    
    Parameters
    ----------
    reactants : tuple[Chem.Mol]
        A tuple of RDKit molecule objects representing the reactants.
    rxn : rdChemReactions.ChemicalReaction
        The RDKit reaction object.
        
    Returns
    -------
    list[tuple[Chem.Mol, dict[str, list[Optional[int]]]]]
        A list of tuples where each tuple contains a product molecule and a dictionary.
        The dictionary has keys 'atom_idx' and 'new_atom_label', which are ordered lists for product atoms:
        - 'atom_idx' holds the corresponding atom indices in reactant. None for newly added atoms. 
        - 'new_atom_label' holds the reaction mapping number, only for newly added atoms. 
    """

    # Prepare for multiple possible outcomes resulted from multiple matched reactive sites in reactant
    outcomes = []
    for products in rxn.RunReactants(reactants): 
        # Assumes single product 
        product = products[0]
        # For each atom, get react_atom_idx if they were in reactant
        atom_idxmap = [
            atom.GetIntProp("react_atom_idx") if atom.HasProp("react_atom_idx")
            else None
            for atom in product.GetAtoms()
        ]
        # For each atom, get the rxn mapping number if the were added in the rxn
        new_atom_label = [
            atom.GetIntProp("old_mapno") if atom.HasProp("old_mapno") and not atom.HasProp("react_atom_idx")
            else None
            for atom in product.GetAtoms()
        ]
        # Collect product and index_map
        index_map = {"atom_idx": atom_idxmap, "new_atom_label": new_atom_label}
        outcomes.append((product, index_map))

    return outcomes

def remove_elements(mol, to_rm=(12, 20, 25, 26, 30)):
    idx_to_rm = {}
    neigh_idx_to_nr_h = {}
    rm_to_neigh = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in to_rm:
            idx_to_rm[atom.GetIdx()] = atom.GetFormalCharge()
            rm_to_neigh[atom.GetIdx()] = set()
            for neigh in atom.GetNeighbors():
                n = neigh.GetNumExplicitHs()
                neigh_idx_to_nr_h[neigh.GetIdx()] = n
                rm_to_neigh[atom.GetIdx()].add(neigh.GetIdx())
    if not idx_to_rm:
        return Chem.Mol(mol), idx_to_rm, rm_to_neigh
    rwmol = Chem.EditableMol(mol)
    for idx in sorted(idx_to_rm, reverse=True):
        rwmol.RemoveAtom(idx)
    mol = rwmol.GetMol()
    for idx in neigh_idx_to_nr_h:
        n = neigh_idx_to_nr_h[idx]
        newidx = idx - sum([i < idx for i in idx_to_rm]) 
        mol.GetAtomWithIdx(newidx).SetNumExplicitHs(n + 1)
    mol.UpdatePropertyCache()
    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol)
    return mol, idx_to_rm, rm_to_neigh

def compute_gasteiger_charges(rdkit_mol):
    things = remove_elements(rdkit_mol)
    copy_mol, idx_rm_to_formal_charge, rm_to_neigh = things
    for atom in copy_mol.GetAtoms():
        if atom.GetAtomicNum() == 34:
            atom.SetAtomicNum(16)
    rdPartialCharges.ComputeGasteigerCharges(copy_mol)
    charges = [a.GetDoubleProp("_GasteigerCharge") for a in copy_mol.GetAtoms()]
    if idx_rm_to_formal_charge:
        ok_charges = charges.copy()
        for i in sorted(idx_rm_to_formal_charge, reverse=False):
            ok_charges.insert(i, 0.0)
        nr_rm = len(idx_rm_to_formal_charge)
        nr_added_h = copy_mol.GetNumAtoms() - rdkit_mol.GetNumAtoms() + nr_rm
        ok_charges = ok_charges[:len(ok_charges)-nr_added_h]
        h_chrg_by_heavy_atom = {}
        for i in range(nr_added_h):
            added_H_idx = rdkit_mol.GetNumAtoms() + i - nr_rm
            neighs = copy_mol.GetAtomWithIdx(added_H_idx).GetNeighbors()
            if len(neighs) != 1:
                raise RuntimeError("H should have 1 neighbor")
            # in iron-sulfur clusters, sulfur will be added more than one hydrogen
            idx = neighs[0].GetIdx()
            h_chrg_by_heavy_atom.setdefault(idx, 0.0)
            h_chrg_by_heavy_atom[idx] += charges[added_H_idx]
        # in iron-sulfur clusters, each sulfur will donate its added H charges
        # to multiple irons, so we must divide the total donated charge
        # by the number of donations
        contributions_by_neigh = {}
        for i, neighs in rm_to_neigh.items():
            for neigh in neighs:
                contributions_by_neigh.setdefault(neigh, 0)
                contributions_by_neigh[neigh] += 1
        for i, neighs in rm_to_neigh.items():
            ok_charges[i] += idx_rm_to_formal_charge[i]
            for idx in neighs:
                newidx = idx - sum([i <= idx for i in idx_rm_to_formal_charge]) 
                ok_charges[i] += h_chrg_by_heavy_atom[newidx] / contributions_by_neigh[idx]
        charges = ok_charges
    return charges

covalent_radius = {  # from wikipedia
    1: 0.31,
    5: 0.84,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    11: 0.00,  # hack to avoid bonds with salt
    12: 0.00,  # hack to avoid bonds with metals
    14: 1.11,
    15: 1.07,
    16: 1.05,
    17: 1.02,
    # 19: 2.03,
    20: 0.00,
    # 24: 1.39,
    25: 0.00,  # hack to avoid bonds with metals
    26: 0.00,
    30: 0.00,  # hack to avoid bonds with metals
    # 34: 1.20,
    35: 1.20,
    53: 1.39,
}
