import gemmi
import json
import pathlib
import copy
import urllib.request
import time
import tempfile
import re
import atexit
import os

from .utils.rdkitutils import covalent_radius

from rdkit import Chem
from rdkit.Chem import rdmolops

from rdkit import RDLogger
logger = RDLogger.logger()
logger.setLevel(RDLogger.CRITICAL)
import sys, logging
logging.basicConfig(stream=sys.stdout, level=logging.WARNING)


list_of_AD_elements_as_AtomicNum = list(covalent_radius.keys())
metal_AtomicNums = {12, 20, 25, 26, 30}  # Mg: 12, Ca: 20, Mn: 25, Fe: 26, Zn: 30

# Utility Functions
def mol_contains_unexpected_element(mol: Chem.Mol, allowed_elements: list[str] = list_of_AD_elements_as_AtomicNum) -> bool:
    """Check if mol contains unexpected elements"""
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in allowed_elements:
            return True
    return False


def get_atom_idx_by_names(mol: Chem.Mol, wanted_names: set[str] = set()) -> set[int]:
    
    if not wanted_names:
        return set()
    
    wanted_atoms_by_names = {atom for atom in mol.GetAtoms() if atom.GetProp('atom_id') in wanted_names}
    names_not_found = wanted_names.difference({atom.GetProp('atom_id') for atom in wanted_atoms_by_names})
    if names_not_found: 
        logging.warning(f"Molecule doesn't contain the requested atoms with names: {names_not_found} -> continue with found names... ")
    return {atom.GetIdx() for atom in wanted_atoms_by_names}


def get_atom_idx_by_patterns(mol: Chem.Mol, allowed_smarts: str, 
                             wanted_smarts_loc: dict[str, set[int]] = None,
                             allow_multiple: bool=False) -> set[int]:
    
    if wanted_smarts_loc is None:
        return set()
    
    wanted_atoms_idx = set()
    
    tmol = Chem.MolFromSmarts(allowed_smarts)
    match_allowed = mol.GetSubstructMatches(tmol)
    if not match_allowed:
        logging.warning(f"Molecule doesn't contain allowed_smarts: {allowed_smarts} -> no pattern-based action will be made. ")
        return set()
    if len(match_allowed) > 1 and not allow_multiple: 
        logging.warning(f"Molecule contain multiple copies of allowed_smarts: {allowed_smarts} -> no pattern-based action will be made. ")
        return set()
    if len(match_allowed) > 1 and allow_multiple:
        match_allowed = {item for sublist in match_allowed for item in sublist}
    else:
        match_allowed = match_allowed[0]
    
    atoms_in_mol = [atom for atom in mol.GetAtoms()]
    for wanted_smarts in wanted_smarts_loc: 
        lmol = Chem.MolFromSmarts(wanted_smarts)
        match_wanted = mol.GetSubstructMatches(lmol)
        if not match_wanted:
            logging.warning(f"Molecule doesn't contain wanted_smarts: {wanted_smarts} -> continue with next pattern... ")
            continue
        for match_copy in match_wanted:
            match_in_copy = (idx for idx in match_copy if match_copy.index(idx) in wanted_smarts_loc[wanted_smarts])
            match_wanted_atoms = {atoms_in_mol[idx] for idx in match_in_copy if idx in match_allowed}
            if match_wanted_atoms: 
                wanted_atoms_idx.update(atom.GetIdx() for atom in match_wanted_atoms)
    
    return wanted_atoms_idx


# Mol Editing Functions
def embed(mol: Chem.Mol, allowed_smarts: str, 
          leaving_names: set[str] = None, leaving_smarts_loc: dict[str, set[int]] = None, 
          alsoHs: bool = True) -> Chem.Mol:
    """
    Remove atoms from the molecule based the union of
    (a) leaving_names: list of atom IDs (names), and
    (b) leaving_smarts_loc: dict to map substructure SMARTS patterns with 
    tuple of 0-based indicies for atoms to delete (restricted by allowed_smarts)
    """
    if leaving_names is None and leaving_smarts_loc is None:
        return mol

    leaving_atoms_idx = set()

    if leaving_names:
        leaving_atoms_idx.update(get_atom_idx_by_names(mol, leaving_names))

    if leaving_smarts_loc:
        leaving_atoms_idx.update(get_atom_idx_by_patterns(mol, allowed_smarts, leaving_smarts_loc))

    if leaving_atoms_idx and alsoHs:
        atoms_in_mol = [atom for atom in mol.GetAtoms()]
        leaving_Hs = (ne for atom_idx in leaving_atoms_idx for ne in atoms_in_mol[atom_idx].GetNeighbors() if ne.GetAtomicNum() == 1)
        leaving_atoms_idx.update(atom.GetIdx() for atom in leaving_Hs)

    if not leaving_atoms_idx:
        logging.warning(f"No matched atoms to delete -> embed returning original mol...")
        return mol
    
    rwmol = Chem.RWMol(mol)
    for atom_idx in sorted(leaving_atoms_idx, reverse=True): 
        rwmol.RemoveAtom(atom_idx)
    rwmol.UpdatePropertyCache()
    return rwmol.GetMol()


def cap(mol: Chem.Mol, allowed_smarts: str, 
        capping_names: set[str] = None, capping_smarts_loc: dict[str, set[int]] = None, 
        protonate: bool = False) -> Chem.Mol:
    """Add hydrogens to atoms with implicit hydrogens based on the union of
    (a) capping_names: list of atom IDs (names), and
    (b) capping_smarts_loc: dict to map substructure SMARTS patterns with 
    tuple of 0-based indicies for atoms."""
   
    if capping_names is None and capping_smarts_loc is None:
        return mol
    
    capping_atoms_idx = set()
    
    if capping_names:
        capping_atoms_idx.update(get_atom_idx_by_names(mol, capping_names))

    if capping_smarts_loc:
        capping_atoms_idx.update(get_atom_idx_by_patterns(mol, allowed_smarts, capping_smarts_loc, allow_multiple = True))

    if not capping_atoms_idx:
        logging.warning(f"No matched atoms to cap -> cap returning original mol...")
        return mol
    
    def get_max_Hid(mol: Chem.Mol) -> int:
        all_Hids = (atom.GetProp('atom_id') for atom in mol.GetAtoms() if atom.GetAtomicNum()==1)
        regular_ids = {Hid for Hid in all_Hids if Hid[0]=='H' and Hid[1:].isdigit()}
        if len(regular_ids) > 0:
            return max(int(x[1:]) for x in regular_ids)
        else:
            return 0
    
    rwmol = Chem.RWMol(mol)
    new_Hid = get_max_Hid(mol) + 1
    atoms_in_mol = [atom for atom in mol.GetAtoms()]
    for atom_idx in capping_atoms_idx:
        parent_atom  = atoms_in_mol[atom_idx]
        if protonate: 
            parent_atom.SetFormalCharge(parent_atom.GetFormalCharge() + 1)
        needed_Hs = parent_atom.GetNumImplicitHs()
        if needed_Hs == 0:
            logging.warning(f"Atom # {atom_idx} ({parent_atom.GetProp('atom_id')}) in mol doesn't have implicit Hs -> continue with next atom... ")
        else:
            new_atom = Chem.Atom("H")
            new_atom.SetProp('atom_id', f"H{new_Hid}")
            new_Hid += 1
            new_idx = rwmol.AddAtom(new_atom)
            rwmol.AddBond(atom_idx, new_idx, Chem.BondType.SINGLE)
    rwmol.UpdatePropertyCache()
    return rwmol.GetMol()


def deprotonate(mol: Chem.Mol, acidic_proton_loc: dict[str, int] = None) -> Chem.Mol:
    """Remove acidic protons from the molecule based on acidic_proton_loc"""
    # acidic_proton_loc is a mapping 
    # keys: smarts pattern of a fragment
    # value: the index (order in smarts) of the leaving proton

    if acidic_proton_loc is None:
        return mol

    # deprotonate all matched protons
    acidic_protons_idx = set()
    for smarts_pattern, idx in acidic_proton_loc.items():
        qmol = Chem.MolFromSmarts(smarts_pattern)
        acidic_protons_idx.update(match[idx] for match in mol.GetSubstructMatches(qmol))
    
    if not acidic_protons_idx:
        logging.warning(f"Molecule doesn't contain matching atoms for acidic_proton_loc:" + 
                        f"{acidic_proton_loc}" + 
                        f"-> deprotonate returning original mol... ")
        return mol
    
    rwmol = Chem.RWMol(mol)
    for atom_idx in sorted(acidic_protons_idx, reverse=True):
        rwmol.RemoveAtom(atom_idx)
        neighbors = mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if neighbors:
            neighbor_atom = rwmol.GetAtomWithIdx(neighbors[0].GetIdx())
            neighbor_atom.SetFormalCharge(neighbor_atom.GetFormalCharge() - 1)
    
    rwmol.UpdatePropertyCache()
    return rwmol.GetMol()


def recharge(rwmol: Chem.RWMol) -> Chem.RWMol: 
    # Recharging metal complexes
    metal_atoms = set(atom for atom in rwmol.GetAtoms() if atom.GetAtomicNum() in metal_AtomicNums)
    coord_valence = {v: k for k, v_set in {
            1: {1, 9, 17, 35, 53}, # monovalent: H, halogens
            2: {8, 16}, # divalent: O, S
            3: {5, 7, 15}, # trivalent: B, N, P (phosphine)
            4: {6, 14}, # tetravalent: C, Si
            }.items() for v in v_set}
    bond_order_mapping = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 1.5  
    }

    if not metal_atoms: 
        return rwmol
    else:
        # Method a: If charge is ignored at metal center
        # -> charge up nonmetal coordinated atoms 
        # (metal ox state will be interpreted from valence of nonmetal coordinated atom)
        metal_to_nonmetal_neighbors = {metal: {atom for atom in metal.GetNeighbors() 
                                               if atom.GetAtomicNum() not in metal_AtomicNums} for metal in metal_atoms}
        nonmetal_coord_atoms = set(atom for v_set in metal_to_nonmetal_neighbors.values() for atom in v_set)
        #if 1:
        if any(atom.GetFormalCharge() == 0 for atom in metal_atoms): 
            logging.warning(f"Molecule contains metal with unspecified charge state -> charging nonmetal coordinated atoms...")
            logging.warning(f"All metals will be neutralized and the nonmetal coordinated atoms will be charged according to their explicit valence. ")
            for atom in metal_atoms: 
                atom.SetFormalCharge(0)
            for atom in nonmetal_coord_atoms: 
                bond_sum = sum(bond_order_mapping.get(bond.GetBondType(), 0) for bond in atom.GetBonds())
                atom.SetFormalCharge(bond_sum - coord_valence[atom.GetAtomicNum()])
  
        # Method b: If charge (ox) state is specified for all metal centers
        # -> ionize (break bonds w/) and charge down nonmetal coordinated atoms
        # (metal ox state will be from input charge state)
        else: 
            logging.warning(f"Molecule contains metal specified charge state -> ionizing metal-nonmetal coordinate bonds...")
            logging.warning(f"All metal-nonmetal coordinate bonds will be polarized. ")
            metal_bonds = {bond.GetIdx(): (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) 
                               for metal in metal_atoms for bond in metal.GetBonds()}

            for bond_idx in sorted(metal_bonds, reverse=True):
                rwmol.RemoveBond(metal_bonds[bond_idx][0], metal_bonds[bond_idx][1])

            for atom in nonmetal_coord_atoms: 
                bond_sum = sum(bond_order_mapping.get(bond.GetBondType(), 0) for bond in atom.GetBonds())
                atom.SetFormalCharge(bond_sum - coord_valence[atom.GetAtomicNum()])

            # to avoid fragmentation, rebuild all metal-nonmetal bonds
            for metal in metal_atoms: 
                for nei in metal_to_nonmetal_neighbors[metal]:
                    rwmol.AddBond(metal.GetIdx(), nei.GetIdx(), Chem.BondType.SINGLE)
                    metal.SetFormalCharge(metal.GetFormalCharge() - 1)
                    nei.SetFormalCharge(nei.GetFormalCharge() + 1)

        logging.warning(f"Total charge of the molecule after recharging: {sum(atom.GetFormalCharge() for atom in rwmol.GetAtoms())}")

    return rwmol


# Attribute Formatters
def get_smiles_with_atom_names(mol: Chem.Mol) -> tuple[str, list[str]]:
    """Generate SMILES with atom names in the order of SMILES output."""
    # allHsExplicit may expose the implicit Hs of linker atoms to Smiles; the implicit Hs don't have names
    smiles_exh = Chem.MolToSmiles(mol, allHsExplicit=True)

    smiles_atom_output_order = mol.GetProp('_smilesAtomOutputOrder')
    delimiters = ('[', ']', ',')
    for delimiter in delimiters:
        smiles_atom_output_order = smiles_atom_output_order.replace(delimiter, ' ')
    smiles_output_order = (int(x) for x in smiles_atom_output_order.split())

    atom_id_dict = {atom.GetIdx(): atom.GetProp('atom_id') for atom in mol.GetAtoms()}
    atom_name = [atom_id_dict[atom_i] for atom_i in smiles_output_order]

    return smiles_exh, atom_name


def get_pretty_smiles(smi: str) -> str: 
    """Convert Smiles with allHsExplicit to pretty Smiles to be put on chem templates"""
    # collect the inside square brackets contents
    contents = set(re.findall(r'\[([^\]]+)\]', smi))

    def is_nonmetal_element(symbol: str) -> bool:
        """Check if a string represents a valid chemical element."""
        try:
            return Chem.GetPeriodicTable().GetAtomicNumber(symbol) not in metal_AtomicNums
        # rdkit throws RuntimeError if invalid
        except RuntimeError:
            return False

    for content in contents:
        # keep [H] for explicit Hs
        if content == 'H': 
            continue
        # drop H in the content to hide implicit Hs
        H_stripped = content.split('H')[0]
        # drop [ ] if the content is an uncharged nonmetal element symbol 
        if is_nonmetal_element(content) or is_nonmetal_element(H_stripped):
            smi = smi.replace(f"[{content}]", f"{H_stripped}" if 'H' in content else f"{content}")
    return smi


class ChemicalComponent_LoggingControler:

    def __init__(self):
        self.logger = logging.getLogger('ChemicalComponent')
        self.original_level = self.logger.level
        self.rdkit_logger = RDLogger.logger()
        self.default_rdkit_level = RDLogger.WARNING

    def __enter__(self):
        self.rdkit_logger.setLevel(RDLogger.CRITICAL)
        self.logger.setLevel(logging.WARNING)
        handler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(handler)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.rdkit_logger.setLevel(self.default_rdkit_level)
        self.logger.setLevel(self.original_level)
        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)


class ChemicalComponent:

    def __init__(self, rdkit_mol: Chem.Mol, resname: str, smiles_exh: str, atom_name: list[str]):
        self.rdkit_mol = rdkit_mol
        self.resname = resname
        self.parent = resname # default parent to itself
        self.smiles_exh = smiles_exh
        self.atom_name = atom_name
        self.link_labels = {} # default to empty dict (free molecular form)

    def __eq__(self, other):
        if isinstance(other, ChemicalComponent):
            return self.smiles_exh == other.smiles_exh and self.atom_name == other.atom_name
        return False

    @classmethod
    def from_cif(cls, source_cif: str, resname: str):
        """Create ChemicalComponent from a chemical component dict file and a searchable residue name in file."""
        
        # Locate block by resname
        doc = gemmi.cif.read(source_cif)
        block = doc.find_block(resname)
        
        # Populate atom table
        atom_category = '_chem_comp_atom.'
        atom_attributes = ['atom_id', # atom names
                           'type_symbol', # atom elements
                           'charge', # (atomic) formal charges
                           'pdbx_leaving_atom_flag', # flags for leaving atoms after polymerization
                           ]
        atom_table = block.find(atom_category, atom_attributes)
        atom_cols = {attr: atom_table.find_column(f"{atom_category}{attr}") for attr in atom_attributes}

        # Summon rdkit atoms into empty RWMol
        rwmol = Chem.RWMol()
        atom_elements = atom_cols['type_symbol']

        for idx, element in enumerate(atom_elements):
            if len(element)==2:
                element = element[0] + element[1].lower()
            rdkit_atom = Chem.Atom(element)
            for attr in atom_attributes:
                rdkit_atom.SetProp(attr, atom_cols[attr][idx])
                # strip double quotes in names
                raw_name = rdkit_atom.GetProp('atom_id')
                rdkit_atom.SetProp('atom_id', raw_name.strip('"'))
            target_charge = atom_cols['charge'][idx]
            if target_charge!='0':
                rdkit_atom.SetFormalCharge(int(target_charge)) # this needs to be int for rdkit
            rwmol.AddAtom(rdkit_atom)

        # Check if rwmol contains unexpected elements
        if mol_contains_unexpected_element(rwmol):
            logging.warning(f"Molecule contains unexpected elements -> template for {resname} will be None. ")
            return None

        # Map atom_id (atom names) with rdkit idx
        name_to_idx_mapping = {atom.GetProp('atom_id'): idx for (idx, atom) in enumerate(rwmol.GetAtoms())}

        # Populate bond table
        bond_category = '_chem_comp_bond.'
        bond_attributes = ['atom_id_1', # atom name 1
                           'atom_id_2', # atom name 2
                           'value_order', # bond order
                           ]
        bond_table = block.find(bond_category, bond_attributes)
        bond_cols = {attr: bond_table.find_column(f"{bond_category}{attr}") for attr in bond_attributes}

        # Populate bond table
        bond_category = '_chem_comp_bond.'
        bond_attributes = ['atom_id_1', # atom name 1
                           'atom_id_2', # atom name 2
                           'value_order', # bond order
                           ]
        bond_table = block.find(bond_category, bond_attributes)
        bond_cols = {attr: bond_table.find_column(f"{bond_category}{attr}") for attr in bond_attributes}

        # Connect atoms by bonds
        bond_type_mapping = {
            'SING': Chem.BondType.SINGLE,
            'DOUB': Chem.BondType.DOUBLE,
            'TRIP': Chem.BondType.TRIPLE,
            'AROM': Chem.BondType.AROMATIC
        }
        bond_types = bond_cols['value_order']

        for bond_i, bond_type in enumerate(bond_types):
            rwmol.AddBond(name_to_idx_mapping[bond_cols['atom_id_1'][bond_i].strip('"')], 
                          name_to_idx_mapping[bond_cols['atom_id_2'][bond_i].strip('"')], 
                          bond_type_mapping.get(bond_type, Chem.BondType.UNSPECIFIED))

        # Try recharging mol (for metals)
        try:
            rwmol = recharge(rwmol)
        except Exception as e:
            logging.error(f"Failed to recharge rdkitmol. Error: {e} -> template for {resname} will be None. ")
            return None

        # Finish eidting mol 
        try:    
            rwmol.UpdatePropertyCache()
        except Exception as e:
            logging.error(f"Failed to create rdkitmol from cif. Error: {e} -> template for {resname} will be None. ")
            return None
        
        # Check implicit Hs
        total_implicit_hydrogens = sum(atom.GetNumImplicitHs() for atom in rwmol.GetAtoms())
        if total_implicit_hydrogens > 0:
            logging.error(f"rdkitmol from cif has implicit hydrogens. -> template for {resname} will be None. ")
            return None

        rdkit_mol = rwmol.GetMol()
            
        # Get Smiles with explicit Hs and ordered atom names
        smiles_exh, atom_name = get_smiles_with_atom_names(rdkit_mol)
        
        return cls(rdkit_mol, resname, smiles_exh, atom_name)


    def make_canonical(self, acidic_proton_loc):
        """Deprotonate acidic groups til the canonical (most deprotonated) state."""
        self.rdkit_mol = deprotonate(self.rdkit_mol, acidic_proton_loc = acidic_proton_loc)
        return self

    def make_embedded(self, allowed_smarts, leaving_names = None, leaving_smarts_loc = None):
        """Remove leaving atoms from the molecule by atom names and/or patterns."""
        self.rdkit_mol = embed(self.rdkit_mol, allowed_smarts = allowed_smarts, 
                               leaving_names = leaving_names, leaving_smarts_loc = leaving_smarts_loc)
        return self
        
    def make_capped(self, allowed_smarts, capping_names = None, capping_smarts_loc = None, protonate = None):
        """Build and name explicit hydrogens for atoms with implicit Hs by atom names and/or patterns."""
        self.rdkit_mol = cap(self.rdkit_mol, allowed_smarts = allowed_smarts, 
                             capping_names = capping_names, capping_smarts_loc = capping_smarts_loc,
                             protonate = protonate)
        return self
        
    def make_pretty_smiles(self):
        """Build and name explicit hydrogens for atoms with implicit Hs by atom names and/or patterns."""
        self.smiles_exh, self.atom_name = get_smiles_with_atom_names(self.rdkit_mol)
        self.smiles_exh = get_pretty_smiles(self.smiles_exh)
        return self

    def make_link_labels_from_patterns(self, pattern_to_label_mapping = {}):
        """Map patterns to link labels based on a given mapping."""
        if not pattern_to_label_mapping:
            return self

        for pattern in pattern_to_label_mapping:
            atom_idx = get_atom_idx_by_patterns(self.rdkit_mol, allowed_smarts = Chem.MolToSmarts(self.rdkit_mol), 
                                                wanted_smarts_loc = {pattern: {0}})
            atoms_in_mol = [atom for atom in self.rdkit_mol.GetAtoms()]
            if not atom_idx:
                logging.warning(f"Molecule doesn't contain pattern: {pattern} -> linker label for {pattern_to_label_mapping[pattern]} will not be made. ")
            elif len(atom_idx) > 1:
                logging.warning(f"Molecule contain multiple copies of pattern: {pattern} -> linker label for {pattern_to_label_mapping[pattern]} will not be made. ")
            else:
                atom_idx = next(iter(atom_idx))
                name = atoms_in_mol[atom_idx].GetProp('atom_id')
                self.link_labels.update({str(self.atom_name.index(name)): pattern_to_label_mapping[pattern]})

        return self
    
    def make_link_labels_from_names(self, name_to_label_mapping = {}):
        """Map atom names to link labels based on a given mapping."""
        if not name_to_label_mapping:
            return self

        for atom in self.rdkit_mol.GetAtoms():
            if atom.GetProp('atom_id') in name_to_label_mapping:
                if atom.GetNumImplicitHs() > 0:
                    name = atom.GetProp('atom_id')
                    self.link_labels.update({str(self.atom_name.index(name)): name_to_label_mapping[name]})

        return self
    
    def ResidueTemplate_check(self) -> bool:
        # ResidueTemplate.check from linked_rdkit_chorizo
        ps = Chem.SmilesParserParams()
        ps.removeHs = False
        mol = Chem.MolFromSmiles(self.smiles_exh, ps)
        have_implicit_hs = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetTotalNumHs() > 0}
        
        if self.link_labels: 
            int_labels = {int(atom_id) for atom_id in self.link_labels}
        else:
            int_labels = set()

        if int_labels != have_implicit_hs:
            raise ValueError(
                f"expected any atom with non-real Hs ({have_implicit_hs}) to tagged in link_labels ({int_labels})"
            )
        
        if not self.atom_name: 
            return
        
        if len(self.atom_name) != mol.GetNumAtoms():
            raise ValueError(f"{len(self.atom_name)=} differs from {mol.GetNumAtoms()=}")
        return


# Export/Writer Function
def export_chem_templates_to_json(cc_list: list[ChemicalComponent], json_fname: str=""):
    """Export list of chem templates to json"""

    basenames = [cc.parent for cc in cc_list]
    ambiguous_dict = {basename: [] for basename in basenames}
    for cc in cc_list:
        if cc.resname not in ambiguous_dict[cc.parent]:
            ambiguous_dict[cc.parent].append(cc.resname)

    data_to_export = {"ambiguous": {basename: basename+".ambiguous" for basename in basenames}}

    residue_templates = {}
    for cc in cc_list:
        residue_templates[cc.resname] = {
            "smiles": cc.smiles_exh,
            "atom_name": cc.resname+".atom_names",
        }
        if cc.link_labels:
            residue_templates[cc.resname]["link_labels"] = cc.resname+".link_labels"
        else:
            residue_templates[cc.resname]["link_labels"] = {}

    data_to_export.update({"residue_templates": residue_templates})

    json_str = json.dumps(data_to_export, indent = 4)

    # format ambiguous resnames to one line
    for basename in data_to_export["ambiguous"]:
        single_line_resnames = json.dumps(ambiguous_dict[basename], separators=(', ', ': '))
        json_str = json_str.replace(json.dumps(data_to_export["ambiguous"][basename], indent = 4), single_line_resnames)

    # format link_labels and atom_name to one line
    for cc in cc_list:
        single_line_atom_name = json.dumps(cc.atom_name, separators=(', ', ': '))
        json_str = json_str.replace(json.dumps(data_to_export["residue_templates"][cc.resname]["atom_name"], indent = 4), single_line_atom_name)
        if cc.link_labels:
            single_line_link_labels = json.dumps(cc.link_labels, separators=(', ', ': '))
            json_str = json_str.replace(json.dumps(data_to_export["residue_templates"][cc.resname]["link_labels"], indent = 4), single_line_link_labels)

    if json_fname:
        with open(pathlib.Path(json_fname), 'w') as f:
            f.write(json_str)
        print(f"{json_fname} <-- Json File for New Chemical Templates")
    else:
        print(" New Template Built ".center(60, "*"))
        print(json_str)
        print("*"*60)
        return json_str


def fetch_from_pdb(resname: str, max_retries = 5, backoff_factor = 2) -> str: 
    
    url = f"https://files.rcsb.org/ligands/download/{resname}.cif"
    file_path = f"{resname}.cif"
    for retry in range(max_retries):
        try:
            with urllib.request.urlopen(url) as response:
                content = response.read()
            logging.info(f"File downloaded successfully: {file_path}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".cif") as temp_file:
                temp_file.write(content)
            atexit.register(
                lambda: os.remove(temp_file.name) 
                if temp_file.name and os.path.exists(temp_file.name) 
                else None
            )
            return temp_file.name 
            
        except Exception as e:
            if retry < max_retries - 1: 
                wait_time = backoff_factor ** retry  
                logging.info(f"Download failed for {resname}. Error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                err = f"Max retries reached. Could not download CIF file for {resname}. Error: {e}"
                raise RuntimeError(err)

# Constants for deprotonate
acidic_proton_loc_canonical = {
        # any carboxylic acid, sulfuric/sulfonic acid/ester, phosphoric/phosphinic acid/ester
        '[H][O]['+atom+'](=O)': 0 for atom in ('CX3', 'SX4', 'SX3', 'PX4', 'PX3')
    } | {
        # any thio carboxylic/sulfuric acid
        '[H][O]['+atom+'](=S)': 0 for atom in ('CX3', 'SX4')
    } | {
        '[H][SX2][a]': 0, # thiophenol
    }

AA_embed_allowed_smarts = "[NX3]([H])([H])[CX4][CX3](=O)[O]"
NA_embed_allowed_smarts = "[O][PX4](=O)([O])[OX2][CX4][CX4]1[OX2][CX4][CX4][CX4]1[OX2][H]"

# Make free (noncovalent) CC
def build_noncovalent_CC(basename: str) -> ChemicalComponent: 

    with ChemicalComponent_LoggingControler(): 
        cc_from_cif = ChemicalComponent.from_cif(fetch_from_pdb(basename), basename)
        if cc_from_cif is None:
            return None

        cc = copy.deepcopy(cc_from_cif)
        logger.info(f"*** using CCD ligand {basename} to construct residue {cc.resname} ***")

        cc = cc.make_canonical(acidic_proton_loc = acidic_proton_loc_canonical)
        if len(rdmolops.GetMolFrags(cc.rdkit_mol))>1:
            err = f"Template Generation failed for {cc.resname}. Error: Molecule breaks into fragments during the deleterious editing. "
            raise RuntimeError(err)

        cc = cc.make_pretty_smiles()

        # Check
        try:
            cc.ResidueTemplate_check()
        except Exception as e:
            err = f"Template {cc.resname} Failed to pass ResidueTemplate check. Error: {e}"
            raise RuntimeError(err)
            
        logger.info(f"*** finish making {cc.resname} ***")
    return cc


def build_linked_CCs(basename: str, AA: bool = False, NA: bool = False, 
                     embed_allowed_smarts: str = None, 
                     cap_allowed_smarts: str = None, cap_protonate: bool = False, 
                     pattern_to_label_mapping_standard = dict[str, str], 
                     variant_dict = dict[str, tuple]) -> list[ChemicalComponent]: 

    with ChemicalComponent_LoggingControler(): 
        cc_from_cif = ChemicalComponent.from_cif(fetch_from_pdb(basename), basename)
        if cc_from_cif is None:
            return None

        if AA or cc_from_cif.rdkit_mol.GetSubstructMatch(Chem.MolFromSmarts(AA_embed_allowed_smarts)): 
            embed_allowed_smarts = AA_embed_allowed_smarts
            cap_allowed_smarts = "[NX3][CX4][CX3](=O)"
            cap_protonate = True
            pattern_to_label_mapping_standard = {'[NX3h1]': 'N-term', '[CX3h1]': 'C-term'}

            variant_dict = {
                    "_":  ({"[NX3]([H])([H])[CX4][CX3](=O)[O]": {1, 6}}, None), # embedded amino acid
                    "_N": ({"[NX3]([H])([H])[CX4][CX3](=O)[O]": {6}}, {"[NX3][CX4][CX3](=O)": {0}}), # N-term amino acid
                    "_C": ({"[NX3]([H])([H])[CX4][CX3](=O)[O]": {1}}, None), # C-term amino acid
                }
        elif NA or cc_from_cif.rdkit_mol.GetSubstructMatch(Chem.MolFromSmarts(NA_embed_allowed_smarts)): 
            embed_allowed_smarts = NA_embed_allowed_smarts
            cap_allowed_smarts = "[OX2][CX4][CX4]1[OX2][CX4][CX4][CX4]1[OX2]"
            cap_protonate = False
            pattern_to_label_mapping_standard = {'[PX4h1]': '5-prime', '[O+0X2h1]': '3-prime'}
            variant_dict = {
                    "_":  ({"[O][PX4](=O)([O])[OX2][CX4]": {0} ,"[CX4]1[OX2][CX4][CX4][CX4]1[OX2][H]": {6}}, None), # embedded nucleotide 
                    "_3": ({"[O][PX4](=O)([O])[OX2][CX4]": {0}}, None), # 3' end nucleotide 
                    "_5p": ({"[CX4]1[OX2][CX4][CX4][CX4]1[OX2][H]": {6}}, None), # 5' end nucleotide (extra phosphate than canonical X5)
                    "_5": ({"[O][PX4](=O)([O])[OX2][CX4]": {0,1,2,3}, "[CX4]1[OX2][CX4][CX4][CX4]1[OX2][H]": {6}}, {"[OX2][CX4][CX4]1[OX2][CX4][CX4][CX4]1[OX2]": {0}}), # 5' end nucleoside (canonical X5 in Amber)
                }

        editable = cc_from_cif.rdkit_mol.GetSubstructMatch(Chem.MolFromSmarts(embed_allowed_smarts))
        if not editable:
            logging.warning(f"Molecule doesn't contain embed_allowed_smarts: {embed_allowed_smarts} -> no templates will be made. ")
            return None

        cc_variants = []
        for suffix in variant_dict:
            cc = copy.deepcopy(cc_from_cif)
            cc.resname += suffix
            logging.info(f"*** using CCD residue {basename} to construct {cc.resname} ***")

            cc = (
                cc
                .make_canonical(acidic_proton_loc = acidic_proton_loc_canonical) 
                .make_embedded(allowed_smarts = embed_allowed_smarts, 
                            leaving_smarts_loc = variant_dict[suffix][0])
                )
            if len(rdmolops.GetMolFrags(cc.rdkit_mol))>1:
                logging.warning(f"Molecule breaks into fragments during the deleterious editing of {cc.resname} -> skipping the vaiant... ")
                continue

            cc = (
                cc
                .make_capped(allowed_smarts = cap_allowed_smarts, 
                            capping_smarts_loc = variant_dict[suffix][1],
                            protonate = cap_protonate) 
                .make_pretty_smiles()
                .make_link_labels_from_patterns(pattern_to_label_mapping = pattern_to_label_mapping_standard)
                )

            try:
                cc.ResidueTemplate_check()
            except Exception as e:
                err = f"Template {cc.resname} Failed to pass ResidueTemplate check. Error: {e}"
                logging.error(err)
                continue
            
            # Check redundancy
            if any(cc == other_variant for other_variant in cc_variants):
                logging.error(f"Template Failed to pass redundancy check -> skipping the template... ")
                continue

            cc_variants.append(cc)
            logging.info(f"*** finish making {cc.resname} ***")
    return cc_variants


# XXX read from prepared? enumerate in stepwise? all protonation state variants, alter charge and update smiles/idx
