import pathlib
import json
import logging
import traceback
from importlib.resources import files
from os import linesep as eol
from sys import exc_info
from typing import Union
from typing import Optional
import numpy as np

# @alphataubio refactoring [2025/03]
from rdkit import Chem
from rdkit.Chem import rdFMCS, rdChemReactions, rdMolInterchange
from rdkit.Geometry import Point3D

from .molsetup import RDKitMoleculeSetup
from .molsetup import MoleculeSetupEncoder
from .utils.jsonutils import rdkit_mol_from_json
from .utils.rdkitutils import mini_periodic_table
from .utils.rdkitutils import react_and_map
from .utils.rdkitutils import AtomField
from .utils.rdkitutils import build_one_rdkit_mol_per_altloc
from .utils.rdkitutils import _aux_altloc_mol_build
from .utils.rdkitutils import covalent_radius
from .utils.pdbutils import PDBAtomInfo
from .chemtempgen import export_chem_templates_to_json
from .chemtempgen import build_noncovalent_CC
from .chemtempgen import build_linked_CCs



logger = logging.getLogger(__name__)
rdkit_logger = logging.getLogger("rdkit")

residues_rotamers = {
    "SER": [("C", "CA", "CB", "OG")],
    "THR": [("C", "CA", "CB", "CG2")],
    "CYS": [("C", "CA", "CB", "SG")],
    "VAL": [("C", "CA", "CB", "CG1")],
    "HIS": [("C", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD2")],
    "ASN": [("C", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND2")],
    "ASP": [("C", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "ILE": [("C", "CA", "CB", "CG2"), ("CA", "CB", "CG2", "CD1")],
    "LEU": [("C", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "PHE": [("C", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD2")],
    "TYR": [("C", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD2")],
    "TRP": [("C", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD2")],
    "GLU": [
        ("C", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "OE1"),
    ],
    "GLN": [
        ("C", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "OE1"),
    ],
    "MET": [
        ("C", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "SD"),
        ("CB", "CG", "SD", "CE"),
    ],
    "ARG": [
        ("C", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "NE"),
        ("CG", "CD", "NE", "CZ"),
    ],
    "LYS": [
        ("C", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "CE"),
        ("CG", "CD", "CE", "NZ"),
    ],
}


class Monomer:
    """Individual subunit in a Polymer. Often called residue.

    Attributes
    ----------
    raw_rdkit_mol: RDKit Mol
        defines element and connectivity within a residue. Bond orders and
        formal charges may be incorrect, and hydrogens may be missing.
        This molecule may originate from a PDB string and it defines also
        the positions of the atoms.
    rdkit_mol: RDKit Mol
        Copy of the molecule from a ResidueTemplate, with positions from
        raw_rdkit_mol. All hydrogens are real atoms except for those
        at connections with adjacent residues.
    mapidx_to_raw: dict (int -> int)
        indices of atom in rdkit_mol to raw_rdkit_mol
    input_resname: str
        usually a three-letter code from a PDB
    template_key: str
        identifies instance of ResidueTemplate in ResidueChemTemplates
    atom_names: list (str)
        names of the atoms in the same order as rdkit_mol
    padded_mol: RDKit Mol
        molecule padded with ResiduePadder
    molsetup: RDKitMoleculeSetup
        An RDKitMoleculeSetup associated with this residue
    molsetup_mapidx: dict (int -> int)
        key: index of atom in padded_mol
        value: index of atom in rdkit_mol
    template: ResidueTemplate
        provides access to link_labels in the template
    """

    def __init__(
        self,
        raw_input_mol,
        rdkit_mol,
        mapidx_to_raw,
        input_resname=None,
        template_key=None,
        atom_names=None,
    ):

        self.raw_rdkit_mol = raw_input_mol
        self.rdkit_mol = rdkit_mol
        self.mapidx_to_raw = mapidx_to_raw
        self.residue_template_key = template_key  # same as pdb_resname except NALA, etc
        self.input_resname = input_resname  # exists even in openmm topology
        self.atom_names = (
            atom_names  # same order as atoms in rdkit_mol, used in rotamers
        )

        # TODO convert link indices/labels in template to rdkit_mol indices herein
        # self.link_labels = {}
        self.template = None

        if mapidx_to_raw is not None:
            self.mapidx_from_raw = {j: i for (i, j) in mapidx_to_raw.items()}
            if len(self.mapidx_from_raw) != len(self.mapidx_to_raw):
                raise RuntimeError(f"index mapping not invertable {mapidx_to_raw=}")
        else:
            self.mapidx_from_raw = None

        self.padded_mol = None
        self.molsetup = None
        self.molsetup_mapidx = None
        self.is_flexres_atom = None  # Check about these data types/Do we want the default to be None or empty
        self.is_movable = False

    def set_atom_names(self, atom_names_list):
        """

        Parameters
        ----------
        atom_names_list

        Returns
        -------

        """
        if self.rdkit_mol is None:
            raise RuntimeError("can't set atom_names if rdkit_mol is not set yet")
        if len(atom_names_list) != self.rdkit_mol.GetNumAtoms():
            raise ValueError(
                f"{len(atom_names_list)=} differs from {self.rdkit_mol.GetNumAtoms()=}"
            )
        name_types = set([type(name) for name in atom_names_list])
        if name_types != {str}:
            raise ValueError(f"atom names must be str but {name_types=}")
        self.atom_names = atom_names_list
        return

    def to_json(self):
        """

        Returns
        -------

        """
        return json.dumps(self, cls=MonomerEncoder)

    @classmethod
    def from_json(cls, json_string):
        """

        Parameters
        ----------
        json_string

        Returns
        -------

        """
        monomer = json.loads(json_string, object_hook=cls.monomer_json_decoder)
        return monomer

    def parameterize(self, mk_prep, residue_id):

        molsetups = mk_prep(self.padded_mol)
        if len(molsetups) != 1:
            raise NotImplementedError(f"need 1 molsetup but got {len(molsetups)}")
        molsetup = molsetups[0]
        self.molsetup = molsetup
        self.is_flexres_atom = [False for _ in molsetup.atoms]

        # set ignore to True for atoms that are padding
        for atom in molsetup.atoms:
            if atom.index not in self.molsetup_mapidx:
                atom.is_ignore = True

        # recalculate flexibility tree after setting ignored atoms
        mk_prep.calc_flex(molsetup)

        # rectify charges to sum to integer (because of padding)
        if mk_prep.charge_model == "zero":
            net_charge = 0
        else:
            rdkit_mol = self.rdkit_mol
            net_charge = sum(
                [atom.GetFormalCharge() for atom in rdkit_mol.GetAtoms()]
            )
        not_ignored_idxs = []
        charges = []
        for atom in molsetup.atoms:
            if atom.index in self.molsetup_mapidx: # TODO offsite not in mapidx
                charges.append(atom.charge)
                not_ignored_idxs.append(atom.index)
        charges = rectify_charges(charges, net_charge, decimals=3)
        for i, j in enumerate(not_ignored_idxs):
            molsetup.atoms[j].charge = charges[i]
        self._set_pdbinfo(residue_id)
        return

    def _set_pdbinfo(self, residue_id):
        not_ignored_idxs = []
        for atom in self.molsetup.atoms:
            if atom.index in self.molsetup_mapidx: # TODO offsite not in mapidx
                not_ignored_idxs.append(atom.index)
        chain, resnum = residue_id.split(":")
        if resnum[-1].isalpha():
            icode = resnum[-1]
            resnum = resnum[:-1]
        else:
            icode = ""
        if self.atom_names is None:
            atom_names = ["" for _ in not_ignored_idxs]
        else:
            atom_names = self.atom_names
        for i, j in enumerate(not_ignored_idxs):
            atom_name = atom_names[self.molsetup_mapidx[j]]
            self.molsetup.atoms[j].pdbinfo = PDBAtomInfo(
                atom_name, self.input_resname, int(resnum), icode, chain
            )
        return


class NoAtomMapWarning(logging.Filter):
    def filter(self, record):
        fields = record.getMessage().split()
        a = " ".join(fields[1:4]) == "product atom-mapping number"
        b = " ".join(fields[5:]) == "not found in reactants."
        is_atom_map_warning = a and b
        return not is_atom_map_warning

class ResiduePadder:
    """
    A class for padding RDKit molecules of residues with parts from adjacent residues.

    Attributes
    ----------
    rxn : rdChemReactions.ChemicalReaction
        Reaction SMARTS of a single-reactant, single-product reaction for padding.
    adjacent_smartsmol : Chem.Mol
        SMARTS molecule with mapping numbers to copy atom positions from part of adjacent residue.
    adjacent_smartsmol_mapidx : list
        Mapping for atoms in adjacent_smartsmol, from mapping numbers to atom indicies. 
    """

    # Replacing ResidueConnection by ResiduePadding
    # Why have two ResiduePadding instances per connection between two-residues?
    #  - three-way merge: if three carbons joined in cyclopropare, we can still pad
    #  - defines padding in the reaction for blunt residues
    #  - all bonds will be defined in the input topology after a future refactor

    # reaction should not delete atoms, not even Hs
    # reaction should create bonds at non-real Hs (implicit or explicit rdktt H)

    def __init__(self, rxn_smarts: str, adjacent_res_smarts: str = None, auto_blunt:bool=False): 
        """
        Initialize the ResiduePadder with reaction SMARTS and optional adjacent residue SMARTS.

        Parameters
        ----------
        rxn_smarts: str
            Reaction SMARTS to pad a link atom of a Monomer molecule.
            Product atoms that are not mapped in the reactants will have
            their coordinates set from an adjacent residue molecule, given
            that adjacent_res_smarts is provided and the atom labels match
            the unmapped product atoms of rxn_smarts.
        adjacent_res_smarts: str
            SMARTS pattern to identify atoms in molecule of adjacent residue
            and copy their positions to padding atoms. The SMARTS atom labels
            must match those of the product atoms of rxn_smarts that are
            unmapped in the reagents.
        auto_blunt: bool
            missing bonds of Monomers will automatically be blunt if
            this parameter is true, and raise an error otherwise
        """

        # Ensure rxn_smarts has single reactant and single product
        self.rxn = self._validate_rxn_smarts(rxn_smarts)
        self.auto_blunt = auto_blunt

        # Fill in adjacent_smartsmol_mapidx
        if adjacent_res_smarts is None:
            self.adjacent_smartsmol = None
            self.adjacent_smartsmol_mapidx = None
            return

        # Ensure adjacent_res_smarts is None or a valid SMARTS        
        self.adjacent_smartsmol = self._initialize_adj_smartsmol(adjacent_res_smarts)

        # Ensure the mapping numbers are the same in adjacent_smartsmol and rxn_smarts's product
        self._check_adj_smarts(self.rxn, self.adjacent_smartsmol)

        self.adjacent_smartsmol_mapidx = {
            atom.GetIntProp("molAtomMapNumber"): atom.GetIdx()
            for atom in self.adjacent_smartsmol.GetAtoms()
            if atom.HasProp("molAtomMapNumber")
        }
        return
    
    @staticmethod
    def _validate_rxn_smarts(rxn_smarts: str) -> rdChemReactions.ChemicalReaction:
        """Validate rxn_smarts and return rxn"""
        rxn = rdChemReactions.ReactionFromSmarts(rxn_smarts)
        if rxn.GetNumReactantTemplates() != 1:
            raise ValueError(f"Expected 1 reactant, got {rxn.GetNumReactantTemplates()}.")
        if rxn.GetNumProductTemplates() != 1:
            raise ValueError(f"Expected 1 product, got {rxn.GetNumProductTemplates()}.")
        return rxn
    
    @staticmethod
    def _initialize_adj_smartsmol(adjacent_res_smarts: str) -> Chem.Mol:
        """Validate adjacent_res_smarts and return adjacent_smartsmol"""
        adjacent_smartsmol = Chem.MolFromSmarts(adjacent_res_smarts)
        if adjacent_smartsmol is None:
            raise RuntimeError("Invalid SMARTS pattern in adjacent_res_smarts")
        return adjacent_smartsmol
    
    @staticmethod
    def _check_adj_smarts(rxn: rdChemReactions.ChemicalReaction, adjacent_smartsmol: Chem.Mol):
        """
        Ensure the atom mapping numbers are the same in adjacent_smartsmol and rxn_smarts's product
        """

        # Assumes single reactant, single product
        reactant_ids = get_molAtomMapNumbers(rxn.GetReactantTemplate(0))
        product_ids = get_molAtomMapNumbers(rxn.GetProductTemplate(0))
        adjacent_ids = get_molAtomMapNumbers(adjacent_smartsmol)
        padding_ids = product_ids.difference(reactant_ids)
        is_ok = padding_ids == adjacent_ids

        if not is_ok:
            raise ValueError(f"SMARTS labels in adjacent_smartsmol ({adjacent_ids}) differ from \
                             unmapped product labels in reaction ({padding_ids})")

    def __call__(self, target_mol: Chem.Mol, adjacent_mol = None, 
                 target_required_atom_index = None, adjacent_required_atom_index = None):
        # add Hs only to padding atoms
        # copy coordinates if adjacent res has Hs bound to heavy atoms
        # labels have been checked upstream

        # Ensure target_mol contains self.rxn's reactant
        rxn = self.rxn
        if not self._check_target_mol(target_mol):
            print(f"target_mol ({Chem.MolToSmiles(target_mol)}) is not fully compliant with the template rxn ({rdChemReactions.ReactionToSmarts(self.rxn)})...")
            # Assumes single reactant and single product
            reactant_smartsmol = rxn.GetReactantTemplate(0)
            reactant_ids = get_molAtomMapNumbers(reactant_smartsmol)

            # Generate fallback options for reactants
            fallback_reactant_smartsmol = Chem.MolFromSmarts(rdFMCS.FindMCS([reactant_smartsmol, target_mol]).smartsString)
            if fallback_reactant_smartsmol is None:
                raise RuntimeError(f"There is no common substructure between target_mol and the expected reactant. ")

            # Add mapping number to fallback reactants and filter the fallback options
            # To be accepted, the fallback reactant needs to at least have a match with target_mol
            # containing target_mol's atom with target_required_atom_index
            fallback_reactants = [
                reactant_mol for reactant_mol in apply_atom_mappings(fallback_reactant_smartsmol, reactant_smartsmol)
                if any(target_required_atom_index in match for match in target_mol.GetSubstructMatches(reactant_mol))
            ]
            if len(fallback_reactants) == 0:
                raise RuntimeError(f"The maximum common substructure between target_mol and the expected reactant does not contain the expected linker atom with target_required_atom_index.")
            
            # Take any fallback reactant; actually, they're the same reactant mols having different mapping numbers
            fallback_reactant = fallback_reactants[0]
            
            # Modify rxn smarts and update rxn
            fallback_reactant_ids = get_molAtomMapNumbers(fallback_reactant)
            skipping_ids = reactant_ids.difference(fallback_reactant_ids)
            fallback_product = remove_atoms_with_mapping(rxn.GetProductTemplate(0), skipping_ids)
            fallback_rxnsmarts = f"{Chem.MolToSmarts(fallback_reactant)}>>{Chem.MolToSmarts(fallback_product)}"
            rxn = rdChemReactions.ReactionFromSmarts(fallback_rxnsmarts)
            print(f"Switched from Template rxn ({rdChemReactions.ReactionToSmarts(self.rxn)}) to Fallback rxn ({fallback_rxnsmarts})")
        
        # Get adjacent_mol's reacting part that contains adjacent_required_atom_index
        if adjacent_mol is not None:

            # Ensure adjacent_mol contains expected_adjacent_smartsmol, and 
            # there's exactly one match that includes atom with adjacent_required_atom_index
            if self._check_adjacent_mol(self.adjacent_smartsmol, adjacent_mol, adjacent_required_atom_index):
                adjacent_smartsmol = self.adjacent_smartsmol
            
            # Remove unmapped atoms from Template adjacent mol SMARTS as the fallback option;
            # The unmapped atoms aren't needed for positions anyways
            else:
                print(f"adjacent_mol ({Chem.MolToSmiles(adjacent_mol)}) is not fully compliant with the template adjacent_smarts ({Chem.MolToSmarts(self.adjacent_smartsmol)})...")
                adjacent_smartsmol = remove_unmapped_atoms_from_mol(self.adjacent_smartsmol)

                # Evaluate adjacent mol against the fallback adjacent mol SMARTS
                if self._check_adjacent_mol(adjacent_smartsmol, adjacent_mol, adjacent_required_atom_index):
                     print(f"Switched from Template adjacent mol ({Chem.MolToSmarts(self.adjacent_smartsmol)}) to Fallback adjacent mol ({Chem.MolToSmarts(adjacent_smartsmol)})")
                else:
                    raise RuntimeError(f"adjacent_mol doesn't contain the mapped atoms in adjacent_smartsmol.") 
            
            # Update hit and adjacent_smartsmol_mapidx 
            hit = adjacent_mol.GetSubstructMatches(adjacent_smartsmol)[0]
            adjacent_smartsmol_mapidx = {
                atom.GetIntProp("molAtomMapNumber"): atom.GetIdx()
                for atom in adjacent_smartsmol.GetAtoms() if atom.HasProp("molAtomMapNumber")
                }

        # suppress rdkit warning about product atom map not found in reactants
        # e.g. in "[C:1]>>[C:1][O:2]" label :2 is missing in reactants
        filtr = NoAtomMapWarning()
        rdkit_logger.addFilter(filtr)
        
        # Get padded mol and index map from the rxn
        outcomes = react_and_map((target_mol,), rxn)
        rdkit_logger.removeFilter(filtr)

        # Filter outcomes by target_required_atom_index
        if target_required_atom_index is not None:
            outcomes = [
                (product, index_map)
                for (product, index_map) in outcomes 
                if target_required_atom_index in index_map["atom_idx"] 
            ]

        # Ensure single outcome
        if len(outcomes) == 0:
            raise RuntimeError(f"The padding reaction of target_mol has no outcome that contains the atom with target_required_atom_index")
        elif len(outcomes) > 1:
            raise RuntimeError(f"The padding reaction of target_mol has multiple outcomes that contain the atom with target_required_atom_index")
        padded_mol, idxmap = outcomes[0]

        padding_heavy_atoms = [
            i for i, j in enumerate(idxmap["atom_idx"])
            if j is None and padded_mol.GetAtomWithIdx(i).GetAtomicNum() != 1
        ]
        mapidx = idxmap["atom_idx"]

        # Add Hs to padded_mol and update mapidx
        if adjacent_mol is None:
            padded_mol.UpdatePropertyCache()  # avoids getNumImplicitHs() called without preceding call to calcImplicitValence()
            Chem.SanitizeMol(padded_mol)  # just in case
            padded_h = Chem.AddHs(padded_mol, onlyOnAtoms=padding_heavy_atoms)
            mapidx += [None] * (padded_h.GetNumAtoms() - padded_mol.GetNumAtoms())
        else:
            # Get coordinates of existing atoms
            adjacent_coords = adjacent_mol.GetConformer().GetPositions()
            for atom in adjacent_smartsmol.GetAtoms():
                if not atom.HasProp("molAtomMapNumber"):
                    continue
                j = atom.GetIntProp("molAtomMapNumber")
                k = idxmap["new_atom_label"].index(j)
                l = adjacent_smartsmol_mapidx[j]
                padded_mol.GetConformer().SetAtomPosition(k, adjacent_coords[hit[l]])
            padded_mol.UpdatePropertyCache()  # avoids getNumImplicitHs() called without preceding call to calcImplicitValence()
            Chem.SanitizeMol(padded_mol)  # got crooked Hs without this
            padded_h = Chem.AddHs(
                padded_mol, onlyOnAtoms=padding_heavy_atoms, addCoords=True
            )

        return padded_h, mapidx
    
    @staticmethod
    def _check_adjacent_mol(expected_adjacent_smartsmol: Chem.Mol, adjacent_mol: Chem.Mol, adjacent_required_atom_index: str):
        """
        Ensure adjacent_mol contains expected_adjacent_smartsmol, and 
        there's exactly one match that includes atom with adjacent_required_atom_index
        """
        if expected_adjacent_smartsmol is None:
            raise RuntimeError("adjacent_res_smarts must be initialized to support adjacent_mol.")

        hits = adjacent_mol.GetSubstructMatches(expected_adjacent_smartsmol)
        if adjacent_required_atom_index is not None:
            hits = [hit for hit in hits if adjacent_required_atom_index in hit]
            if len(hits) > 1:
                raise RuntimeError(f"adjacent_mol has multiple matches for adjacent_smartsmol.")  
            elif len(hits) == 0:
                return False
        return True

    def _check_target_mol(self, target_mol: Chem.Mol):
        """Ensure target_mol contains self.rxn's reactant"""
        # Assumes single reactant
        if target_mol.GetSubstructMatches(self.rxn.GetReactantTemplate(0)):
            return True
        else:
            return False

    @classmethod
    def from_json(cls, string):
        d = json.loads(string)
        return cls(**d)
    
    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__)

# Utility Functions

def get_molAtomMapNumbers(mol: Chem.Mol) -> set[int]:
    """Return the set of mapping numbers in a molecule."""
    return {atom.GetIntProp("molAtomMapNumber") for atom in mol.GetAtoms() if atom.HasProp("molAtomMapNumber")}

def remove_unmapped_atoms_from_mol(mol: Chem.Mol) -> Chem.Mol:
    """Remove atoms without mapping numbers from a molecule."""
    atoms_to_remove = [
        atom.GetIdx() for atom in mol.GetAtoms() 
        if not atom.HasProp("molAtomMapNumber")
        ]

    if len(atoms_to_remove) > 0:
        mol = Chem.RWMol(mol)
        for idx in sorted(atoms_to_remove, reverse=True):
            mol.RemoveAtom(idx)
        mol = mol.GetMol()

    return mol

def apply_atom_mappings(mcs_mol: Chem.Mol, original_mol: Chem.Mol) -> list[Chem.Mol]:
    """
    Apply atom mappings from the original molecule to the MCS molecule by substructure match.
    Be prepared for multiple matches, return a list for further evaluation
    """

    # Assumes original_mol contains mcs_mol
    matches = original_mol.GetSubstructMatches(mcs_mol)
    mapped_mcs_molecules = []

    for match in matches:
        rw_mcs_mol = Chem.RWMol(mcs_mol)
        
        for i, mcs_atom in enumerate(rw_mcs_mol.GetAtoms()):
            original_atom_idx = match[i]
            original_atom = original_mol.GetAtomWithIdx(original_atom_idx)
            
            if original_atom.HasProp("molAtomMapNumber"):
                mcs_atom.SetProp("molAtomMapNumber", original_atom.GetProp("molAtomMapNumber"))

        mapped_mcs_molecules.append(rw_mcs_mol.GetMol())
    
    return mapped_mcs_molecules

def remove_atoms_with_mapping(product: Chem.Mol, mapping_numbers: set) -> Chem.Mol:
    """Remove atoms with specific atom mapping numbers from a molecule."""
    editable_product = Chem.RWMol(product)

    atoms_to_remove = [
        atom.GetIdx() 
        for atom in editable_product.GetAtoms() 
        if atom.HasProp("molAtomMapNumber") and int(atom.GetProp("molAtomMapNumber")) in mapping_numbers
    ]
    for idx in sorted(atoms_to_remove, reverse=True):
        editable_product.RemoveAtom(idx)
    
    return editable_product.GetMol()


class ResidueTemplate:
    """
    Data and methods to pad rdkit molecules of polymer residues with parts of adjacent residues.

    Attributes
    ----------
    mol: RDKit Mol
        molecule with the exact atoms that constitute the system.
        All Hs are explicit, but atoms bonded to adjacent residues miss an H.
    link_labels: dict (int -> string)
        Keys are indices of atoms that need padding
        Values are strings to identify instances of ResiduePadder
    atom_names: list (string)
        list of atom names, matching order of atoms in rdkit mol
    """

    def __init__(self, smiles, link_labels=None, atom_names=None):
        ps = Chem.SmilesParserParams()
        ps.removeHs = False
        mol = Chem.MolFromSmiles(smiles, ps)
        self.check(mol, link_labels, atom_names)
        self.mol = mol
        self.link_labels = link_labels
        self.atom_names = atom_names
        return

    def check(self, mol, link_labels, atom_names):
        have_implicit_hs = set()
        for atom in mol.GetAtoms():
            if atom.GetTotalNumHs() > 0:
                have_implicit_hs.add(atom.GetIdx())
        if link_labels is not None and set(link_labels) != have_implicit_hs:
            raise ValueError(
                f"expected any atom with non-real Hs ({have_implicit_hs}) to be in {link_labels=}"
            )
        if atom_names is None:
            return
        # data_lengths = set([len(values) for (_, values) in data.items()])
        # if len(data_lengths) != 1:
        #    raise ValueError(f"each array in data must have the same length, but got {data_lengths=}")
        # data_length = data_lengths.pop()
        if len(atom_names) != mol.GetNumAtoms():
            raise ValueError(f"{len(atom_names)=} differs from {mol.GetNumAtoms()=}")
        return

    def match(self, input_mol):
        mapping = mapping_by_mcs(self.mol, input_mol)
        mapping_inv = {value: key for (key, value) in mapping.items()}
        if len(mapping_inv) != len(mapping):
            raise RuntimeError(
                f"bug in atom indices, repeated value different keys? {mapping=}"
            )
        # atoms "missing" exist in self.mol but not in input_mol
        # "excess" atoms exist in input_mol but not in self.mol
        result = {
            "H": {"found": 0, "missing": 0, "excess": []},
            "heavy": {"found": 0, "missing": 0, "excess": 0},
        }
        for atom in self.mol.GetAtoms():
            element = "H" if atom.GetAtomicNum() == 1 else "heavy"
            key = "found" if atom.GetIdx() in mapping else "missing"
            result[element][key] += 1
        for atom in input_mol.GetAtoms():
            element = "H" if atom.GetAtomicNum() == 1 else "heavy"
            if atom.GetIdx() not in mapping_inv:
                if element == "H": 
                    nei_idx = atom.GetNeighbors()[0].GetIdx()
                    if nei_idx in mapping_inv: 
                        result[element]["excess"].append(mapping_inv[nei_idx])
                    else:
                        result[element]["excess"].append(-1)
                else:
                    result[element]["excess"] += 1
        return result, mapping

def rdkit_or_none_to_json(rdkit_mol):
    if rdkit_mol is None:
        return None
    return rdMolInterchange.MolToJSON(rdkit_mol)



def add_rotamers_to_polymer_molsetups(rotamer_states_list, polymer):
    """

    Parameters
    ----------
    rotamer_states_list
    polymer

    Returns
    -------

    """

    # FIXME: is add_rotamers_to_polymer_molsetups() orphan code ?
    # not called anywhere ?? [@alphataubio 2025/03]
    pass

"""
    rotamer_res_disambiguate = {}
    for (
        primary_res,
        specific_res_list,
    ) in polymer.residue_chem_templates.ambiguous.items():
        for specific_res in specific_res_list:
            rotamer_res_disambiguate[specific_res] = primary_res

    no_resname_to_resname = {}
    for res_with_resname in polymer.monomers:
        chain, resname, resnum = res_with_resname.split(":")
        no_resname_key = f"{chain}:{resnum}"
        if no_resname_key in no_resname_to_resname:
            errmsg = "both %s and %s would be keyed by %s" % (
                res_with_resname,
                no_resname_to_resname[no_resname_key],
                no_resname_key,
            )
            raise RuntimeError(errmsg)
        no_resname_to_resname[no_resname_key] = res_with_resname

    state_indices_list = []
    for state_index, state_dict in enumerate(rotamer_states_list):
        print(f"adding rotamer state {state_index + 1}")
        state_indices = {}
        for res_no_resname, angles in state_dict.items():
            res_with_resname = no_resname_to_resname[res_no_resname]
            if polymer.monomers[res_with_resname].molsetup is None:
                raise RuntimeError(
                    "no molsetup for %s, can't add rotamers" % (res_with_resname)
                )
            # next block is inefficient for large rotamer_states_list
            # refactored polymers could help by having the following
            # data readily available
            molsetup = polymer.monomers[res_with_resname].molsetup
            name_to_molsetup_idx = {}
            for atom in molsetup.atoms:
                atom_name = atom.pdbinfo.name
                name_to_molsetup_idx[atom_name] = atom.index

            resname = res_with_resname.split(":")[1]
            resname = rotamer_res_disambiguate.get(resname, resname)

            atom_names = residues_rotamers[resname]
            if len(atom_names) != len(angles):
                raise RuntimeError(
                    f"expected {len(atom_names)} angles for {resname}, got {len(angles)}"
                )

            atom_idxs = []
            for names in atom_names:
                tmp = [name_to_molsetup_idx[name] for name in names]
                atom_idxs.append(tmp)

            state_indices[res_with_resname] = len(molsetup.rotamers)
            molsetup.add_rotamer(atom_idxs, np.radians(angles))

        state_indices_list.append(state_indices)

    return state_indices_list
"""
