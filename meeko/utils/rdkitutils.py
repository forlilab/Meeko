from rdkit import Chem
from rdkit.Chem import rdChemReactions
from .utils import mini_periodic_table
from .pdbutils import PDBAtomInfo


"""
create new RDKIT residue

mi  =  Chem.AtomPDBResidueInfo()
mi.SetResidueName('MOL')
mi.SetResidueNumber(1)
mi.SetOccupancy(0.0)
mi.SetTempFactor(0.0)

source: https://sourceforge.net/p/rdkit/mailman/message/36404394/
"""


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
        The dictionary has keys 'atom_idx' and 'new_atom_label', which are lists:
        - 'atom_idx' maps the product atoms to their corresponding reactant atom indices (if applicable).
        - 'new_atom_label' maps the product atoms to their reaction mapping numbers for newly added atoms.
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
