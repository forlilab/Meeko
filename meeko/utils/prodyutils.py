from typing import Union
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.rdchem import Mol
from rdkit.Geometry import Point3D

from prody.atomic import ATOMIC_FIELDS
from prody.atomic.atomgroup import AtomGroup
from prody.atomic.selection import Selection
from prody.atomic.segment import Segment
from prody.atomic.chain import Chain
from prody.atomic.residue import Residue

# used to convert from and to element symbols ans atomic number
periodic_table = Chem.GetPeriodicTable()

# The following are not used yet...
# https://chemicbook.com/2021/03/01/how-to-show-atom-numbers-in-rdkit-molecule.html
_bondtypes = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    6: Chem.BondType.AROMATIC,
}
_bondstereo = {
    0: Chem.rdchem.BondStereo.STEREONONE,
    1: Chem.rdchem.BondStereo.STEREOE,
    2: Chem.rdchem.BondStereo.STEREOZ,
}
prody_atom_properties = {"sec_structure": "getSecstr"}
names = Chem.rdchem.AtomMonomerType.names
###


# TODO WORK IN PROGRESS
# NOT WORKING
def rdkit_to_prody(
    rdkit_mol: Mol,
    name: str = "",
    conformer_idx: int = 0,
    default_chain: str = "R",
    default_resname: str = "UNK",
):
    """
    THIS FUNCTION IS INCOMPLETE
    """
    if name == "":
        name = rdkit_mol.GetProp("_Name")
    if name == "" or name is None:
        name = "rdkit_to_prody_molecule"
    prody_obj = AtomGroup(name)
    idx_prody_to_rdkit = {}
    idx_rdkit_to_prody = {}
    idx_rdkit = 0
    atom_count = rdkit_mol.GetNumAtoms()
    # get coordinates
    conformer = rdkit_mol.GetConformer(conformer_idx)
    coord_array = conformer.GetPositions()

    # see python4.9/site-packages/prody/compounds/pdbligands.py]
    chain_array = np.zeros(atom_count, dtype=ATOMIC_FIELDS["chain"].dtype)
    resname_array = np.zeros(atom_count, dtype=ATOMIC_FIELDS["resname"].dtype)
    resnum_array = np.zeros(atom_count, dtype=ATOMIC_FIELDS["resnum"].dtype)
    aname_array = np.zeros(atom_count, dtype=ATOMIC_FIELDS["name"].dtype)
    atype_array = np.zeros(atom_count, dtype=ATOMIC_FIELDS["type"].dtype)
    # properties
    hetero_array = np.zeros(atom_count, dtype=bool)
    serial_array = np.zeros(atom_count, dtype=ATOMIC_FIELDS["serial"].dtype)
    segment_array = np.zeros(atom_count, dtype=ATOMIC_FIELDS["segment"].dtype)
    element_array = np.zeros(atom_count, dtype=ATOMIC_FIELDS["element"].dtype)

    # iterate on atoms
    # see issue https://github.com/rdkit/rdkit/issues/6208
    # to be fixed in 2023.09 version of RDKit.
    for i in range(0, rdkit_mol.GetNumAtoms()):
        atom = rdkit_mol.GetAtomWithIdx(i)
        atomic_num = atom.GetAtomicNumber()
        element_array[i] = periodic_table.GetElement(atomic_num)
        if hasattr(rdkit_mol, "prody_to_rdkit"):
            serial_array[i] = rdkit_mol.prody_to_rdkit[i]
        else:
            serial_array[i] = i + 1

        # gather residue info
        chain_array
        resname_array
        resnum_array
        aname_array
        segment_array

        # if non_std residue, thebmn set flag hetero
        hetero_array

    # charges = np.zeros(asize, dtype=ATOMIC_FIELDS["charge"].dtype)
    # radii = np.zeros(asize, dtype=ATOMIC_FIELDS["radius"].dtype)

    for bond in rdkit_mol.GetBonds():
        pass

    # https://sourceforge.net/p/rdkit/mailman/rdkit-discuss/thread/60825b0f0804152055x4ba39e13k7d62f81634413ca6@mail.gmail.com/


ALLOWED_PRODY_TYPES = Union[Selection, AtomGroup, Chain, Residue]


def prody_to_rdkit(
    prody_obj: ALLOWED_PRODY_TYPES,
    name: str = "molecule",
    sanitize: bool = True,
    keep_bonds: bool = False,
    keep_charges: bool = False,
) -> Mol:
    """
    Convert a ProDy selection or atom group into an RDKit molecule.

    The function accepts both Selections or AtomGroups and generates the
    correspodning RDKit object, retaining most of the information available
    from the ProDy object into RDKit atoms using the AtomPDBResidueInfo class.

    By default, only coordinates, atom element and name,
    and residue info are retained. Bond and charge info,
    even if available are ignored.

    Two convenience lookup tables are attached to the new RDKit molecule to use
    ProDy indices to retrieve the corresponding RDKit indices
    (`mol._idx_prody_to_rdkit`) and vice versa (`mol._idx_rdkit_to_prody`), e.g.:

    >>> tyr_oh = prody_mol.select("resname TYR and name OH and resnum 108 and chid A")
    <Selection: 'resname TYR and... 108 and chid A' from 1jff (1 atoms)>
    >>> idx = tyr_oh.getIndices()[0]
    >>> rdkit_mol._idx_prody_to_rdkit[idx]
    11
    >>> original_res_atom_idx = rdkit_mol._idx_rdkit_to_prody[11]
    640
    >>> rdkit_mol._idx_rdkit_to_prody
    {0: 629, 1: 630, 2: 631, 3: 632, 4: 633, 5: 634, 6: 635, 7: 636, 8: 637, 9:
    638, 10: 6 39, 11: 640}

    If the ProDy object contains a charges atom field, and the flag
    `keep_charges` is set to True, charges will be extracted
    (`atom.GetPartialCharge()`) and saved as an RDKit Property
    ("_prody_charges") for each atom.

    By default, even if available, bonds defined in the ProDy object are
    ignored. In order to use ProDy bond info, the flag `keep_charges` needs to
    be set to True and bonds assigned or perceived prior to the conversion
    (e.g. `AtomGroup.inferBonds()`, which is usually slow).

    Parameters
    ----------
    prody_obj : Union[Selection, AtomGroup]
        input molecule (AtomGroup) or subset selection (e.g., Selection
        containign one residue)
    name : str (default: "molecule" )
        name assigned to the new molecule
    sanitize : bool (default: True)
        perform the sanitization of the RDKit molecule prior to returning it;
        any exception during this process will be raised as RDKit would do.
    keep_bonds : bool (default: False)
        use bonds defined in the ProDy object, instead of perceiving them with
        RDKit using rdDetermineBonds.DetermineConnectivity()
    keep_charges : bool (default: False)
        use partial charges defined in the ProDy object; when True, partial
        charges are stored as  "_prody_charges" property in the RDKit molecule

    Returns
    -------
    Mol
        RDKit molecule
    """

    atom_count = len(prody_obj)
    # create rdkit data
    rdmol = Chem.Mol()
    mol = Chem.EditableMol(rdmol)
    conformer = Chem.Conformer(atom_count)
    # initialize boookkeeping objects
    idx_prody_to_rdkit = {}
    idx_rdkit_to_prody = {}
    idx_rdkit = 0

    # start molecule editing
    mol.BeginBatchEdit()
    for atom in prody_obj:
        # gather atom info
        idx_prody = int(atom.getIndex())
        element = atom.getData("element")
        # f8x PDB quirks like Zn/ZN
        if len(element) > 1:
            element = f"{element[0]}{element[1].lower()}"
        atomic_num = periodic_table.GetAtomicNumber(element)
        rdkit_atom = Chem.Atom(atomic_num)
        # TODO check which property to use
        if keep_charges is True:
            partial_charge = atom.GetPartialCharge()
            rdkit_atom.SetProp("_prody_charges", partial_charge)
        x, y, z = atom.getCoords()
        conformer.SetAtomPosition(idx_rdkit, Point3D(x, y, z))
        res_info = Chem.rdchem.AtomPDBResidueInfo()
        res_info.SetResidueName(str(atom.getResname()))
        res_info.SetName(str(atom.getName()))
        res_info.SetResidueNumber(int(atom.getResnum()))
        res_info.SetChainId(str(atom.getChid()))
        res_info.SetInsertionCode("")
        res_info.SetTempFactor(0.0)
        res_info.SetIsHeteroAtom(False)
        res_info.SetSecondaryStructure(0)
        res_info.SetSegmentNumber(0)
        rdkit_atom.SetPDBResidueInfo(res_info)
        mol.AddAtom(rdkit_atom)
        idx_prody_to_rdkit[idx_prody] = idx_rdkit
        idx_rdkit_to_prody[idx_rdkit] = idx_prody
        idx_rdkit += 1
    # iterate and create bonds
    if keep_bonds is True:
        if isinstance(prody_obj, AtomGroup):
            bonds = prody_obj.getBonds()
        else:
            bonds = prody_obj.getAtomGroup().getBonds()
    else:
        bonds = None
    if bonds is not None:
        for bond in bonds:
            bond_indices = [int(x) for x in bond.getIndices()]
            # bond with atoms outside this selection
            if (
                not bond_indices[0] in idx_rdkit_to_prody
                or not bond_indices[1] in idx_rdkit_to_prody
            ):
                continue
            # bond_type = bond_types[bond[-1]]
            bond_order = 1
            bond_type = _bondtypes[bond_order]
            print("bonds", bond_indices, bond)
            mol.AddBond(bond_indices[0], bond_indices[1], bond_type)
    # finalize molecule changes
    mol.CommitBatchEdit()
    rdmol = mol.GetMol()
    # add coordinates
    rdmol.AddConformer(conformer, assignId=True)
    # the molecule needs bonds, one way or another
    if bonds is None:
        rdDetermineBonds.DetermineConnectivity(rdmol)
    # sanitize the molecule if necessary
    # TODO before or after ond perception?
    if sanitize:
        Chem.SanitizeMol(rdmol)
    # attach the bookkeeping to the molecule
    rdmol._idx_prody_to_rdkit = idx_prody_to_rdkit
    rdmol._idx_rdkit_to_prody = idx_rdkit_to_prody
    return rdmol


if __name__ == "__main__":
    import sys
    import prody

    # # test rdkit
    # mol = Chem.SDMolSupplier("rdkit_input.sdf")[0]
    # prody_mol = rdkit_to_prody(mol)
    # sys.exit(0)
    # test prody
    # prot = prody.parseMMCIF("1crn", headers=True)
    pdb_id = "1jff"
    selection_string = "resname TA1"
    if len(sys.argv) > 1:
        pdb_id = sys.argv[1]
        selection_string = sys.argv[2]
    print("- parsing protein %s..." % pdb_id, end="")
    prot = prody.parseMMCIF(pdb_id, headers=True)

    # uncomment this to perceive bonds with ProDy (slow)
    # prot.inferBonds()
    print(">> %d atoms" % len(prot))
    # http://prody.csb.pitt.edu/manual/reference/atomic/select.html#module-prody.atomic.select
    sel = prot.select(selection_string)
    # sel = prot.select("resname THR resnum 1")
    print('- selected %d atoms with the string "%s"' % (len(sel), selection_string))
    # x = input()
    print("- converting selection to RDKit...")
    frag = prody_to_rdkit(sel)
    writer = Chem.SDWriter("fragment.sdf")
    writer.write(frag)
    writer.close()
    print("  [ written fragment.sdf ]")

    print("- converting the entire input as RDKit...")
    whole = prody_to_rdkit(prot)
    writer = Chem.SDWriter("whole.sdf")
    writer.write(whole)
    writer.close()
    print("  [ written whole.sdf ]")
