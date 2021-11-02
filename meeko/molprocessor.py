from rdkit import Chem
from rdkit.Chem import rdPartialCharges
from openbabel import openbabel as ob

import numpy as np

from .setup import MoleculeSetup
from .utils import obutils, rdkitutils


OB_BOND_TABLE = {}
RDKIT_BOND_TABLE = {}
RDKIT_BOND_TABLE = {
        "UNSPECIFIED" : 0,
        "SINGLE" : 1,
        "DOUBLE" : 2,
        "TRIPLE" : 3,
        "AROMATIC" : 12,
        "ZERO" : 21,
        "IONIC" : 13,

        "DATIVE" : 17,
        "DATIVEL" : 18,
        "DATIVEONE" : 16,
        "DATIVER" : 19,
        "FIVEANDAHALF" : 11,
        "FOURANDAHALF" : 10,
        "HEXTUPLE" : 6,
        "HYDROGEN" : 14,
        "ONEANDAHALF" : 7,
        "OTHER" : 20,
        "QUADRUPLE" : 4,
        "QUINTUPLE" : 5,
        "THREEANDAHALF" : 9,
        "THREECENTER" : 15,
        "TWOANDAHALF" : 8,
                    }


# TODO : remove flexible amide option here: it should belong to the legacy bond typer

class MoleculeSetupInit(object):
    """ prototype class for generating molecule setups """
    def __init__(self): #, mol, flexible_amides=False, is_protein_sidechain=False):
        # TODO change the init options to be bond_options = {'flexible_amides': False, 'something':True}
        # and special={'is_protein_side_chain': False}
        self.init_mol(flexible_amides, is_protein_sidechain)

    def init_mol(self, mol, flexible_amides=False, is_protein_sidechain=False):
        """
        define a smarts pattern matching function/object that can be called with:
           self.smarts.find_pattern(pattern_string)
        """
        # define a smarts pattern matching function/object that can be called with find_pattern():
        self.smarts = None
        # set the number of true atoms to the total count of atoms in the input molecule
        self.atom_true_count = 0
        # extract atom properties
        self.init_atoms()
        # perceive rings and store the information
        self.perceive_rings()
        # extract bond properties
        self.init_bonds(flexible_amides)
        # bind the setup object to the molecule
        self.mol.setup = self
        if is_protein_sidechain:
            self.ignore_backbone()
        raise NotImplementedError("This method must be overloaded by inheriting class")

    def init_bond(self, flexible_amides):
        """ iterate through molecule bonds and build the bond table (id, table)
            INPUT
                flexible_amides: bool
                    this flag is used to define if primary and secondary amides
                    need to be considered flexible or not (bond_order= 999)
            CALCULATE
                bond_order: int
                    0       : pseudo-bond (for pseudoatoms)
                    1,2,3   : single-triple bond
                    5       : aromatic
                    999     : rigid

                if bond is in ring (both start and end atom indices in the bond are in the same ring)

            SETUP OPERATION
                Setup.add_bond(idx1, idx2, order, in_rings=[])
        """
        raise NotImplementedError("This method must be overloaded by inheriting class")

    def init_atom(self):
        """ iterate through molecule atoms and build the atoms table """
        raise NotImplementedError("This method must be overloaded by inheriting class")

    def perceive_rings(self):
        """ populate the rings and aromatic rings tableshe atoms table:
        self.rings_aromatics : list
            contains the list of ring_id items that are aromatic
        self.rings: dict
            store information about rings, like if they have corners that can
            be flipped and the graph of atoms that belong to them:

                self.rings[ring_id] = {
                                'corner_flip': False
                                'graph': {}
                                }

            The atom is built using the `walk_recursive` method
        self.ring_atom_to_ring_id: dict
            mapping of each atom belonginig to the ring: atom_idx -> ring_id
        """
        raise NotImplementedError("This method must be overloaded by inheriting class")



# TODO rename MoleculeSetupOB
class MoleculeSetupFromOB():
    """ create an instance of the setup from an OB molecule """
    def __init__(self, mol, flexible_amides=False, is_protein_sidechain=False, assign_charges=True):
        """ initialize the setup definition with an OBMol """
        if not isinstance(mol, ob.OBMol):
            raise TypeError('Input molecule must be an OBMol but is %s' % type(obmol))
        self.init_mol(mol, flexible_amides, is_protein_sidechain, assign_charges)

    def init_mol(self, mol, flexible_amides, is_protein_sidechain, assign_charges):
        """generate a new molecule setup

            NOTE: OpenBabel uses 1-based index
        """
        # save molecule information
        self.mol = mol
        # TODO check if the molecule has properties already (populate setup fields from SDF fields)
        # create a setup for the molecule
        self.setup = MoleculeSetup(self.mol)
        self.setup.name = self.mol.GetTitle()
        # define an OB SMARTS pattern matcher
        self.setup.smarts = obutils.SMARTSmatcher(self.mol)
        # initialize the total count of true atoms
        self.setup.atom_true_count = self.mol.NumAtoms()
        # extract atom information
        self.init_atom(assign_charges)
        # perceive ring information
        self.perceive_rings()
        # initialize bonds
        self.init_bond(flexible_amides)
        # update info for proteins (flex side chains?)
        if is_protein_sidechain:
            self.ignore_backbone()

    def init_atom(self, assign_charges):
        """initialize atom data table"""
        for a in ob.OBMolAtomIter(self.mol):
            # TODO fix this to be zero-based?
            partial_charge = a.GetPartialCharge() * float(assign_charges)
            self.setup.add_atom( a.GetIdx(),
                    coord=np.asarray(obutils.getAtomCoords(a), dtype='float'),
                    element=a.GetAtomicNum(),
                    charge=partial_charge,
                    atom_type=None,
                    pdbinfo = obutils.getPdbInfoNoNull(a),
                    neighbors=[x.GetIdx() for x in ob.OBAtomAtomIter(a)],
                    ignore=False, chiral=a.IsChiral())
            # TODO check consistency for chiral model between OB and RDKit

    def perceive_rings(self):
        """ collect information about rings"""
        perceived = self.mol.GetSSSR()
        for r in perceived:
            ring_id = tuple(r._path)
            if r.IsAromatic():
                self.setup.rings_aromatic.append(ring_id)
            self.setup.rings[ring_id] = {'corner_flip': False}
            graph = {}
            for member in ring_id:
                # atom to ring lookup
                self.setup.atom_to_ring_id[member].append(ring_id)
                # graph of atoms affected by potential ring movements
                graph[member] = self.setup.walk_recursive(member, collected=[], exclude=list(ring_id))
            self.setup.rings[ring_id]['graph'] = graph

    def init_bond(self, flexible_amides):
        """initialize bond data table"""
        for b in ob.OBMolBondIter(self.mol):
            idx1 = b.GetBeginAtomIdx()
            idx2 = b.GetEndAtomIdx()
            bond_order = b.GetBondOrder()
            if b.IsAromatic():
                bond_order = 5
            if b.IsAmide() and not b.IsTertiaryAmide():
                if not flexible_amides:
                    bond_order = 999
            # check if bond is a ring bond, i.e., both atoms belongs to the same ring
            idx1_rings = set(self.mol.setup.get_atom_rings(idx1))
            idx2_rings = set(self.mol.setup.get_atom_rings(idx2))
            in_rings = list(set.intersection(idx1_rings, idx2_rings))
            self.setup.add_bond(idx1, idx2, order=bond_order, in_rings=in_rings)

    def ignore_backbone(self):
        """ set ignore for PDB atom names 'C', 'N', 'H', and 'O'
            these atoms are kept in the rigid PDBQT by ReactiveReceptor"""
        # TODO this function is very fragile, to be replaced by SMARTS
        # also, not sure where it is used...
        exclude_pdbname = {'C': 0, 'N': 0, 'H': 0, 'O': 0} # store counts of found atoms
        for atom in ob.OBMolAtomIter(self.mol):
            pdbinfo = obutils.getPdbInfo(atom)
            if pdbinfo.name.strip() in exclude_pdbname:
                idx = atom.GetIdx()
                self.set_ignore(idx, True)
                exclude_pdbname[pdbinfo.name.strip()] += 1
        for name in exclude_pdbname:
            n_found = exclude_pdbname[name]
            if n_found != 1:
                print("Warning: expected 1 atom with PDB name '%s' but found %d" % (name, n_found))



class MoleculeSetupFromRDKit:
    """ create molecular setup for an RDKit molecule """
    def __init__(self, mol, flexible_amides=False, is_protein_sidechain=False, assign_charges=True):
        # FIXME these special types are necessary only to keep track of amide bonds
        # to make them non rotatable with the old AD4/Vina force fields
        # eventually it will be removed when the new ff will be deployed
        self._amide_bonds = {'any': ['[NX3][$([CX3](=[OX1])[#6])]', []],
                        'tertiary':['[NX3;H0][$([CX3](=[OX1])[#6])]',[] ],
                         }
        if not isinstance(mol, Chem.rdchem.Mol):
            raise TypeError('Input molecule must be a RDKit.Chem.Mol but is %s' % type(mol))
        self.init_mol(mol, flexible_amides, is_protein_sidechain, assign_charges)

    def init_mol(self, mol, flexible_amides, is_protein_sidechain, assign_charges):
        """perform the list of operations required to process the molecule
        """
        # store molecule object
        self.mol = mol
        # create a setup for the molecule
        self.setup = MoleculeSetup(self.mol)
        try:
            self.setup.name = self.mol.GetProp('_Name')
        except:
            pass
        # define an RDKit SMARTS pattern matcher
        # (in RDKit every molecule can match a SMARTS, but for compatibility, we need an object providing find_pattern()
        self.setup.smarts = rdkitutils.RDKitSMARTSHelper(mol)
        # cache special types
        self._cache_amides()
        # initialize the total count of true atoms
        self.atom_true_count = self.mol.GetNumAtoms()
        # extract atom information
        self.init_atom(assign_charges)
        # perceive ring information
        self.perceive_rings()
        # initialize bonds
        self.init_bond(flexible_amides)
        # update info for proteins (flex side chains?)
        if is_protein_sidechain:
            self.ignore_backbone()

    def _cache_amides(self):
        """store information used for the bond parametrization """
        for name, pattern_info in self._amide_bonds.items():
            pattern = pattern_info[0]
            print("caching",name)
            found = self.setup.smarts.find_pattern(pattern)
            if len(found):
                for f in found:
                    self._amide_bonds[name][1].append(set(f))

    def init_atom(self, assign_charges):
        """ initialize the atom table information """
        # extract the coordinates
        c = self.mol.GetConformers()[0]
        coords = c.GetPositions()
        # extract/generate charges
        if assign_charges:
            Chem.rdPartialCharges.ComputeGasteigerCharges(self.mol)
            charges = [a.GetDoubleProp('_GasteigerCharge') for a in self.mol.GetAtoms()]
        else:
            charges = [0.0] * self.mol.GetNumAtoms()
        # perceive chirality
        # TODO check consistency for chiral model between OB and RDKit
        chiral_info = {}
        for data in Chem.FindMolChiralCenters(self.mol, includeUnassigned=True):
            chiral_info[data[0]] = data[1]
        # register atom
        for a in self.mol.GetAtoms():
            idx = a.GetIdx()
            chiral = False
            if idx in chiral_info:
                chiral = chiral_info[idx]
            self.setup.add_atom(idx,
                    coord=coords[idx],
                    element=a.GetAtomicNum(),
                    charge=charges[idx],
                    atom_type=None,
                    pdbinfo = rdkitutils.getPdbInfoNoNull(a),
                    neighbors = [n.GetIdx() for n in a.GetNeighbors() ],
                    chiral=False,
                    ignore=False)

    def init_bond(self, flexible_amides):
        """ initialize bond information """
        for b in self.mol.GetBonds():
            idx1 = b.GetBeginAtomIdx()
            idx2 = b.GetEndAtomIdx()
            bond_order = int(b.GetBondType())
            # fix the RDKit aromatic type (FIXME)
            if bond_order == 12: # aromatic
                bond_order = 5
            if bond_order == 1:
                rotatable = True
                if flexible_amides:
                    continue
                # check if bond is a tertiary amide bond
                if set((idx1, idx2)) in self._amide_bonds['tertiary'][1]:
                    # TODO check what is the best to keep here (rotatable or bond_order)
                    rotatable=False
                    bond_order = 999
            else:
                rotatable = False
            idx1_rings = set(self.mol.setup.get_atom_rings(idx1))
            idx2_rings = set(self.mol.setup.get_atom_rings(idx2))
            in_rings = list(set.intersection(idx1_rings, idx2_rings))
            self.setup.add_bond(idx1, idx2, order=bond_order, rotatable=rotatable, in_rings=in_rings)


    def perceive_rings(self):
        """ perceive ring information """

        def isRingAromatic(bonds_in_ring):
            """ Index ID#: RDKitCB_8
            RDKit recipe for identifying if ring is aromatic
            """
            for bond_idx in bonds_in_ring:
                if not self.mol.GetBondWithIdx(bond_idx).GetIsAromatic():
                    return False
            return True

        ring_info = self.mol.GetRingInfo()
        perceived = ring_info.AtomRings()
        bond_rings = ring_info.BondRings()
        for idx, ring_id in enumerate(perceived):
            if isRingAromatic(bond_rings[idx]):
                self.setup.rings_aromatic.append(ring_id)
            self.setup.rings[ring_id] = {'corner_flip':False}
            graph = {}
            for member in ring_id:
                # atom to ring lookup
                self.setup.atom_to_ring_id[member].append(ring_id)
                # graph of atoms affected by potential ring movements
                graph[member] = self.setup.walk_recursive(member, collected=[], exclude=list(ring_id))
            self.setup.rings[ring_id]['graph'] = graph

    def ignore_backbone(self): # TODO
        """ ignore backbone information """
        raise NotImplementedError

    def write_mol(self, fname=None, _format='sdf'):
        """ write the fully configured molecule in a standard format (SDF?) """
        raise NotImplementedError
        # SDF MOL FORMAT
        # - extract coordinates
        # upate molecule coordinates from the setup.coord field:
        #   from rdkit.Geometry import Point3D
        #   conf = mol.GetConformer()
        #   for i in range(mol.GetNumAtoms()):
        #     x,y,z = new_atom_ps[i]
        #     conf.SetAtomPosition(i,Point3D(x,y,z))
        # SDF EXTRA FIELDS
        # - extract charges
        # - extract graph
        # - extract bond parameters
        # - extract rotatable flags
        # - extract ignore
        # - extract pseudoatoms
        # - ring perception stuff
        # - protein backbone stuff
        # - interaction vectors [ DEPRECATED? ]
