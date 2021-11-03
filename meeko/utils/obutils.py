#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Raccoon
#

# TODO this should be in its own obabelutils module
# from collections import namedtuple

import numpy as np
from openbabel import openbabel as ob

from . import geomutils
from . import utils
from . import pdbutils

mini_periodic_table = {
        1: 'H', 2: 'He', 3: 'Li', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 11: 'Na', 12: 'Mg',
        15: 'P', 16: 'S', 17: 'Cl', 19: 'K', 20: 'Ca', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni',
        29: 'Cu', 30: 'Zn', 34: 'Se', 35: 'Br', 53: 'I'}


# named tuple to contain information about an atom
# PDBAtomInfo = namedtuple('PDBAtomInfo', "name resName resNum chain")
# PDBResInfo  = namedtuple('PDBResInfo',       "resName resNum chain")


def getAtomIdxCoords(obmol, atom_idx):
    """return coordinates of atom idx """
    atom = obmol.GetAtom(atom_idx)
    return getAtomCoords(atom)


def getAtomCoords(atom):
    """ convert an OB atom into a numpy vector of coordinates """
    return np.array([atom.GetX(), atom.GetY(), atom.GetZ()], 'f')


def getCoordsFromAtomIndices(obmol, atomIdxList):
    """ extract coordinates for requested atom indices (return Numpy.array)"""
    coord = []
    for idx in atomIdxList:
        a = obmol.GetAtom(idx)
        coord.append(getAtomCoords(a))
    return np.array(coord)

def getAtoms(obmol):
    return ob.OBMolAtomIter(obmol)

def getAtomRes(atom):
    """ retrieve residue info about the atom """
    r = atom.GetResidue()
    data = {'num': r.GetNum(), 'name': r.GetName(), 'chain': r.GetChain()}
    return data


def atomsCentroid(obmol, atomIndices):
    """ calculate centroid from list of atom indices"""
    coord = []
    for i in atomIndices:
        atom = obmol.GetAtom(i)
        coord.append(atomToCoord(atom))
    return geomutils.averageCoords(coord)


def atomNeighbors(atom):
    """ return atom neighbors"""
    return [x for x in ob.OBAtomAtomIter(a)]


def load_molecule_from_file(fname, molecule_format=None):
    """ load molecule with openbabel"""
    if molecule_format is None:
        n, ftype = utils.getNameExt(fname)
        ftype = ftype.lower()

    mol = ob.OBMol()
    conv = ob.OBConversion()
    conv.SetInFormat(molecule_format)
    conv.ReadFile(mol, fname)

    return mol


def load_molecule_from_string(string, molecule_format):
    """ load molecule with openbabel"""
    mol = ob.OBMol()
    conv = ob.OBConversion()
    conv.SetInFormat(molecule_format)
    conv.ReadString(mol, string)

    return mol


def writeMolecule(mol, fname=None, ftype=None):
    """ save a molecule with openbabel"""
    if ftype is None:
        n, ftype = utils.getNameExt(fname)
        ftype = ftype.lower()

    conv = ob.OBConversion()
    conv.SetOutFormat(ftype)

    if not fname is None:
        conv.WriteFile(mol, fname)
    else:
        return conv.WriteString(mol)


def getPdbInfo(atom):
    """extract information for populating an ATOM/HETATM line
    in the PDB"""
    res = atom.GetResidue()
    if res is None:
        return None
    name = res.GetAtomID(atom)
    chain = res.GetChain()
    resNum = int(res.GetNumString())  # safe way for negative resnumbers
    resName = res.GetName()

    return pdbutils.PDBAtomInfo(name=name, resName=resName, resNum=resNum, chain=chain)


def getPdbInfoNoNull(atom):
    """extract information for populating an ATOM/HETATM line
    in the PDB"""
    res = atom.GetResidue()
    if res is None:
        name = '%-2s' % mini_periodic_table[atom.GetAtomicNum()]
        chain = ' '
        resNum = 1
        resName = 'UNL'
    else:
        name = res.GetAtomID(atom)
        chain = res.GetChain()
        resNum = int(res.GetNumString())  # safe way for negative resnumbers
        resName = res.GetName()
    return pdbutils.PDBAtomInfo(name=name, resName=resName, resNum=resNum, chain=chain)


class OBMolSupplier:
    def __init__(self, fname, _format):
        """  """
        self.fname = fname
        self.conv = ob.OBConversion()
        status = self.conv.SetInFormat(_format)
        if not status:
            raise RuntimeError('could not set OBConversion input format: %s' % _format)
        self.got_mol_in_cache = False
        self.cached_mol = None

    def __iter__(self):
        self.cached_mol = ob.OBMol()
        self.got_mol_in_cache = self.conv.ReadFile(self.cached_mol, self.fname)
        return self

    def __next__(self):
        if self.got_mol_in_cache:
            mol = self.cached_mol
            self.cached_mol = ob.OBMol()
            self.got_mol_in_cache = self.conv.Read(self.cached_mol)
            return mol
        else:
            raise StopIteration


