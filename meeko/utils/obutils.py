#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Raccoon
#

# TODO this should be in its own obabelutils module
from collections import namedtuple

import numpy as np
from openbabel import openbabel as ob

from . import geomutils
from . import utils


# named tuple to contain information about an atom
PDBAtomInfo = namedtuple('PDBAtomInfo', "name resName resNum chain")


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

    return PDBAtomInfo(name=name, resName=resName, resNum=resNum, chain=chain)


class SmartsFinder:
    """ simple SMARTS pattern finder"""

    def __init__(self):
        self.finder = ob.OBSmartsPattern()
        self.mol = None

    def setMolecule(self, mol):
        self.mol = mol

    def find(self, pattern):
        self.finder.Init(pattern)
        found = self.finder.Match(self.mol)
        if not found:
            return None
        return [list(x) for x in self.finder.GetUMapList()]


class SMARTSmatcher(object):
    """ base class to match SMARTS patterns in an OBMol"""

    def __init__(self, mol):
        if isinstance(mol, ob.OBMol):
            # use the OB smarts matcher
            self._finder = ob.OBSmartsPattern()
            self.find_pattern = self.find_pattern_OB
        else:
            print("Only OBMol supported for now")
            raise NotImplementedError
        self.mol = mol

    def find_pattern_OB(self, pattern, unique=True):
        """ use OB to find SMARTS patterns  """
        self._finder.Init(pattern)
        found = self._finder.Match(self.mol)
        if not found:
            # print "WARNING: MODIFIED FROM NONE TO []"
            return []
        # TODO consider if non-unique pattern matching is what we want
        # NOTE IMPORTANT!
        if unique == True:
            return [list(x) for x in self._finder.GetUMapList()]
        else:
            return [list(x) for x in self._finder.GetMapList()]
