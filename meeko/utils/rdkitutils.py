from rdkit import Chem


"""
create new RDKIT residue

mi  =  Chem.AtomPDBResidueInfo()
mi.SetResidueName('MOL')
mi.SetResidueNumber(1)
mi.SetOccupancy(0.0)
mi.SetTempFactor(0.0)

source: https://sourceforge.net/p/rdkit/mailman/message/36404394/




"""
from collections import namedtuple
PDBAtomInfo = namedtuple('PDBAtomInfo', "name resName resNum chain")

mini_periodic_table = {
                1: 'H', 2: 'He', 3: 'Li', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 11: 'Na', 12: 'Mg',
                        15: 'P', 16: 'S', 17: 'Cl', 19: 'K', 20: 'Ca', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni',
                                29: 'Cu', 30: 'Zn', 34: 'Se', 35: 'Br', 53: 'I'
        }






def getPdbInfoNoNull(atom):
    """extract information for populating an ATOM/HETATM line
    in the PDB"""
    # res = atom.GetResidue()
    minfo = atom.GetMonomerInfo()
    if minfo is None:
        name = '%-2s' % mini_periodic_table[atom.GetAtomicNum()]
        chain = ' '
        resNum = 1
        resName = 'UNL'
    else:
        name = minfo.GetName()
        chain = minfo.GetChainId()
        resNum = minfo.GetResidueNum() 
        resName = minfo.GetResidueName()
    return PDBAtomInfo(name=name, resName=resName, resNum=resNum, chain=chain)


class RDKitSMARTSHelper:
    def __init__(self, mol):
        self.mol = mol
    def find_pattern(self, pattern):
        patt = Chem.MolFromSmarts(pattern)
        return self.mol.GetSubstructMatch(patt)


