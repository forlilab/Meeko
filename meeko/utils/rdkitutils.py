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
         1:'H',   2:'He',
         3:'Li',  4:'Be',  5:'B',   6:'C',   7:'N',   8:'O',   9:'F',  10:'Ne',
        11:'Na', 12:'Mg', 13:'Al', 14:'Si', 15:'P',  16:'S',  17:'Cl', 18:'Ar',
        19:'K',  20:'Ca', 21:'Sc', 22:'Ti', 23:'V',  24:'Cr', 25:'Mn', 26:'Fe', 27:'Co', 28:'Ni', 29:'Cu', 30:'Zn',
        31:'Ga', 32:'Ge', 33:'As', 34:'Se', 35:'Br', 36:'Kr',
        37:'Rb', 38:'Sr', 39:'Y',  40:'Zr', 41:'Nb', 42:'Mo', 43:'Tc', 44:'Ru', 45:'Rh', 46:'Pd', 47:'Ag', 48:'Cd',
        49:'In', 50:'Sn', 51:'Sb', 52:'Te', 53:'I',  54:'Xe',
        55:'Cs', 56:'Ba',
        57:'La', 58:'Ce', 59:'Pr', 60:'Nd', 61:'Pm', 62:'Sm', 63:'Eu', 64:'Gd', 65:'Tb', 66:'Dy', 67:'Ho', 68:'Er', 69:'Tm', 70:'Yb',
        71:'Lu', 72:'Hf', 73:'Ta', 74:'W',  75:'Re', 76:'Os', 77:'Ir', 78:'Pt', 79:'Au', 80:'Hg',
        81:'Tl', 82:'Pb', 83:'Bi', 84:'Po', 85:'At', 86:'Rn',
        87:'Fr', 88:'Ra'
        }

def getPdbInfoNoNull(atom):
    """extract information for populating an ATOM/HETATM line
    in the PDB"""
    # res = atom.GetResidue()
    minfo = atom.GetMonomerInfo()
    if minfo is None:
        atomic_number = atom.GetAtomicNum()
        if atomic_number == 0:
            name = '%-2s' % '*'
        else:
            name = '%-2s' % mini_periodic_table[atomic_number]
        chain = ' '
        resNum = 1
        resName = 'UNL'
    else:
        name = minfo.GetName()
        chain = minfo.GetChainId()
        resNum = minfo.GetResidueNumber()
        resName = minfo.GetResidueName()
    return PDBAtomInfo(name=name, resName=resName, resNum=resNum, chain=chain)


class Mol2MolSupplier():
    """ RDKit Mol2 molecule supplier.
    Parameters
        sanitize: perform RDKit sanitization of Mol2 molecule"""
    def __init__(self, filename, sanitize=True, removeHs=False, cleanupSubstructures=True):
        self.fp = open(filename, 'r')
        self._opts = {'sanitize':sanitize,
                'removeHs':removeHs,
                'cleanupSubstructures':cleanupSubstructures }
        self.buff = []

    def __iter__(self):
        return self

    def __next__(self):
        """ iterator step """
        while True:
            line = self.fp.readline()
            # empty line
            if not line:
                if len(self.buff):
                    # buffer full, returning last molecule
                    mol=Chem.MolFromMol2Block("".join(self.buff), **self._opts)
                    self.buff = []
                    return mol
                # buffer empty, stopping the iteration
                self.fp.close()
                raise StopIteration
            if '@<TRIPOS>MOLECULE' in line:
                # first molecule parsed
                if len(self.buff)==0:
                    self.buff.append(line)
                else:
                    # found the next molecule, breaking to return the complete one
                    break
            else:
                # adding another line in the current molecule
                self.buff.append(line)
        # found a complete molecule, returning it
        mol=Chem.MolFromMol2Block("".join(self.buff), **self._opts)
        self.buff = [line]
        return mol
