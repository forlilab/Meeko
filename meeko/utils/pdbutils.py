
from collections import namedtuple

# named tuple to contain information about an atom
PDBAtomInfo = namedtuple('PDBAtomInfo', "name resName resNum chain")
PDBResInfo  = namedtuple('PDBResInfo',       "resName resNum chain")

# standard nucleic acid residue names
nucleic = ['U', 'A', 'C', 'G', 'T']
