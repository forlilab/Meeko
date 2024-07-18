from collections import namedtuple

# named tuple to contain information about an atom
PDBAtomInfo = namedtuple("PDBAtomInfo", "name resName resNum icode chain")
PDBResInfo = namedtuple("PDBResInfo", "resName resNum chain")  # used in obutils, maybe
