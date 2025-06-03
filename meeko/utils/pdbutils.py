from collections import namedtuple

# named tuple to contain information about an atom
PDBAtomInfo = namedtuple("PDBAtomInfo", "name resName resNum icode chain")
PDBResInfo = namedtuple("PDBResInfo", "resName resNum chain")  # used in obutils, maybe

def strip_altloc_from_pdb_file(filename):
    """this is a hack to pass a PDB string to RDKit's PDB parser because
    it ignores atoms for which altloc isn't either "A" or " ", but we
    may want to pass a file with "B" altlocs to the --box_enveloping
    argument of mk_prepare_receptor.py"""

    pdbstr = ""
    with open(filename) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                pdbstr += line[:16] + " " + line[17:]
            else:
                pdbstr += line
    return pdbstr
