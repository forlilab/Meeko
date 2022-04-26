from collections import namedtuple
from rdkit import Chem
from rdkit.Chem import AllChem, rdGeometry
import prody
import warnings


from collections import defaultdict
def nested_dict():
    return defaultdict(nested_dict)



# class CovLigandPrepared(object):

#     """ container class to store the prepared ligand molecule with associated information"""
#     def __init__(self, mol, res_id, at_names, smarts, indices, smarts_indices):
#         self.mol = mol
#         self.res_id = res_id
#         self.at_names = at_names
#         self.smarts = smarts
#         self.indices = indices
#         self.smarts_indices = smarts_indices

#     @property
#     def label(self):
#         """ DOCUMENTATION GOES HERE """
#         pass

#     @property
#     def label(self):
#         """ DOCUMENTATION GOES HERE """
#         pass





CovLigandPrepared = namedtuple("CovalentLigandPrepared", ["mol", "res_id", "at_names", "smarts", "smarts_indices", "indices", "label"])


class CovalentBuilder(object):
    """ Class to perform structural alignments of ligands containing the target side chain attached
        to run tethered covalent dockings

        The class is instantiated for a given target, with a list of one or more residues, then
        ligands can be processed sequentially ( CovalentBuilder.process() )

            receptor_mol    :       ProDy molecule
            residue_string  :       string specifying residues to process. Examples:
                                    ":LYS:"              (all LYS, default atom names: CA,CB)
                                    "A:LYS:204:CA,CB"
                                    "A:HIS::ND1,CE1"     (all HIS on chain A)

        (ProDy regular expressions could be also injected here).
    """
    def __init__(self, receptor_mol, residue_string):
        self.rec = receptor_mol
        selection_tuple = self.parse_residue_string(residue_string, force_CA_CB=True)
        # selection tuple: (chain, res, num, atname1, atname2)
        # only 'res' and 'atname[1|2]' are required, the rest is optional,
        # e.g: (None, res, None, 'CA', 'CB')
        out = self._generate_prody_selection(selection_tuple)
        self._compact_selection(out)

    def _generate_prody_selection(self, residue):
        """ generate the string to perform a Prody.Selection """
        # sel_string = 'chid %s AND resname %s AND resnum %s AND (name %s or name %s)'
        sel_string = []
        # out = []
        chid, res_type, res_num, atname1, atname2 = residue
        if not chid == "":
            sel_string.append("chid %s" % chid)
        sel_string.append('resname %s' % res_type)
        if not res_num == "":
            sel_string.append("resnum %s" % res_num)
        sel_string.append("(name %s or name %s)" % (atname1, atname2))
        sel_string = " and ".join(sel_string)
        print("CovalentBuilder> searching for residue:",sel_string)
        found = self.rec.select( sel_string )
        if found is None:
            print("ERROR: no residue found with the following specification: chain[%s] residue[%s] number[%s] atom names [%s,%s]"% (
                chid, res_type, res_num, atname1, atname2))
            raise ValueError
        return (found, atname1, atname2)
        # returnsout

    def _compact_selection(self, sel_info, allow_missing=False):
        """ process a ProDy selection and return a dictionary with the structure { res_id : (at1_coord, at2_coord), ...} """
        self.residues = {}
        # for (sel, at1, at2) in sel_info:
        sel, at1, at2 = sel_info
        pairs = (at1, at2)
        chains = sel.getChids()
        res = sel.getResnames()
        num = sel.getResnums()
        names = sel.getNames()
        coords = sel.getCoords()
        for i in range(len(chains)):
            res_id = (chains[i], res[i], num[i])
            if not res_id in self.residues:
                self.residues[res_id] = [None, None]
            # preserve the order of the coordinates
            # as specified by the atom names order
            idx = pairs.index(names[i])
            self.residues[res_id][idx] = {'coords':coords[i], 'atname':names[i]}
        res_id_list = list(self.residues.keys())
        for res_id in res_id_list:
            if None in self.residues[res_id]:
                c,r,n = res_id
                f = [ x['atname'] if not x is None else "None" for x in  self.residues[res_id]]
                print("WARNING: one or more atoms are missing in residue %s:%s%d (requested: %s,%s | found: %s,%s)" % (c,r,n,at1,at2, f[0], f[1]) )
                if not allow_missing:
                    raise ValueError
                del self.residues[res_id]
        # from pprint import pprint as pp
        # pp(self.residues)

    def process(self, ligand, smarts=None, smarts_indices=None, indices=None, first_only=False):
        """ process the ligand for the residue(s) specified for the current receptor"""
        if (smarts is None) and (indices is None):
            raise ValueError("Specify at least one criterion, either SMARTS pattern or atom indices (2)")
        # if SMARTS are specified, use that to define (or override) indices
        if not smarts is None:
            indices = self.find_smarts(ligand, smarts, smarts_indices, first_only)
        if len(indices) == 0:
            warnings.warn("SMARTS pattern didn't match any atoms", RuntimeWarning)
        #print("CovalentBuilder> Generating %d ouput alignments (%d residues)" % (len(indices)*len(self.residues), len(self.residues) ))
        # perform alignments
        for i, idx_pair in enumerate(indices):
            for res_info, res_coord in self.residues.items():
                if len(indices)>1:
                    counter = "_%d" % (i+1)
                else:
                    counter = ""
                chain, res, num, = res_info
                at_names = [ x['atname'] for x in res_coord ]
                coord = [ x['coords'] for x in res_coord ]
                label = "%s:%s%s%s" % (chain, res, num, counter)
                mol = self.transform(ligand, idx_pair, coord)
                yield CovLigandPrepared(mol, res_info, at_names, smarts, smarts_indices, idx_pair,label)

    def find_smarts(self, mol, smarts, smarts_indices, first_only):
        """ find occurrences of the SMARTS indices atoms in the requested SMARTS"""
        indices = []
        patt = Chem.MolFromSmarts(smarts)
        smarts_size = patt.GetNumAtoms()
        if smarts_indices[0] >= smarts_size or smarts_indices[1] >= smarts_size:
            raise ValueError("SMARTS index exceeds number of atoms in SMARTS (%d)" % (smarts_size))
        found = mol.GetSubstructMatches(patt)
        #print("CovalentBuilder> ligand patterns found: ", found, "[ use only first: %s ]" % first_only)
        if len(found)>1 and first_only:
            print("WARNING: the specified ligand pattern returned more than one match: [%d] (potential ambiguity?)" % len(found))
        for f in found:
            #print("CovalentBuilder> processing:", f, "with ", smarts_indices)
            indices.append([f[x] for x in smarts_indices])
            #print("CovalentBuilder> ligand indices stored:", indices)
            if first_only:
                return indices
        return indices

    def transform(self, ligand, index_pair, coord):
        """ generate translatead and aligned molecules for each of the indices requested
            and for all the residues defined in the class constructor

            SOURCE: https://sourceforge.net/p/rdkit/mailman/message/36750909/
        """
        # make a copy of the ligand
        # TODO: maybe define new conformers?
        mol = Chem.Mol(ligand)
        target = Chem.MolFromSmiles("CC")
        # add hydrogens
        target = Chem.AddHs(target)
        # generate 3D coords
        AllChem.EmbedMolecule(target)
        # get the first conformer
        conf = target.GetConformer()
        # set coordinates to the ones of the actual target residue
        c1 = rdGeometry.Point3D( coord[0][0], coord[0][1], coord[0][2] )
        c2 = rdGeometry.Point3D( coord[1][0], coord[1][1], coord[1][2] )
        conf.SetAtomPosition(0, c1)
        conf.SetAtomPosition(1, c2)
        # perform alignment
        Chem.rdMolAlign.AlignMol(mol, target, -1, -1,[(index_pair[0],0), (index_pair[1], 1)] )
        return mol

    @classmethod
    def parse_residue_string(cls, string, force_CA_CB=True):
        chain, res, num, *atom_names = string.split(":")
        #print("PARSED c:%s r:%s n:%s AAA:%s" % (chain, res, num, atom_names))
        if not num == "":
            if not num.isdigit():
                raise ValueError('NUM must be an integer in "CHAIN:RES:NUM" or "CHAIN:RES:NUM:ATOM1,ATOM2"')
        # parse atom names
        #  TODO clean all specifications
        if atom_names == []:
            atom_names = "CA,CB"
        else:
            atom_names = atom_names[0]
        # trigger error if comma not found
        a1, a2 = atom_names.split(",")
        # check that the legacy_format flag is not used
        if force_CA_CB and (not a1 == 'CA' or not a2 == 'CB'):
            msg  = "Atom names expected to be CA,CB but got %s,%s instead.\n" % (a1, a2)
            raise RuntimeError(msg)
        residue = (chain, res, num, a1, a2)
        return residue
