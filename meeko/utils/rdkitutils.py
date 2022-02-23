from operator import itemgetter
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

class HJKRingDetection(object):
    """Implementation of the Hanser-Jauffret-Kaufmann exhaustive ring detection
    algorithm:
        ref:
        Th. Hanser, Ph. Jauffret, and G. Kaufmann
        J. Chem. Inf. Comput. Sci. 1996, 36, 1146-1152
    """

    def __init__(self, mgraph):
        self.mgraph = {key: [x for x in values] for (key, values) in mgraph.items()}
        self.rings = []
        self._iterations = 0

    def scan(self):
        """run the full protocol for exhaustive ring detection"""
        self.prune()
        self.build_pgraph()
        self.vertices = self._get_sorted_vertices()
        while self.vertices:
            self._remove_vertex(self.vertices[0])
        output_rings = []
        for ring in self.rings:
            output_rings.append(tuple(ring[:-1]))
        return output_rings

    def _get_sorted_vertices(self):
        """function to return the vertices to be removed, sorted by increasing
        connectivity order (see paper)"""
        vertices = ((k, len(v)) for k, v in self.mgraph.items())
        return [x[0] for x in sorted(vertices, key=itemgetter(1))]

    def prune(self):
        """iteratively prune graph until there are no nodes with only one
        connection"""
        while True:
            prune = []
            for node, neighbors in self.mgraph.items():
                if len(neighbors) == 1:
                    prune.append((node, neighbors))
            if len(prune) == 0:
                break
            for node, neighbors in prune:
                self.mgraph.pop(node)
                for n in neighbors:
                    self.mgraph[n].remove(node)

    def build_pgraph(self, prune=True):
        """convert the M-graph (molecular graph) into the P-graph (path/bond graph)"""
        self.pgraph = []
        for node, neigh in self.mgraph.items():
            for n in neigh:
                # use sets for unique id
                edge = set((node, n))
                if not edge in self.pgraph:
                    self.pgraph.append(edge)
        # re-convert the edges to lists because order matters in cycle detection
        self.pgraph = [list(x) for x in self.pgraph]

    def _remove_vertex(self, vertex):
        """remove a vertex and join all edges connected by that vertex (this is
        the REMOVE function from the paper)
        """
        visited = {}
        remove = []
        pool = []
        for path in self.pgraph:
            if self._has_vertex(vertex, path):
                pool.append(path)
        for i, path1 in enumerate(pool):
            for j, path2 in enumerate(pool):
                if i == j:
                    continue
                self._iterations += 1
                pair_id = tuple(set((i, j)))
                if pair_id in visited:
                    continue
                visited[pair_id] = None
                common = list(set(path1) & set(path2))
                common_count = len(common)
                # check if two paths have only this vertex in common or (or
                # two, if they're a cycle)
                if not 1 <= common_count <= 2:
                    continue
                # generate the joint path
                joint_path = self._concatenate_path(path1, path2, vertex)
                is_ring = joint_path[0] == joint_path[-1]
                # if paths share more than two vertices but they're not a ring, then skip
                if (common_count == 2) and not is_ring:
                    continue
                # store the ring...
                if is_ring:
                    self._add_ring(joint_path)
                # ...or the common path
                elif not joint_path in self.pgraph:
                    self.pgraph.append(joint_path)
        # remove used paths
        for p in pool:
            self.pgraph.remove(p)
        # remove the used vertex
        self.vertices.remove(vertex)

    def _add_ring(self, ring):
        """add newly found rings to the list (if not already there)"""
        r = set(ring)
        for candidate in self.rings:
            if r == set(candidate):
                return
        self.rings.append(ring)

    def _has_vertex(self, vertex, edge):
        """check if the vertex is part of this edge, and if true, return the
        sorted edge so that the vertex is the first in the list"""
        if edge[0] == vertex:
            return edge
        if edge[-1] == vertex:
            return edge[::-1]
        return None

    def _concatenate_path(self, path1, path2, v):
        """concatenate two paths sharing a common vertex
        a-b, c-b => a-b-c : idx1=1, idx2=1
        b-a, c-b => a-b-c : idx1=0, idx2=1
        a-b, b-c => a-b-c : idx1=1, idx2=0
        b-a, b-c => a-b-c : idx1=0, idx2=0
        """
        if not path1[-1] == v:
            path1.reverse()
        if not path2[0] == v:
            path2.reverse()
        return path1 + path2[1:]

    def _edge_in_pgraph(self, edge):
        """check if edge is already in pgraph"""
        e = set(edge)
        for p in self.pgraph:
            if e == set(p) and len(p) == len(edge):
                return True
        return False

