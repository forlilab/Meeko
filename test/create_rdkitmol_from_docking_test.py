from meeko import RDKitMolCreate
from meeko import PDBQTMolecule
from rdkit import Chem
import pathlib

workdir = pathlib.Path(__file__)
datadir = workdir.parents[0] / "create_rdkitmol_from_docking_data"

def test_asn_phe():
    fpath = datadir / "macrocycle-water-asn-phe.pdbqt"
    pdbqtmol = PDBQTMolecule.from_file(fpath, skip_typing=True)
    mols = RDKitMolCreate.from_pdbqt_mol(pdbqtmol) 
    assert(mols.count(None) == 0)

    bond_range_H = [0.85**2, 1.15**2] # square distances to avoid sqrt later
    bond_range_no_H = [1.2**2, 1.7**2]
    nr_conformers = set()
    for mol in mols:
        nr_conformers.add(mol.GetNumConformers())
        assert(len(nr_conformers) == 1)
        for confid in range(list(nr_conformers)[0]):
            positions = mol.GetConformer(confid).GetPositions()
            for bond in mol.GetBonds():
                has_H = bond.GetBeginAtom().GetAtomicNum() == 1
                has_H = has_H or (bond.GetEndAtom().GetAtomicNum() == 1)
                bond_range = bond_range_H if has_H else bond_range_no_H
                a = positions[bond.GetBeginAtomIdx(), :]
                b = positions[bond.GetEndAtomIdx(), :]
                dist_square = sum([(a[i]-b[i])**2 for i in range(3)])
                assert(dist_square > bond_range[0])
                assert(dist_square < bond_range[1])
