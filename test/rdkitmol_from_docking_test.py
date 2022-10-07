from meeko import RDKitMolCreate
from meeko import PDBQTMolecule
from rdkit import Chem
import pathlib

workdir = pathlib.Path(__file__)
datadir = workdir.parents[0] / "rdkitmol_from_docking_data"

# test for reasonable bond lengths. This will fail if the atom names
# in RDKiMolCreate.flexres[<resname>]["atom_names"] are in the wrong
# order w.r.t RDKitMolCreate.flexres[<resname>]["smiles"]
bond_range_H = [0.85, 1.15]
bond_range_no_H = [1.16, 1.65]
bond_S_bonus = 0.2

def run(fpath, nr_expected_none, is_dlg, skip_typing):
    pdbqtmol = PDBQTMolecule.from_file(fpath, is_dlg=is_dlg, skip_typing=skip_typing)
    mols = RDKitMolCreate.from_pdbqt_mol(pdbqtmol) 
    assert(mols.count(None) == nr_expected_none)
    nr_conformers = set()
    for mol in mols:
        if mol is None:
            continue
        nr_conformers.add(mol.GetNumConformers())
        assert(len(nr_conformers) == 1)
        for confid in range(list(nr_conformers)[0]):
            positions = mol.GetConformer(confid).GetPositions()
            for bond in mol.GetBonds():
                has_H = bond.GetBeginAtom().GetAtomicNum() == 1
                has_H = has_H or (bond.GetEndAtom().GetAtomicNum() == 1)
                bond_range = bond_range_H if has_H else bond_range_no_H
                has_S = bond.GetBeginAtom().GetAtomicNum() == 16
                has_S = has_S or (bond.GetEndAtom().GetAtomicNum() == 16)
                if has_S:
                    bond_range = [value + bond_S_bonus for value in bond_range] 
                a = positions[bond.GetBeginAtomIdx(), :]
                b = positions[bond.GetEndAtomIdx(), :]
                dist = sum([(a[i]-b[i])**2 for i in range(3)])**0.5
                assert(dist > bond_range[0])
                assert(dist < bond_range[1])

def test_asn_phe():
    fpath = datadir / "macrocycle-water-asn-phe.pdbqt"
    run(fpath, nr_expected_none=0, is_dlg=False, skip_typing=True)

def test_22_flexres():
    fpath = datadir / "22-flexres.pdbqt"
    run(fpath, nr_expected_none=0, is_dlg=False, skip_typing=True)

def test_phe_badphe():
    fpath = datadir / "arg_gln_asn_phe_badphe.pdbqt"
    run(fpath, nr_expected_none=1, is_dlg=False, skip_typing=True)
