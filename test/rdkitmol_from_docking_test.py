from meeko import RDKitMolCreate
from meeko import PDBQTMolecule
from meeko import MoleculePreparation
from meeko import PDBQTWriterLegacy
from rdkit import Chem
import numpy as np
import pathlib
import json
import io

workdir = pathlib.Path(__file__)
datadir = workdir.parents[0] / "rdkitmol_from_docking_data"

# test for reasonable bond lengths. This will fail if the atom names
# in RDKiMolCreate.flexres[<resname>]["atom_names"] are in the wrong
# order w.r.t RDKitMolCreate.flexres[<resname>]["smiles"]
bond_range_H = [0.85, 1.15]
bond_range_no_H = [1.16, 1.65]
bond_S_bonus = 0.2

def check_rdkit_bond_lengths(fpath, nr_expected_none, is_dlg, skip_typing):
    pdbqtmol = PDBQTMolecule.from_file(fpath, is_dlg=is_dlg, skip_typing=skip_typing)
    return run_from_pdbqtmol(pdbqtmol, nr_expected_none)

def run_from_pdbqtmol(pdbqtmol, nr_expected_none=0):
    mols = RDKitMolCreate.from_pdbqt_mol(pdbqtmol, keep_flexres=True)
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
                has_S = bond.GetBeginAtom().GetAtomicNum() in [16, 17, 35]
                has_S = has_S or (bond.GetEndAtom().GetAtomicNum() in [16, 17, 35])
                if has_S:
                    bond_range = [value + bond_S_bonus for value in bond_range] 
                a = positions[bond.GetBeginAtomIdx(), :]
                b = positions[bond.GetEndAtomIdx(), :]
                dist = sum([(a[i]-b[i])**2 for i in range(3)])**0.5
                assert(dist > bond_range[0])
                assert(dist < bond_range[1])

def test_asn_phe():
    fpath = datadir / "macrocycle-water-asn-phe.pdbqt"
    check_rdkit_bond_lengths(fpath, nr_expected_none=0, is_dlg=False, skip_typing=True)

def test_22_flexres():
    fpath = datadir / "22-flexres.pdbqt"
    check_rdkit_bond_lengths(fpath, nr_expected_none=0, is_dlg=False, skip_typing=True)

def test_phe_badphe():
    fpath = datadir / "arg_gln_asn_phe_badphe.pdbqt"
    check_rdkit_bond_lengths(fpath, nr_expected_none=1, is_dlg=False, skip_typing=True)

def test_arg_his():
    fpath = datadir / "arg_his.pdbqt"
    check_rdkit_bond_lengths(fpath, nr_expected_none=0, is_dlg=False, skip_typing=True)


# The following tests  generate the PDBQT and convert it back to RDKit,
# as opposed to the tests above which start from PDBQT.

mk_prep = MoleculePreparation()
mk_prep_wet = MoleculePreparation(hydrate=True)

def run(sdfname, wet=False):
    fpath = datadir / sdfname
    for mol in Chem.SDMolSupplier(str(fpath), removeHs=False):
        if wet:
            setups = mk_prep_wet.prepare(mol)
            pdbqt, is_ok, error_msg = PDBQTWriterLegacy.write_string(setups[0])
        else:
            setups = mk_prep.prepare(mol)
            pdbqt, is_ok, error_msg = PDBQTWriterLegacy.write_string(setups[0])
        pmol = PDBQTMolecule(pdbqt)
        run_from_pdbqtmol(pmol)

def test_small_01_zero_deuterium(): run("small-01_zero-deuterium.sdf")
def test_small_01_one_deuterium(): run("small-01_one-deuterium.sdf")
def test_small_01_two_deuterium(): run("small-01_two-deuterium.sdf")
def test_small_01_three_deuterium(): run("small-01_three-deuterium.sdf")
def test_small_01_four_deuterium(): run("small-01_four-deuterium.sdf")

def test_small_02_zero_deuterium(): run("small-02_zero-deuterium.sdf")
def test_small_02_one_deuterium_A(): run("small-02_one-deuterium-A.sdf")
def test_small_02_one_deuterium_B(): run("small-02_one-deuterium-B.sdf")
def test_small_02_one_deuterium_C(): run("small-02_one-deuterium-C.sdf")
def test_small_02_two_deuterium_A(): run("small-02_two-deuterium-A.sdf")
def test_small_02_two_deuterium_B(): run("small-02_two-deuterium-B.sdf")
def test_small_02_two_deuterium_C(): run("small-02_two-deuterium-C.sdf", wet=True)
def test_small_02_two_deuterium_D(): run("small-02_two-deuterium-D.sdf", wet=True)
def test_small_02_five_deuterium(): run("small-02_five-deuterium.sdf", wet=True)

def test_small_03_zero_deuterium(): run("small-03_zero-deuterium.sdf", wet=True)
def test_small_03_one_deuterium_A(): run("small-03_one-deuterium-A.sdf", wet=True)
def test_small_03_one_deuterium_B(): run("small-03_one-deuterium-B.sdf", wet=True)
def test_small_03_two_deuterium(): run("small-03_two-deuterium.sdf")
def test_small_03_three_deuterium(): run("small-03_three-deuterium.sdf")

def test_small_04(): run("small-04.sdf", wet=True)

def test_meeko_free_energy_prop_vina():
    fpath = datadir/ "vina-result-ethanol.pdbqt"
    pdbqt_mol = PDBQTMolecule.from_file(fpath)
    ref_charges = [0.034, 0.152, -0.397, 0.21]
    tolerance = 0.01
    assert np.all(np.abs(pdbqt_mol.atoms()["partial_charges"] - ref_charges) < tolerance)
    sd_string, failures = RDKitMolCreate.write_sd_string(pdbqt_mol)
    assert len(failures) == 0
    sio = io.BytesIO(sd_string.encode())
    nr_mols = 0
    for rdkit_mol in Chem.ForwardSDMolSupplier(sio):
        nr_mols += 1
    assert nr_mols == 1
    data = json.loads(rdkit_mol.GetProp("meeko"))
    assert "free_energy" in data
    assert data["free_energy"] == -2.7

def test_meeko_free_energy_prop_adgpu():
    fpath = datadir/ "adgpu-result-ethanol.dlg"
    pdbqt_mol = PDBQTMolecule.from_file(fpath, is_dlg=True)
    sd_string, failures = RDKitMolCreate.write_sd_string(pdbqt_mol, only_cluster_leads=True)
    assert len(failures) == 0
    sio = io.BytesIO(sd_string.encode())
    rdkit_mols = [mol for mol in Chem.ForwardSDMolSupplier(sio)]
    assert len(rdkit_mols) == 4 # 20 runs, 4 clusters
    data = json.loads(rdkit_mols[0].GetProp("meeko"))
    assert "free_energy" in data
    assert data["free_energy"] == -2.11

def test_offsite_charges():
    mk_config = {
        "input_offatom_params": {
            "offsite": [
                {"smarts": "[#7X3;v3;!+]([*])([*])[*]",
                 "IDX": [0],
                 "OFFATOMS": [{"z": [1, 2, 3], "phi": 0, "distance": 0.2, "atype": "LP", "pull_charge_fraction": 1.08}]
                 }
            ]
        }
    }

    sdfname = "small-01_three-deuterium.sdf"
    fpath = datadir / sdfname
    mk_prep = MoleculePreparation.from_config(mk_config)
    for mol in Chem.SDMolSupplier(str(fpath), removeHs=False):
        break # we have just 1 mol
    setups = mk_prep.prepare(mol)
    molsetup = setups[0]
    pdbqt, is_ok, error_msg = PDBQTWriterLegacy.write_string(molsetup)
    assert is_ok
    assert "LP" in pdbqt
    pmol = PDBQTMolecule(pdbqt, skip_typing=True)
    rdkit_mols = RDKitMolCreate.from_pdbqt_mol(pmol, keep_flexres=True)
    assert len(rdkit_mols) == 1
    smiles = Chem.MolToSmiles(rdkit_mols[0])
    assert type(smiles) == str
    assert len(smiles) > 0

def test_chiral_deutorated():
    def export_wrapper(mk_prep, mol):
        molsetup = mk_prep(mol)[0]
        pdbqt_str, is_ok, err = PDBQTWriterLegacy.write_string(molsetup)
        assert is_ok
        pdbqt_mol = PDBQTMolecule(pdbqt_str, skip_typing=True)
        mol_list = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
        return mol_list[0]  # not multi-mol or flexrex
    mk_prep = MoleculePreparation()
    l_gly_fn = str(datadir / "chiral_deutorated_L_gly.sdf")
    d_gly_fn = str(datadir / "chiral_deutorated_D_gly.sdf")
    l_gly = next(Chem.SDMolSupplier(l_gly_fn, removeHs=False))
    d_gly = next(Chem.SDMolSupplier(d_gly_fn, removeHs=False))
    l_gly_exported = export_wrapper(mk_prep, l_gly)
    d_gly_exported = export_wrapper(mk_prep, d_gly)
    Chem.AssignStereochemistryFrom3D(l_gly_exported)
    Chem.AssignStereochemistryFrom3D(d_gly_exported)
    assert Chem.MolToSmiles(l_gly_exported) == Chem.MolToSmiles(l_gly)
    assert Chem.MolToSmiles(d_gly_exported) == Chem.MolToSmiles(d_gly)
