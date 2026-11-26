import sys
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from meeko.cli import mk_prepare_ligand


def _make_3d_mol(smiles, name):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 42
    rdDistGeom.EmbedMolecule(mol, params)
    mol.SetProp("_Name", name)
    return mol


def test_output_reactive_variants_not_reported_as_duplicate_names(tmp_path):
    output = mk_prepare_ligand.Output(
        multimol_output_dir=str(tmp_path),
        multimol_targz=False,
        multimol_targz_size=10000,
        multimol_prefix=None,
        redirect_stdout=False,
        output_filename=None,
        name_from_prop=None,
    )
    output("ATOM 1\n", "ligand", ("mk1",), mol_id=1)
    output("ATOM 2\n", "ligand", ("mk2",), mol_id=1)

    assert len(output.duplicate_names) == 0
    assert len(output.duplicate_filenames) == 0
    assert len(output.reactive_variant_mols) == 1
    info = output.get_duplicates_info_string()
    assert "No duplicate molecule names were found" in info
    assert "_mk<n>" in info
    assert "duplicated names" not in info
    assert "repeated names" not in info


def test_output_reactive_variants_without_explicit_mol_id(tmp_path):
    output = mk_prepare_ligand.Output(
        multimol_output_dir=str(tmp_path),
        multimol_targz=False,
        multimol_targz_size=10000,
        multimol_prefix=None,
        redirect_stdout=False,
        output_filename=None,
        name_from_prop=None,
    )
    output("ATOM 1\n", "ligand", ("mk1",))
    output("ATOM 2\n", "ligand", ("mk2",))

    assert len(output.duplicate_names) == 0
    assert len(output.duplicate_filenames) == 0
    assert len(output.reactive_variant_mols) == 1
    info = output.get_duplicates_info_string()
    assert "No duplicate molecule names were found" in info
    assert "_mk<n>" in info


def test_output_true_duplicate_names_still_warned(tmp_path):
    outdir1 = tmp_path / "dir_names"
    output1 = mk_prepare_ligand.Output(
        multimol_output_dir=str(outdir1),
        multimol_targz=False,
        multimol_targz_size=10000,
        multimol_prefix=None,
        redirect_stdout=False,
        output_filename=None,
        name_from_prop=None,
    )
    output1("ATOM 1\n", "dup_lig", ("",), mol_id=1)
    output1("ATOM 2\n", "dup_lig", ("",), mol_id=2)

    assert output1.duplicate_names == {"dup_lig"}
    assert output1.duplicate_filenames == {"dup_lig"}
    assert len(output1.reactive_variant_mols) == 0
    assert (
        output1.get_duplicates_info_string()
        == "Warning: 1 molecules with repeated names were suffixed with -again<n>"
    )

    outdir2 = tmp_path / "dir_prefix"
    output2 = mk_prepare_ligand.Output(
        multimol_output_dir=str(outdir2),
        multimol_targz=False,
        multimol_targz_size=10000,
        multimol_prefix="pref",
        redirect_stdout=False,
        output_filename=None,
        name_from_prop=None,
    )
    output2("ATOM 1\n", "dup_lig", ("",), mol_id=1)
    output2("ATOM 2\n", "dup_lig", ("",), mol_id=2)

    assert output2.duplicate_names == {"dup_lig"}
    assert len(output2.duplicate_filenames) == 0
    assert output2.get_duplicates_info_string() == "Warning: 1 molecules have duplicated names"


def test_output_duplicate_mols_each_with_reactive_variants(tmp_path):
    output = mk_prepare_ligand.Output(
        multimol_output_dir=str(tmp_path),
        multimol_targz=False,
        multimol_targz_size=10000,
        multimol_prefix=None,
        redirect_stdout=False,
        output_filename=None,
        name_from_prop=None,
    )
    output("ATOM 1\n", "dup_lig", ("mk1",), mol_id=1)
    output("ATOM 2\n", "dup_lig", ("mk2",), mol_id=1)
    output("ATOM 3\n", "dup_lig", ("mk1",), mol_id=2)
    output("ATOM 4\n", "dup_lig", ("mk2",), mol_id=2)

    assert output.duplicate_names == {"dup_lig"}
    assert output.duplicate_filenames == {"dup_lig_mk1", "dup_lig_mk2"}
    assert len(output.reactive_variant_mols) == 2
    info = output.get_duplicates_info_string()
    assert "Warning: 2 molecules with repeated names were suffixed with -again<n>" in info
    assert "_mk<n>" in info


def test_cli_reactive_multiple_matches_message(tmp_path, monkeypatch, capsys):
    sdf_path = tmp_path / "diol.sdf"
    outdir = tmp_path / "out"
    mol = _make_3d_mol("OCCO", "diol_lig")
    with Chem.SDWriter(str(sdf_path)) as writer:
        writer.write(mol)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mk_prepare_ligand.py",
            "-i",
            str(sdf_path),
            "--multimol_outdir",
            str(outdir),
            "--reactive_smarts",
            "[OX2H]",
            "--reactive_smarts_idx",
            "1",
        ],
    )
    mk_prepare_ligand.main()
    captured = capsys.readouterr()

    assert (outdir / "diol_lig_mk1.pdbqt").exists()
    assert (outdir / "diol_lig_mk2.pdbqt").exists()
    assert "No duplicate molecule names were found" in captured.out
    assert "_mk<n>" in captured.out
    assert "duplicated names" not in captured.out
    assert "repeated names" not in captured.out


def test_cli_duplicate_mol_names_still_suffixed_again(tmp_path, monkeypatch, capsys):
    sdf_path = tmp_path / "dups.sdf"
    outdir = tmp_path / "out"
    mol1 = _make_3d_mol("CCO", "same_name")
    mol2 = _make_3d_mol("CCC", "same_name")
    with Chem.SDWriter(str(sdf_path)) as writer:
        writer.write(mol1)
        writer.write(mol2)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mk_prepare_ligand.py",
            "-i",
            str(sdf_path),
            "--multimol_outdir",
            str(outdir),
        ],
    )
    mk_prepare_ligand.main()
    captured = capsys.readouterr()

    assert (outdir / "same_name.pdbqt").exists()
    assert (outdir / "same_name-again1.pdbqt").exists()
    assert (
        "Warning: 1 molecules with repeated names were suffixed with -again<n>"
        in captured.out
    )


def test_cli_single_mol_reactive_without_multimol_outdir(tmp_path, monkeypatch, capsys):
    sdf_path = tmp_path / "diol.sdf"
    mol = _make_3d_mol("OCCO", "diol_lig")
    with Chem.SDWriter(str(sdf_path)) as writer:
        writer.write(mol)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mk_prepare_ligand.py",
            "-i",
            str(sdf_path),
            "--reactive_smarts",
            "[OX2H]",
            "--reactive_smarts_idx",
            "1",
        ],
    )
    mk_prepare_ligand.main()
    captured = capsys.readouterr()

    assert (tmp_path / "diol_lig_mk1.pdbqt").exists()
    assert (tmp_path / "diol_lig_mk2.pdbqt").exists()
    assert "No duplicate molecule names were found" in captured.out
    assert "_mk<n>" in captured.out
    assert "duplicated names" not in captured.out


def test_cli_reactive_with_multimol_prefix(tmp_path, monkeypatch, capsys):
    sdf_path = tmp_path / "diol.sdf"
    outdir = tmp_path / "out"
    mol = _make_3d_mol("OCCO", "diol_lig")
    with Chem.SDWriter(str(sdf_path)) as writer:
        writer.write(mol)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mk_prepare_ligand.py",
            "-i",
            str(sdf_path),
            "--multimol_outdir",
            str(outdir),
            "--multimol_prefix",
            "pref",
            "--reactive_smarts",
            "[OX2H]",
            "--reactive_smarts_idx",
            "1",
        ],
    )
    mk_prepare_ligand.main()
    captured = capsys.readouterr()

    assert (outdir / "pref-1_mk1.pdbqt").exists()
    assert (outdir / "pref-2_mk2.pdbqt").exists()
    assert "No duplicate molecule names were found" in captured.out
    assert "_mk<n>" in captured.out
    assert "duplicated names" not in captured.out

