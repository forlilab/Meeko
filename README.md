# Meeko: preparation of small molecules for AutoDock

[![API stability](https://img.shields.io/badge/stable%20API-no-orange)](https://shields.io/)
[![PyPI version fury.io](https://img.shields.io/badge/version-0.6.0--alpha-green.svg)](https://pypi.python.org/pypi/ansicolortags/)

Meeko reads an RDKit molecule object and writes a PDBQT string (or file)
for [AutoDock-Vina](https://github.com/ccsb-scripps/AutoDock-Vina)
and [AutoDock-GPU](https://github.com/ccsb-scripps/AutoDock-GPU).
It converts the docking output to RDKit molecules and
SD files, without loss of bond orders.

Meeko is developed by the [Forli lab](https://forlilab.org/) at the
[Center for Computational Structural Biology (CCSB)](https://ccsb.scripps.edu)
at [Scripps Research](https://www.scripps.edu/).


## Usage notes

Meeko does not calculate 3D coordinates or assign protonation states.
Input molecules must have explicit hydrogens.

Sampling of macrocycle conformers ([paper](https://doi.org/10.1017/qrd.2022.18))
is enabled by default.

SDF format strongly preferred over Mol2.
See
[this discussion](https://github.com/rdkit/rdkit/discussions/3647), and
[this one](https://sourceforge.net/p/rdkit/mailman/message/37668451/),
[also this](rdkit/rdkit#4061),
[and this](https://sourceforge.net/p/rdkit/mailman/message/37374678/).
and RDKit issues
[1755](https://github.com/rdkit/rdkit/issues/1755) and
[917](https://github.com/rdkit/rdkit/issues/917). So, what could go wrong?
For example, reading Mol2 files from ZINC
[led to incorrect net charge of some molecules.](https://github.com/forlilab/Meeko/issues/63)


## v0.6.0-alpha

This release aims to distribute an enhanced `mk_prepare_receptor.py`.
Some features are still being developed, hence the `-alpha` suffix in the version.
Reactive docking is not working in v0.6.0-alpha, but should be restored soon.
Documentation is also work in progress.


## API changes in v0.5

Class `MoleculePreparation` no longer has method `write_pdbqt_string()`.
Instead, `MoleculePreparation.prepare()` returns a list of `MoleculeSetup` objects
that must be passed, individually, to `PDBQTWriterLegacy.write_string()`.
```python
from meeko import MoleculePreparation
from meeko import PDBQTWriterLegacy

preparator = MoleculePreparation()
mol_setups = preparator.prepare(rdkit_molecule_3D_with_Hs)
for setup in mol_setups:
    pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)
    if is_ok:
        print(pdbqt_string, end="")
```

Argument `keep_nonpolar_hydrogens` is replaced by `merge_these_atom_types`, both in the Python
interface and for script `mk_prepare_ligand.py`.
The default is `merge_these_atom_types=("H",)`, which
merges hydrogens typed `"H"`, keeping the current default behavior.
To keep all hydrogens, set `merge_these_atom_types` to an empty
list when initializing `MoleculePreparation`, or pass no atom types
to `--merge_these_atom_types` from the command line:
```sh
mk_prepare_ligand.py -i molecule.sdf --merge_these_atom_types
``` 

## Dependencies

* Python (>=3.5)
* Numpy
* Scipy
* RDKit
* ProDy (optionally, for covalent docking)

Conda or Miniconda can install the dependencies:
```bash
conda install -c conda-forge numpy scipy rdkit
pip install prody # optional. pip recommended at http://prody.csb.pitt.edu/downloads/
```

## Installation (from PyPI)
```bash
$ pip install meeko
```
If using conda, `pip` installs the package in the active environment.

## Installation (from source)
You'll get the develop branch, which may be ahead of the latest release.
```bash
$ git clone https://github.com/forlilab/Meeko
$ cd Meeko
$ pip install .
```

Optionally include `--editable`. Changes in the original package location
take effect immediately without the need to run `pip install .` again.
```bash
$ pip install --editable .
```


## Examples using the command line scripts

#### 1. make PDBQT files
AutoDock-GPU and Vina read molecules in the PDBQT format. These can be prepared
by Meeko from SD files, or from Mol2 files, but SDF is strongly preferred.
```console
mk_prepare_ligand.py -i molecule.sdf -o molecule.pdbqt
mk_prepare_ligand.py -i multi_mol.sdf --multimol_outdir folder_for_pdbqt_files
```

#### 2. convert docking results to SDF
AutoDock-GPU and Vina write docking results in the PDBQT format. The DLG output
from AutoDock-GPU contains docked poses in PDBQT blocks.
Meeko generates RDKit molecules from PDBQT files (or strings) using the SMILES
string in the REMARK lines. The REMARK lines also have the mapping of atom indices
between SMILES and PDBQT. SD files with docked coordinates are written
from RDKit molecules.

```console
mk_export.py molecule.pdbqt -o molecule.sdf
mk_export.py vina_results.pdbqt -o vina_results.sdf
mk_export.py autodock-gpu_results.dlg -o autodock-gpu_results.sdf
```

Making RDKit molecules from SMILES is safer than guessing bond orders
from the coordinates, specially because the PDBQT lacks hydrogens bonded
to carbon. As an example, consider the following conversion, in which
OpenBabel adds an extra double bond, not because it has a bad algorithm,
but because this is a nearly impossible task.
```console
$ obabel -:"C1C=CCO1" -o pdbqt --gen3d | obabel -i pdbqt -o smi
[C]1=[C][C]=[C]O1
```

## Python tutorial

#### 1. making PDBQT strings for Vina or for AutoDock-GPU

```python
from meeko import MoleculePreparation
from meeko import PDBQTWriterLegacy
from rdkit import Chem

input_molecule_file = "example/BACE_macrocycle/BACE_4.sdf"

# there is one molecule in this SD file, this loop iterates just once
for mol in Chem.SDMolSupplier(input_molecule_file, removeHs=False):
    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(mol)
    for setup in mol_setups:
        setup.show() # optional
        pdbqt_string = PDBQTWriterLegacy.write_string(setup)
```
At this point, `pdbqt_string` can be written to a file for
docking with AutoDock-GPU or Vina, or passed directly to Vina within Python
using `set_ligand_from_string(pdbqt_string)`. For context, see
[the docs on using Vina from Python](https://autodock-vina.readthedocs.io/en/latest/docking_python.html).


#### 2. RDKit molecule from docking results

```python
from meeko import PDBQTMolecule
from meeko import RDKitMolCreate

fn = "autodock-gpu_results.dlg"
pdbqt_mol = PDBQTMolecule.from_file(fn, is_dlg=True, skip_typing=True)
rdkitmol_list = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
```
The length of `rdkitmol_list` is one if there are no sidechains and only one
ligand was docked.
If multiple ligands and/or sidechains are docked simultaneously, each will be
an individual RDKit molecule in `rdkitmol_list`.
Sidechains are truncated at the C-alpha.
Note that docking multiple
ligands simultaneously is only available in Vina, and it differs from docking
multiple ligands one after the other. Each failed creation of an RDKit molecule
for a ligand or sidechain results in a `None` in `rdkitmol_list`.

For Vina's output PDBQT files, omit `is_dlg=True`.
```python
pdbqt_mol = PDBQTMolecule.from_file("vina_results.pdbqt", skip_typing=True)
rdkitmol_list = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
```

When using Vina from Python, the output string can be passed directly.
See [the docs](https://autodock-vina.readthedocs.io/en/latest/docking_python.html)
for context on the `v` object.
```python
vina_output_string = v.poses()
pdbqt_mol = PDBQTMolecule(vina_output_string, is_dlg=True, skip_typing=True)
rdkitmol_list = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
```

#### 3. Initializing MoleculePreparation from a dictionary (or JSON)

This is useful for saving and loading configuration files with json.
```python
import json
from meeko import MoleculePreparation

mk_config = {"hydrate": True} # any arguments of MoleculePreparation.__init__
print(json.dumps(mk_config), file=open('mk_config.json', 'w'))
with open('mk_config.json') as f:
    mk_config = json.load(f)
preparator = MoleculePreparation.from_config(mk_config)
```
The command line tool `mk_prepare_ligand.py` can read the json files using
option `-c` or `--config`.


## Possibly useful configurations of MoleculePreparation

Here we create an instance of MoleculePreparation that attaches pseudo
waters to the ligand ([see paper on hydrated docking](https://pubs.acs.org/doi/abs/10.1021/jm2005145)),
keeps macrocycles rigid (generally a bad idea),
and keeps conjugated bonds and amide bonds rigid. 
By default, most amides are kept rigid, except for tertiary amides with
different substituents on the nitrogen.

```python
preparator = MoleculePreparation(
    hydrate=True,
    rigid_macrocycles=True,
    rigidify_bonds_smarts = ["C=CC=C", "[CX3](=O)[NX3]"],
    rigidify_bonds_indices = [(1, 2), (0, 2)],
)
```

The same can be done with the command line script. Note that indices of the
atoms in the SMARTS are 0-based for the Python API but
1-based for the command line script:
```console
mk_prepare_ligand.py\
    --hydrate\
    --rigid_macrocycles\
    --rigidify_bonds_smarts "C=CC=C"\
    --rigidify_bonds_indices 2 3\
    --rigidify_bonds_smarts "[CX3](=O)[NX3]"\
    --rigidify_bonds_indices 1 3
```

## Docking covalent ligands as flexible sidechains

The input ligand must be the product of the reaction and contain the
atoms of the flexible sidechain up to (and including) the C-alpha.
For example, if the ligand is an acrylamide (smiles: `C=CC(=O)N`) reacting
with a cysteine (sidechain smiles: `CCS`), then the input ligand for
Meeko must correspond to smiles `CCSCCC(=O)N`.

Meeko will align the ligand atoms that match the C-alpha and C-beta of
the protein sidechain. Options `--tether_smarts` and `--tether_smarts_indices`
define these atoms. For a cysteine, `--tether_smarts "SCC"` and
`--tether_smarts_indices 3 2` would work, although it is safer to define
a more spefic SMARTS to avoid matching the ligand more than once. The first
index (3 in this example) defines the C-alpha, and the second index defines
the C-beta. 

For the example in this repository, which is based on PDB entry 7aeh,
the following options prepare the covalently bound ligand for tethered docking:
```console
cd example/covalent_docking

mk_prepare_ligand.py\
    -i ligand_including_cys_sidechain.sdf\
    --receptor protein.pdb\
    --rec_residue ":CYS:6"\
    --tether_smarts "NC(=O)C(O)(C)SCC"\
    --tether_smarts_indices 9 8\
    -o prepared.pdbqt
```

## Reactive Docking

### 1. Prepare protein with waterkit
Follow `wk_prepare_receptor.py` instructions and run with `--pdb`.
The goal of this step is to perform essential fixes to the protein
(such as missing atoms), to add hydrogens, and to follow the Amber
naming scheme for atoms and residues, e.g., `HIE` or `HID`
instead of `HIS`.

### 2. Prepare protein pdbqt
Here, `wk.pdb` was written by waterkit.
Here, `wk.pdb` was written by waterkit. The example below will center a gridbox of specified size on the given reactive residue.

```console
   $ mk_prepare_receptor.py\
    --pdb wk.pdb\
    -o receptor.pdbqt\
    --flexres " :ARG:348"\
    --reactive_flexres " :SER:308"
    --reactive_flexres " :SER:308"\
    --box_center_on_reactive_res\
    --box_size 40 40 40  # x y z (angstroms)
```
A manual box center can be specified with `--box_center`.
Reactive parameters can also be modified:
```sh
  --r_eq_12 R_EQ_12     r_eq for reactive atoms (1-2 interaction)
  --eps_12 EPS_12       epsilon for reactive atoms (1-2 interaction)
  --r_eq_13_scaling R_EQ_13_SCALING
                        r_eq scaling for 1-3 interaction across reactive atoms
  --r_eq_14_scaling R_EQ_14_SCALING
                        r_eq scaling for 1-4 interaction across reactive atoms
```

Receptor preparation can't handle heteroatoms for the moment.
Also nucleic acids, ions, and post-translational modifications (e.g.
phosphorilation) are not supported. Only the 20 standard amino acids
can be parsed, and it is required to have Amber atom names and
hydrogens. No atoms can be missing.

### 3. Run autogrid

Make affinity maps for the `_rigid.pdbqt` part of the receptor.
Make affinity maps for the `_rigid.pdbqt` part of the receptor. `mk_prepare_receptor.py` will prepare the GPF for you.

### 4. Write ligand PDBQT
mk_prepare_ligand.py -i sufex1.sdf --reactive_smarts "S(=O)(=O)F" --reactive_smarts_idx 1 -o sufex1.pdbqt\

### 5. Configure AD-GPU for reactive docking

For reactive docking there are two options that need to be passed to AutoDock-GPU:
For reactive docking there are an additional option that needs to be passed to AutoDock-GPU:
    ```console
    --import_dpf
    ```

The `--derivtype` option, if needed, was written by `mk_prepare_receptor.py` to a file suffixed with `.derivtype`.

The filename to be passed to `--import_dpf` was written by `mk_prepare_receptor.py`
and it is suffixed with `reactive_config`.
```sh
ADGPU -I *.reactive_config -L sufex1.pdbqt -N sufex1_docked_ -F *_flex.pdbqt -C 1
