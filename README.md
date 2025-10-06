# Meeko: interface for AutoDock

[![API stability](https://img.shields.io/badge/stable%20API-no-orange)](https://shields.io/)
[![PyPI version fury.io](https://img.shields.io/badge/version-0.6.1-green.svg)](https://pypi.python.org/pypi/meeko/)
[![Documentation Status](https://readthedocs.org/projects/meeko/badge/?version=release)](https://meeko.readthedocs.io/en/release/?badge=release)

Meeko prepares the input for AutoDock and processes its output.
It is developed alongside AutoDock-GPU and AutoDock-Vina.
Meeko parameterizes both small organic molecules (ligands) and proteins
and nucleic acids (receptors).

Meeko is developed by the [Forli lab](https://forlilab.org/) at the
[Center for Computational Structural Biology (CCSB)](https://ccsb.scripps.edu)
at [Scripps Research](https://www.scripps.edu/).


## Documentation

The docs are hosted on [meeko.readthedocs.io](https://meeko.readthedocs.io/en/release)


## Reporting bugs

Please check if a similar bug has been reported and, if not, [open an issue](https://github.com/forlilab/Meeko/issues).


## Installation

Visit the docs for a more complete description. One option is conda or mamba:

```bash
micromamba install meeko
```

or from PyPI:

```bash
pip install meeko
```

## Usage

Meeko exposes a Python API to enable scripting. Here we share very minimal examples
using the command line scripts just to give context.
Please visit the [meeko.readthedocs.io](https://meeko.readthedocs.io/en/release) for more information.

Parameterizing a ligand and writing a PDBQT file:
```bash
mk_prepare_ligand.py -i molecule.sdf -o molecule.pdbqt
```

Parameterizing a receptor with a flexible sidechain and writing a PDBQT file
as well as a JSON file that stores the entire receptor datastructure. In this
example, the `-o` option sets the output base name, `-j` triggers writing the
.json file, `-p` triggers writting the .pdbqt file, and `-f` makes residue
42 in chain A flexible.

```bash
mk_prepare_receptor.py -i nucleic_acid.cif -o my_receptor -j -p -f A:42
```

Finally, converting docking results to SDF for the ligand, and PDB for the
receptor with updated sidechain positions:

```bash
mk_export.py vina_results.pdbqt -j my_receptor.json -s lig_docked.sdf -p rec_docked.pdb
```

### Box-only grid generation (no receptor input)

When you only need a docking box (for Vina or visualization) you can now write **just** the box information without providing a receptor file.

**What it does**
- Writes a Vina-style box file (`.box.txt`) and a PDB to visualize the box (`.box.pdb`).
- No receptor input required.

**How to use**
- Choose one of the following to define the box:
  1) `--box_center X Y Z` **and** `--box_size X Y Z`, **or**
  2) `--box_enveloping <ligand file>` **and** `--padding <Å>`
- Provide either `-v/--write_vina_box` (filename optional) or `-o/--output_basename` to set output names.

**Examples**

Explicit center & size (no receptor):
```bash
mk_prepare_receptor.py   --box_only   --box_center 10 20 30   --box_size 22 24 26   -o my_target -v
```
This writes:
- `my_target.box.txt` (Vina-style config with center/size)
- `my_target.box.pdb` (box visualization)

Envelope a ligand file (no receptor):
```bash
mk_prepare_receptor.py   --box_only   --box_enveloping ligand.sdf   --padding 6.0   -v my_box.txt
```
This writes:
- `my_box.txt`
- `my_box.pdb`

**Notes**
- `--box_only` **cannot** be combined with receptor inputs (`-i/--read_with_prody`, `--read_pdb`, `--read_pqr`).
- `--write_gpf` is **not** supported in `--box_only` mode.
- `--box_center_off_reactive_res` requires a receptor and is **not** available in `--box_only` mode.
