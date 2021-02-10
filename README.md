# Meeko
A new package to prepare small molecules for docking

## Prerequisites

You need, at a minimum (requirements):
* Python (>=3.6)
* Numpy
* Openbabel

## Installation

```bash
$ git clone https://github.com/ccsb-scripps/Meeko
$ cd Meeko
$ python setup.py build install
```

## Quick tutorial

```python
from meeko import MoleculePreparation
from meeko import obutils


input_molecule_file = 'example/BACE_macrocycle/BACE_4.mol2'
mol = obutils.load_molecule_from_file(input_molecule_file)

preparator = MoleculePreparation(merge_hydrogens=True, macrocycle=True, hydrate=True, amide_rigid=True)
preparator.prepare(mol)
preparator.show_setup()

output_pdbqt_file = "test_macrocycle_hydrate.pdbqt"
preparator.write_pdbqt_file(output_pdbqt_file)
```
