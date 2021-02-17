# Meeko

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version fury.io](https://img.shields.io/badge/version-0.1-green.svg)](https://pypi.python.org/pypi/ansicolortags/)

The Python package meeko is a new type of package developped and maintained by the Forli lab also at the [Center for Computational Structural Biology (CCSB)](https://ccsb.scripps.edu). It provides tools covering other docking aspects not handled by the ADFR software suite. This package provides addtionnal tools for the following docking protocols:

* Hydrated docking
* Macrocycles


## Prerequisites

You need, at a minimum (requirements):
* Python (>=3.5)
* Numpy
* Openbabel (>=3)

## Installation (from source)

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

preparator = MoleculePreparation(merge_hydrogens=True, macrocycle=True, hydrate=True)
preparator.prepare(mol)
preparator.show_setup()

output_pdbqt_file = "test_macrocycle_hydrate.pdbqt"
preparator.write_pdbqt_file(output_pdbqt_file)
```
