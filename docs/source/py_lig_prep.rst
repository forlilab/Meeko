Ligand preparation in Python
============================


Creating a PDBQT string from an RDKit molecule
----------------------------------------------
.. code-block:: python

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

At this point, ``pdbqt_string`` can be written to a file for
docking with AutoDock-GPU or Vina, or passed directly to Vina within Python
using `set_ligand_from_string(pdbqt_string)`. For context, see
[the docs on using Vina from Python](https://autodock-vina.readthedocs.io/en/latest/docking_python.html).

One advantage of this approach is that input PDBQT files are not written to the filesystem.
The PDBQT format is lossy, because it lacks bond orders and non-polar hydrogens,
making it a poor choice to store molecular information.

Another advantage is to write custom workflows entirely from Python without external
calls to ``mk_prepare_ligand.py``.
