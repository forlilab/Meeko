mk_prepare_ligand.py
====================

Command line tool to prepare small organic molecules.

Write PDBQT files
-----------------

AutoDock-GPU and Vina read molecules in the PDBQT format. These can be prepared
by Meeko from SD files, or from Mol2 files, but SDF is strongly preferred.

.. code-block:: bash

    mk_prepare_ligand.py -i molecule.sdf -o molecule.pdbqt
    mk_prepare_ligand.py -i multi_mol.sdf --multimol_outdir folder_for_pdbqt_files

