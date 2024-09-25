Exporting docking results in Python
===================================

.. code-block:: python

    from meeko import PDBQTMolecule
    from meeko import RDKitMolCreate
    
    fn = "autodock-gpu_results.dlg"
    pdbqt_mol = PDBQTMolecule.from_file(fn, is_dlg=True, skip_typing=True)
    rdkitmol_list = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)

The length of ``rdkitmol_list`` is one if there are no sidechains and only one
ligand was docked.
If multiple ligands and/or sidechains are docked simultaneously, each will be
an individual RDKit molecule in ``rdkitmol_list``.
Sidechains are truncated at the C-alpha.
Note that docking multiple
ligands simultaneously is only available in Vina, and it differs from docking
multiple ligands one after the other. Each failed creation of an RDKit molecule
for a ligand or sidechain results in a ``None`` in ``rdkitmol_list``.

For Vina's output PDBQT files, omit ``is_dlg=True``.

.. code-block:: python

    pdbqt_mol = PDBQTMolecule.from_file("vina_results.pdbqt", skip_typing=True)
    rdkitmol_list = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)

When using Vina from Python, the output string can be passed directly.
See [the docs](https://autodock-vina.readthedocs.io/en/latest/docking_python.html)
for context on the `v` object.

.. code-block:: python

    vina_output_string = v.poses()
    pdbqt_mol = PDBQTMolecule(vina_output_string, is_dlg=True, skip_typing=True)
    rdkitmol_list = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
