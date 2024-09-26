Meeko
=====

Preparation of molecules for AutoDock
-------------------------------------

Meeko assigns parameters to small organic molecules (a.k.a. ligands), and to
biological polymers (proteins and nucleic acids).
This includes assigning atom types, partial charges, setting
bonds as rotatable or fixed, and making receptor sidechains flexible.
It does not calculate 3D positions or assign protonation states of ligands
but for receptors it allows the user to manually choose the protonation
variant of tritable amino acids.

Meeko write the input PDBQT files for AutoDock-Vina and AutoDock-GPU, and it
It also converts the output files from docking, which are PDBQT for Vina and
DLG for AutoDock-GPU, into useful file formats that other software besides
AutoDock can read: SDF for ligands and PDB for receptor.

AutoDock ecosystem
------------------

Meeko by itself it's not very useful. It is part of a larger collection of
tools to computationally dock ligands onto receptors.

 * interface with RDKit by desing, for Python scripting
 * AutoDock-Vina
 * AutoDOck-GPU
 * Ringtail
 * Molscrub
 * RSD3 website

Comparison to MGLTools
----------------------

Meeko superseeds the preparation scripts from MGLTools. List advantages here (**TODO**).


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting started

   installation
   tutorials
   about

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Ligand preparation

   cli_lig_prep
   In Python <py_lig_prep>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Receptor preparation

   cli_rec_prep

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Exporting results

   cli_export_result
   In Python <py_export_result>



