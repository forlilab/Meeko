Meeko
=====

Preparation of molecules for AutoDock
-------------------------------------

Meeko assigns parameters to small organic molecules, to proteins, and to
nucleic acids. It prepares the input needed to run AutoDock from SD files
for small organic molecules, and from PDB or MMCIF files for proteins and
nucleic acids. It processes the output of AutoDock back into these formats.

Meeko replaces MGLTools for preparing molecules.


What Meeko does **not** do
--------------------------

Meeko doesn't calculate 3D positions or assign protonation states.

Small organic molecules
-----------------------

Assign atom types, set rotatable bonds, assign partial charges, etc.

Proteins and nucleic acids
--------------------------

Protonation variants (e.g. HIE/HID) can be manually set.

Interface with RDKit
--------------------

The Python API uses RDKit as input and output for small
organic molecules. This makes it easier to write custom programs
that integrate AutoDock with other software that also uses RDKit.

.. toctree::
   :maxdepth: 2
   :caption: MANUAL

   installation

