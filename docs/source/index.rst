Meeko: interface for AutoDock
=============================

Parameterization of molecules
-----------------------------

Meeko assigns parameters to small organic molecules, referred to as ligands,
and to proteins and nucleic acids, referred to as receptors.
Parameterization includes assigning atom types, partial charges, setting
bonds as rotatable, and making receptor sidechains flexible.

Prepare input and convert output
--------------------------------

Meeko writes the input PDBQT files (or strings in Python) for AutoDock-Vina
and AutoDock-GPU, and exports docking results in SDF format for ligands and
in PDB format for receptor.

Python API
----------

Meeko is written in Python and exposes functions and classes that operate on
RDKit molecules for the ligands, leveraging RDKit's popularity to facilitate
integration with external software that also interfaces with RDKit.

Command line scripts
--------------------

There are two scripts to write the input files for docking,
``mk_prepare_ligand.py`` and ``mk_prepare_receptor.py``, and one script to
convert docking output files  ``mk_export.py``.

Running a docking
-----------------

To run a docking, more packages are required besides Meeko. At a minimum,
either Vina or AutoDock-GPU are needed to run the actual docking.

.. grid:: 3

    .. grid-item-card:: AutoDock-GPU
        :link: https://github.com/ccsb-scripps/AutoDock-GPU

        Docking for GPUs. Implements the AutoDock4.2 scoring function.
        Command line executable only.

    .. grid-item-card:: AutoDock-Vina
        :link: https://autodock-vina.readthedocs.io/

        Docking on CPUs.
        Implements Vina and AutoDock4.2 scoring functions.
        Has a Python API and command line executable.

    .. grid-item-card:: Ringtail
        :link: https://github.com/forlilab/ringtail

        Store and analyze virtual screening with SQLite.
        Has a Python API and command line scripts.


Check the tutorials page to learn about using meeko with these other packages
to run molecular docking and virtual screening.


.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Getting started

   installation
   colab_examples
   tutorials

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Ligand preparation

   Overview <lig_overview>
   Basic usage <lig_prep_basic>
   Advanced usage <lig_prep_advanced>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Receptor preparation

   Overview <rec_overview>
   Info on templates  <py_build_temp>
   Command line usage <cli_rec_prep>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Exporting results

   Usage <export_usage>

.. toctree::
   :maxdepth: 2
   :caption: CLI Reference

   meeko.mk_prepare_receptor
   meeko.mk_prepare_ligand
   meeko.mk_export
   ...

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   meeko.molecule_pdbqt
   meeko.rdkit_mol_create
   meeko.molsetup
   meeko.preparation
   meeko.polymer
   meeko.chemtempgen
   ...