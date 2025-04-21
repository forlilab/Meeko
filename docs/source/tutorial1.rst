.. _tutorial1:

Basic Docking 
-------------------------------------

This tutorial provides practice examples and a step-by-step guide for the two basic procedures, **Ligand Preparation** and **Receptor Preparation**, with Meeko for molecular docking and virtual screening with `AutoDock Vina <https://github.com/ccsb-scripps/AutoDock-Vina>`_ and `AutoDock-GPU <https://github.com/ccsb-scripps/AutoDock-GPU>`_. It is based on, but not a full version of the tutorial materials in `Forlilab tutorials <https://github.com/forlilab/tutorials>`_. 

Prerequisites and Environment Setup
===================================

Create a new virtual environment (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   micromamba create -c conda-forge -n meeko_tutorial_py39 python=3.9 -y
   micromamba activate meeko_tutorial_py39         

In this tutorial, we will use ``micromamba`` as the example package manager. Visit `this official guide  <https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html>`_ for a quick install and setup of micromamba. There are many equivalent ways to manage Python packages, such as ``conda`` and ``mamba``. You can easily adapt the commands to your preferred tool, as the syntax is largely compatible across these package managers. 

Install the required Python packages through ``conda-forge``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   micromamba install -c conda-forge meeko numpy scipy rdkit gemmi vina -y

Install the additional packages and data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- (Python package) Molscrub
Git Repository: `Forlilab Molscrub <https://github.com/forlilab/molscrub>`_

.. code-block:: bash

   pip install molscrub

- (Example files for this tutorial) Meeko/example/tutorial1
Originally from `Forlilab tutorials <https://github.com/forlilab/tutorials>`_

.. code-block:: bash

  git clone --branch docwork --depth=1 --filter=tree:0 https://github.com/forlilab/Meeko.git
  cd Meeko; git sparse-checkout set --no-cone example; git checkout; cd ..

Ligand Preparation
==================

Ligand Preparation is the process that generates ligand input files for docking calculation and virtual screening. At present, AutoDock Vina and AutoDock-GPU need the ligand input files in the PDBQT format. In this example, we will use ``mk_prepare_ligand.py``, a command-line script in Meeko, to prepare such ligand PDBQT files. 

Prepare a Single Ligand from a Smiles String
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Imatinib <https://pubchem.ncbi.nlm.nih.gov/compound/Imatinib>`_ is a small-molecule drug. You can find the SMILES string for Imatinib from various reliable chemical databases and resources, including but not limited to `PubChem <https://pubchem.ncbi.nlm.nih.gov/>`_ and `DrugBank <https://go.drugbank.com/>`_. 

``scrub.py`` is a command-line script in Molscrub that generates 3D conformers of protomers and tautomers for given small molecules at a specified (range of) pH. Given a pH range of 5 to 9, the output protomers will include those which make up no less than 1% of the total population at pH = 7. Based on the reference pKa values, the amine nitrogens and the pyridine nitrogen will be considered for acid/base enumeration. With the ``meeko_tutorial_py39`` micromamba environment active, run ``scrub.py`` to generate 3D conformers of Imatinib from the SMILES string. 

.. code-block:: bash

    smiles_string="CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"
    scrub.py $smiles_string -o imatinib.sdf --skip_tautomers --ph_low 5 --ph_high 9

The output file ``imatinib.sdf`` will contain two protomers of Imatinib, one with a neutral pyridine group and the other with a (+1) pyridinium group. All of the aliphatic ammonium nitrogens will be protonated. 

.. code-block:: bash

    Scrub completed.
    Summary of what happened:
    Input molecules supplied: 1
    mols processed: 1, skipped by rdkit: 0, failed: 0
    nr isomers (tautomers and acid/base conjugates): 2 (avg. 2.000 per mol)
    nr conformers:  2 (avg. 1.000 per isomer, 2.000 per mol)

In case there are multiple molecules in the SDF file, ``mk_prepare_ligand.py`` needs to know the prefix of filenames (by ``--multimol_prefix``) or alternatively where to output (by ``--multimol_outdir``) the multiple PDBQT files. Here, we will give the PDBQT files a prefix ``imatinib_protomer`` in the names. The output PDBQT files will be ``imatinib_protomer-1.pdbqt`` and ``imatinib_protomer-2.pdbqt``. 

.. code-block:: bash

    mk_prepare_ligand.py -i imatinib.sdf --multimol_prefix imatinib_protomer


Batched Ligand Preparation from a ``.smi`` File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For virtual screening, it is possible to prepare ligands in batch mode from a ``.smi`` File. There is one such example file at ``Meeko/example/tutorial1/input_files/mols.smi``. Follow the example commands to process ``mols.smi``: 

.. code-block:: bash

    smi_file="Meeko/example/tutorial1/input_files/mols.smi"
    scrub.py $smi_file -o mols.sdf

At the end of the execution, the expected standard output will tell you the total number of isomers written to the multi-molecule SDF file ``mols.sdf``. This will help you estimate the expected file size and system requirements beforehand. 

.. code-block:: bash

    Scrub completed.
    Summary of what happened:
    Input molecules supplied: 491
    mols processed: 491, skipped by rdkit: 0, failed: 0
    nr isomers (tautomers and acid/base conjugates): 741 (avg. 1.509 per mol)
    nr conformers:  741 (avg. 1.000 per isomer, 1.509 per mol)

For ``mols.sdf``, we will run ``mk_prepare_ligand.py`` with ``--multimol_prefix mols_pdbqt``, a directory to be created to hold the ligand PDBQT files. If you expect a large number of isomers (potentially millions), consider writing to a temporary directory or scratch space to manage storage efficiently. 

.. code-block:: bash

    mk_prepare_ligand.py -i mols.sdf --multimol_outdir mols_pdbqt

Receptor Preparation
====================

Receptor Preparation is the process that generates receptor input files for docking calculation and virtual screening. It typically begins with a PDB file of a biomacromolecule system, with or without coordinates of explicit hydrogens. At present, AutoDock Vina and AutoDock-GPU may require different types of files as receptor inputs. ``mk_prepare_receptor.py`` is the command-line script in Meeko that is designed to handle the different situations. 

For AutoDock-Vina
~~~~~~~~~~~~~~~~~

Docking with AutoDock-Vina requires the following receptor input files: 

- Receptor PDBQT file
- (Optional) a TXT file that contains the box specifications, which can be reused as the config file for Vina

Starting from a provided PDB file at ``Meeko/example/tutorial1/input_files/1iep_protein.pdb``, the generation of a Receptor PDBQT file is very straightforward: 

.. code-block:: bash

    pdb_file="Meeko/example/tutorial1/input_files/1iep_protein.pdb"
    mk_prepare_receptor.py --read_pdb $pdb_file -o rec_1iep -p 

Here, we use ``-o`` to set the basename of the output files to ``rec_1iep`` with request ``-p``. The execution will generate only the receptor PDBQT file, ``rec_1iep.pdbqt``. 

Note that ``--read_pdb``, which uses the PDB parser in RDKit, is not the only way for ``mk_prepare_receptor.py`` to parse a receptor PDB file. The alternate is ``-i`` (short for ``--read_with_prody``) and it requires ProDy as an additional dependency. If you wish to use the ProDy parser, run ``pip install prody`` to install ProDy. 

To generate the TXT file that has the box dimension, we must find a way to define the wanted docking box. In this example, we will use a provided PDB file of ligand Imatinib at ``Meeko/example/tutorial1/input_files/xray-imatinib.pdb`` that has been aligned to the expected binding site of the provided receptor PDB file. 

.. code-block:: bash

    pdb_file="Meeko/example/tutorial1/input_files/1iep_protein.pdb"
    lig_file="Meeko/example/tutorial1/input_files/xray-imatinib.pdb"
    mk_prepare_receptor.py --read_pdb $pdb_file -o rec_1iep -p -v \
    --box_enveloping $lig_file --padding 5

Here, we add the ``-v`` to request the Vina-style box files to be generated along with the receptor PDBQT files. To define the box, we are using the combination of ``--box_enveloping`` and ``--padding``, which is to set the center of the box by the given object, and the size of the box by a constant padding in each dimension around the given object. Note that this is not the only way to define the box. Read the help message printed from ``mk_prepare_receptor.py -h`` to learn about other combinations. 

At the end of the execution with ``-p -v``, the expected standard output will be: 

.. code-block:: bash

    Files written:
      rec_1iep.pdbqt <-- static (i.e., rigid) receptor input file
    rec_1iep.box.txt <-- Vina-style box dimension file
    rec_1iep.box.pdb <-- PDB file to visualize the grid box

.. _receptor_preparation_for_vina_with_adf4sf:

For AutoDock-Vina (and with AutoDock4 Scoring Function)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use the AutoDock4 Scoring Function in AutoDock-Vina, an additional step needs to be taken to compute the grid maps prior to the docking calculation. At present, this is only possible with AutoGrid; therefore, a Grid Parameter File (GPF) is required. Using ``mk_prepare_receptor.py`` option ``-g``, such GPF file can be generated in the same step along with the receptor PDBQT file and the box dimension files. Here's an example: 

.. code-block:: bash

    pdb_file="Meeko/example/tutorial1/input_files/1iep_protein.pdb"
    lig_file="Meeko/example/tutorial1/input_files/xray-imatinib.pdb"
    mk_prepare_receptor.py --read_pdb $pdb_file -o rec_1iep -p -v -g \
    --box_enveloping $lig_file --padding 5

At the end of the execution with ``-p -v -g``, the expected standard output is now: 

.. code-block:: bash

    Files written:
                rec_1iep.pdbqt <-- static (i.e., rigid) receptor input file
    boron-silicon-atom_par.dat <-- atomic parameters for B and Si (for autogrid)
                  rec_1iep.gpf <-- autogrid input file
              rec_1iep.box.txt <-- Vina-style box dimension file
              rec_1iep.box.pdb <-- PDB file to visualize the grid box

To compute the grid maps, the GPF file (``rec_1iep.gpf``) will be the input command file for AutoGrid. The receptor PDBQT file (``rec_1iep.pdbqt``) and the additional parameter file (``boron-silicon-atom_par.dat``) need to be in the same directory from which AutoGrid is run. 

For AutoDock-GPU
~~~~~~~~~~~~~~~~

At present, AutoDock-GPU also needs the pre-computed grid maps from AutoGrid. Therefore, Receptor Preparation for docking calculations with AutoDock-GPU is similar to preparation in the previous section :ref:`receptor_preparation_for_vina_with_adf4sf`. But in this case, we can drop the ``-v`` option as the Vina-style box definition TXT file is no longer needed for AutoDock-GPU. 

Below is the sample command: 

.. code-block:: bash

    pdb_file="Meeko/example/tutorial1/input_files/1iep_protein.pdb"
    lig_file="Meeko/example/tutorial1/input_files/xray-imatinib.pdb"
    mk_prepare_receptor.py --read_pdb $pdb_file -o rec_1iep -p -g \
    --box_enveloping $lig_file --padding 5

And the expected standard output will be: 

.. code-block:: bash

    Files written:
                rec_1iep.pdbqt <-- static (i.e., rigid) receptor input file
    boron-silicon-atom_par.dat <-- atomic parameters for B and Si (for autogrid)
                  rec_1iep.gpf <-- autogrid input file
              rec_1iep.box.pdb <-- PDB file to visualize the grid box

Save a Receptor JSON File for Docking with Flexible and/or Reactive Residues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Docking with flexible and/or reactive residues may require more files than basic docking, and ``mk_prepare_receptor.py`` is able to prepare those simultaneously when creating the receptor PDBQT file. The detailed procedure for Reactive Docking can be found in :ref:`tutorial2`. Here, we will use a different PDB file at ``Meeko/example/tutorial1/input_files/2hzn_protein.pdb`` to showcase a simple docking preparation with flexible sidechains: 

.. code-block:: bash

    pdb_file="Meeko/example/tutorial1/input_files/2hzn_protein.pdb"
    lig_file="Meeko/example/tutorial1/input_files/xray-imatinib.pdb"
    mk_prepare_receptor.py --read_pdb $pdb_file -o rec_2hzn -p -v -g -j \
    --box_enveloping $lig_file --padding 5 \
    -f A:286,359 --allow_bad_res

Note that several additional arguments are introduced for this particular receptor structure and for flexible docking. First off, ``-f A:286,359`` specifies that we are making two residues flexible, which are Glu286 and Phe359 in chain A of the receptor PDB file ``2hzn_protein.pdb``. Moreover, we add the ``--allow_bad_res`` so that partially resolved residues in the input PDB file can be ignored. Finally, we make the request ``-j`` to not only write the typical input files for docking calculations, but also a receptor JSON file. This receptor JSON file may be used in future steps in order to export the full receptor structure with updated sidechain conformations from the docking output. 

With that, the standard output and the list of generated files from ``mk_prepare_receptor.py`` will be: 

.. code-block:: bash

    - Template matching failed for: ['A:238', 'A:262', 'A:263', 'A:264', 'A:281', 'A:356', 'A:462', 'A:466', 'A:502'] Ignored due to allow_bad_res.

    Flexible residues:
    chain resnum is_reactive reactive_atom
        A    359       False              
        A    286       False              

    Files written:
                 rec_2hzn.json <-- parameterized receptor
           rec_2hzn_flex.pdbqt <-- flexible receptor input file
          rec_2hzn_rigid.pdbqt <-- static (i.e., rigid) receptor input file
    boron-silicon-atom_par.dat <-- atomic parameters for B and Si (for autogrid)
            rec_2hzn_rigid.gpf <-- autogrid input file
              rec_2hzn.box.txt <-- Vina-style box dimension file
              rec_2hzn.box.pdb <-- PDB file to visualize the grid box

Export Poses from Docking
=========================

From AutoDock-Vina
~~~~~~~~~~~~~~~~~~

With AutoDock-Vina, The required files (generated from the previous steps) and the command to run a basic docking calculation of a single ligand is as follows: 

.. code-block:: bash

    lig_pdbqt="imatinib_protomer-1.pdbqt"
    rec_pdbqt="rec_1iep.pdbqt"
    config_txt="rec_1iep.box.txt"
    ./vina --ligand $lig_pdbqt --receptor $rec_pdbqt --config $config_txt

Without giving Vina a custom output name, the default output PDBQT file will be named ``imatinib_protomer-1_out.pdbqt``. Using the Smiles and mapping information stored in the REMARKS section of the PDBQT file, ``mk_export.py`` is able to reconstruct the all-atom structures of the docked ligand and export the poses to a SDF file, ``imatinib_protomer-1_vina_out.sdf``, which includes the reconstructed coordinates of all hydrogen atoms: 

.. code-block:: bash

    docked_pdbqt="imatinib_protomer-1_out.pdbqt"
    mk_export.py $docked_pdbqt -s imatinib_protomer-1_vina_out.sdf

From AutoDock-GPU
~~~~~~~~~~~~~~~~~

With AutoDock-GPU, the required files (generated from the previous steps) and the command to run a basic docking calculation of a single ligand is as follows: 

.. code-block:: bash

    lig_name="imatinib_protomer-1"
    lig_pdbqt="${lig_name}.pdbqt"
    rec_prefix="rec_1iep"
    rec_map_fld="${rec_prefix}.maps.fld"
    ./adgpu --lfile $lig_pdbqt --ffile $rec_map_fld --resnam $lig_name

With that, the output DLG file will be named ``imatinib_protomer-1.dlg``. Similarly, ``mk_export.py`` is able to reconstruct the atomistic structures of the docked ligand and export the poses to a SDF file as follows: 

.. code-block:: bash

    docked_dlg="imatinib_protomer-1.dlg"
    mk_export.py $docked_dlg -s imatinib_protomer-1_adgpu_out.sdf

Note that by default, only the cluster leads will be exported to the SDF file. To export all generated poses in the DLG file, add the ``--all_dlg_poses`` option when exporting the poses. 

From Flexible Receptor Docking (with AutoDock-Vina)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With AutoDock-Vina, the required files (generated from the previous steps) and the command to run a flexible docking calculation of a single ligand is as follows: 

.. code-block:: bash

    lig_name="imatinib_protomer-1"
    lig_pdbqt="${lig_name}.pdbqt"
    rec_prefix="rec_2hzn"
    flexres_pdbqt="${rec_prefix}_flex.pdbqt"
    rec_pdbqt="${rec_prefix}_rigid.pdbqt"
    config_txt="${rec_prefix}.box.txt"
    ./vina --ligand $lig_pdbqt --flex $flexres_pdbqt --receptor $rec_pdbqt --config $config_txt --out ${lig_name}_flexres.pdbqt

With that, the output PDBQT file will be named ``imatinib_protomer-1_flexres.pdbqt``. If given the receptor JSON file (``rec_2hzn.json``) generated when the other receptor files were created, ``mk_export.py`` is able to reconstruct the atomistic structures of the full receptor and export the updated models to a multi-model PDB file (``imatinib_protomer-1_flexres_vina_out.pdb``) with the following command: 

.. code-block:: bash

    rec_json="rec_2hzn.json"
    docked_pdbqt="imatinib_protomer-1_flexres.pdbqt"
    mk_export.py $docked_pdbqt -j $rec_json -p imatinib_protomer-1_flexres_vina_out.pdb

From Flexible Receptor Docking (with AD-GPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With AutoDock-GPU, the required files (generated from the previous steps) and the command to run a flexible docking calculation of a single ligand is as follows: 

.. code-block:: bash

    lig_name="imatinib_protomer-1"
    lig_pdbqt="${lig_name}.pdbqt"
    rec_prefix="rec_2hzn"
    flexres_pdbqt="${rec_prefix}_flex.pdbqt"
    rec_map_fld="${rec_prefix}_rigid.maps.fld"
    ./adgpu --lfile $lig_pdbqt --flexres $flexres_pdbqt --ffile $rec_map_fld --resnam ${lig_name}_flexres

With that, the output DLG file will be named ``imatinib_protomer-1_flexres.dlg``. Again, if given the receptor JSON file (``rec_2hzn.json``) generated when the other receptor files were created, ``mk_export.py`` is able to export the updated models to a PDB file (``imatinib_protomer-1_flexres_adgpu_out.pdb``): 

.. code-block:: bash

    rec_json="rec_2hzn.json"
    docked_dlg="imatinib_protomer-1_flexres.dlg"
    mk_export.py $docked_dlg -j $rec_json -p imatinib_protomer-1_flexres_adgpu_out.pdb

At present, all docking poses will be exported, whether they are cluster leads or not. 

Processing the Screening (Batch Docking) Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To process results from Screening (Batch Docking), please use the `Ringtail <https://github.com/forlilab/Ringtail>`_ package for SQL-based data management, streamlined analysis and filtering. The documentation of Ringtail can be found `here <https://ringtail.readthedocs.io/en/latest/>`_. 

What's Next?
^^^^^^^^^^^^

Now that you've completed this tutorial, you're ready to move on to :ref:`tutorial2` and :ref:`tutorial3` where we dive deeper into more advanced docking methods: reactive docking and tethered docking.
