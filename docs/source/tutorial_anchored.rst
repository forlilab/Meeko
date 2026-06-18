.. _tutorial_anchored:

================
Anchored Docking
================

This tutorial describes docking while anchoring specific ligand atoms to
target coordinates in space. The ligand atoms can be specified using SMARTS patterns
as described in step 2. The defined ligand atoms will be penalized by large energies
if they deviate from the target coordinates. This works only with autodock_gpu.

Knowledge of the basic docking tutorial is assumed. Only the steps
that differ from basic docking are written here.


1. Receptor preparation
-----------------------

Prepare receptor as for basic docking and make sure to use option ``-g`` to
write the input file for autogrid (``.gpf`` extension).


2. Ligand preparation
---------------------

With ``mk_prepare_ligand.py`` Use option `--add_atom_types` to type phenol
oxygens that are ortho to a carbonyl with atom type `OX`. The SMARTS
`"[#1][O]ccC=O"` matches such phenol oxygens. In this SMARTS, the second atom
is the one we want to type, hence we use `"IDX": [2]`.

.. code-block:: bash

    mk_prepare_ligand.py --add_atom_types '{"smarts": "[#1][O]ccC=O", "IDX": [2], "atype": "OX"}' -i ligand.sdf -o ligand.pdbqt 


If more than one atom are subjected to the anchoring potential, it will likely be impossible to fit more than a single
atom near the target coordinates. Since the docking scores will force all
matching atoms to be as close as possible to the target coordinates,
unreasonable binding modes are likely. In ligand.sdf there are two
phenol oxygens, but only one is ortho to a carbonyl, so the provided
SMARTS matches only one atom. The number of matched atoms can be
verified in the ligand PDBQT file written in step 3.


3. Run autogrid
---------------

The ``.gpf`` file was written by ``mk_prepare_receptor.py`` with ``-g`` option.
Here's an example call to autogrid:

.. code-block:: bash

    autogrid4 -p rec.gpf -l rec.glg


4. Create a new map with a penalty
----------------------------------

The penalty is a high score (kcal/mol) everywhere except near the target coordinates. 
By default, a 1.2 A radius is used for the un-penalized volume.

.. code-block:: bash

    python addbias.py -i rec.OA.map -o rec.OX.map -x -3 11 25.2


5. Add OX map to FLD file
-------------------------

.. code-block:: bash

    python insert_type_in_fld.py rec.maps.fld --newtype OX


9. Dock
-------

.. code-block:: bash

    autodock_gpu -L ligand.pdbqt -M rec.maps.fld -T OX=OA
    mk_export.py ligand.dlg -o ligand_docked.sdf
