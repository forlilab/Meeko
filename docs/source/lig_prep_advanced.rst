Advanced ligand preparation
===========================

Parameterization of RDKit molecules is controlled by several parameters
that can be used when creating an instance of ``MoleculePreparation``.
Many of such parameters are exposed to the command line script, 
``mk_prepare_ligand.py``. Here, we cover the most relevant options, and
how to use them both from command line and Python scripts. Familiarity with
Python and with the command line is required.

Configuring MoleculePreparation
-------------------------------

From Python
^^^^^^^^^^^

An instance of ``MoleculePreparation`` is configured by passing
any number of optional parameters during initialization:

.. code-block:: python

   from meeko import MoleculePreparation

   mk_prep = MoleculePreparation(
       merge_these_atom_types=("H"),
       charge_model="gasteiger",
   )

Here we are passing two parameters, one that will merge atoms typed "H" into
their parent atom, and another to use Gasteiger charges. Both are the default
parameter, so we would have gotten the same configuration by passing zero
parameters: ``mk_prep = MoleculePreparation()``.

Parameters can be in a dictionary and use the ``from_config`` constructor:

.. code-block:: python

   config_dict = {"merge_these_atom_types": (), "charge_model": "gasteiger"}
   mk_prep = MoleculePreapration.form_config(config_dict)

The configuration dictionary can be written to a JSON file:

.. code-block:: python

   import json

   with open("my_config.json", "w") as f:
       json.dump(config_dict, f)


From command line
^^^^^^^^^^^^^^^^^

The command line script ``mk_prepare_ligand.py`` can read a JSON file
using the ``-c`` or ``--config_file`` option. The contents of a JSON file
corresponding to the Python example above is:

.. code-block:: json

   {"merge_these_atom_types": ["H"], "charge_model": "gasteiger"}

Passing the filename to the script with option ``-c``:

.. code-block:: bash

   mk_prepare_ligand.py -i mol.sdf -c my_config.json

Here, ``mol.sdf`` is some file in the current working directory.

Alternatively, each parameter can be passed directly:

.. code-block:: bash

   mk_prepare_ligand.py -i mol.sdf --charge_model gasteiger --merge_these_atom_types H

Parameters passed directly as command line arguments override those in the
configuration file. Exceptionally, option `add_atom_types` extends the
parameters in the configuration file.


Merging hydrogens (or not)
--------------------------

By default, atoms of type ``H`` are merged. Merging consists of adding
their partial charge to the parent atom, and setting the ``is_ignore``
attribute to ``True`` in the instance of ``MoleculeSetup``, which in turn
prevents the atom from being written to PDBQT. Since the atoms are merged
based on their atom type, changing the atom typing can change the set of
atoms that is merged.

Option name is ``merge_these_atom_types``. To prevent merging of any atom
types set the variable to an empty tuple in Python:

.. code-block:: python

   mk_prep = MoleculePreparation(merge_these_atom_types=())

or pass no parameters in command line

.. code-block:: bash

   mk_prepare_ligand.py -i mol.sdf --merge_these_atom_types



Modifying atom types
--------------------

Atom typing relies on SMARTS patterns to identify chemical substructures.
AutoDock4 atom types are set by default. The easiest way to modify typing
is to add new SMARTS that will superseed the existing ones. For example, let's
assume we want to type hydrogens bound to aromatic carbons as ``HX``. By default,
hydrogens bound to carbon are typed ``H``. A SMARTS pattern to matches
hydrogen bound to carbon is ``"[H][c]"``. From command line:

.. code-block:: bash

   mk_prepare_ligand.py --add_atom_types '[{"smarts": "[H]c", "atype": "HX"}]' -i mol.sdf

We pass a JSON string to ``--add_atom_types`` that is a list of dictionaries. Each
dictionary has a ``"smarts"`` and ``"atype"`` key, and an optional ``"IDX"`` key
that can be used to specify a list of atom indices (0-based) of the atoms in the SMARTS
string that will be typed. By default ``IDX = [0]``.

The equivalent from Python is:

.. code-block:: python

   mk_prep = MoleculePreparation(
       add_atom_types=[{"smarts": "c[H]", "atype": "HX", "IDX": [1]}],
   )

Note that we swapped the order of the atoms in the SMARTS, and are now
explicitly defining the ``"IDX"`` key to type the second atom in the SMARTS.

The full set of atom types can also be specified. This can only be done from
Python or by passing the equivalent configuration JSON file to ``mk_prepare_ligand.py``.
The easiest way to do so, is to put all SMARTS in a JSON file. See the default
file for an example, it is located at ``meeko/data/params/ad4_types.json``.
The ``IDX`` key can be used as described above. Entries are matched in the
order they appear in the file, the last SMARTS pattern that matches an atom is
the one that determines the atom type.
Then the filename can be passed to option ``-p/--load_atom_params``.


Rigidifying bonds
-----------------

By default, single bonds are made rotatable except bonds in rings and amide bonds.
Thioamide and amidine bonds are also not rotatable.
Tertiary amides with non-equivalent substituents on the nitrogen are still made
rotatable, which often leads to unreasonable geometries, but is necessary to
visit both amide rotamers during docking.

Here, we configure Meeko to make single bonds in some conjugated systems rigid,
as defined by the SMARTS ``"C=CC=C"``, and rigidify all amide bonds matched
by ``"[CX3](=O)[NX3]"``, which includes tertiary amides but not thioamides or
amidines:

.. code-block:: bash

   mk_prepare_ligand.py\
     --rigidify_bonds_smarts "C=CC=C"\
     --rigidify_bonds_indices 2 3\
     --rigidify_bonds_smarts "[CX3](=O)[NX3]"\
     --rigidify_bonds_indices 1 3\
     -i mol.sdf

The equivalent code in Python to initialize the molecule preparator is: 

.. code-block:: python

   mk_prep = MoleculePreparation(
       rigidify_bonds_smarts = ["C=CC=C", "[CX3](=O)[NX3]"],
       rigidify_bonds_indices = [(1, 2), (0, 2)],
)

The indices are the indices of the atoms in the SMARTS strings. Note that
we use 0-based indices from the Python API, but 1-based indices from the
command line script. In a future version of Meeko we may use 0-based indices
everywhere.
