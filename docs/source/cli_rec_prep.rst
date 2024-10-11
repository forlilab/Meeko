
The input structure is matched against templates to
guarantee chemical correctness and identify problems with the input structures.
This allows the user to identify and fix problems, resulting in a molecular
model that is correct with respect to heavy atoms, protonation state,
connectivity, bond orders, and formal charges.

The matching algorithm uses the connectivity and elements, but not bond orders
or atom names. Hydrogens are optional. This makes it compatible with input
files from various sources.

Templates are matched on a per residue basis. Each residue is represented
as an instance of a PolymerResidue object, which contains:
 - an RDKit molecule that represents the actual state
 - a padded RDKit molecule containing a few atoms from the adjacent residues
 - parameters such as partial charges

The positions are set by the input, and the connectivity and formal charges
are defined by the templates. Heavy atoms must match exactly. If heavy atoms
are missing or in excess, the templates will fail to match.

Missing hydrogens are added by RDKit, but are not subjected to minimization
with a force field. Thus, their bond lengths are not super accurate.

Different states of the same residue are stored as different templates,
for example different protonation states of HIS, N-term, LYN/LYS, etc.
Residue name is primary key unless user overrides.

Currently not supported: capped residues from charmm-gui.

mk_prepare_receptor
===================

Basic usage
-----------

.. code-block:: bash

    mk_prepare_receptor -i examples/system.pdb --write_pdbqt prepared.pdbqt




Protonation states
------------------


Adding templates
----------------

Write flags
-----------

The option flags starting with ``--write`` in  ``mk_prepare_receptor`` can
be used both with an argument to specify the outpuf filename:
.. code-block:: bash

    --write_pdbqt myenzyme.pdbqt --write_json myenzyme.json

and without the filename argument as long as a default basename is provided:

.. code-block:: bash

    --output_basename myenzyme --write_pdbqt --write_json

It is also possible to combine the two types of usage:

.. code-block:: bash

    --output_basename myenzyme
    --write_pdbqt
    --write_json
    --write_vina_box box_for_myenzyme.txt

in which case the specified filenames have priority over the default basename.
