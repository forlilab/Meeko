Parameterizing receptor from Python
===================================

Parameterizing the receptor means creating a ``Polymer`` instance. That can
be done from a PDB string, or directly from a PDB file. The following example
runs from `test/polymer_data <https://github.com/forlilab/Meeko/tree/develop/test/polymer_data>`_
using file `AHHY.pdb <https://github.com/forlilab/Meeko/blob/develop/test/polymer_data/AHHY.pdb>`_.

.. code-block:: python

    from meeko import Polymer

    fn = "AHHY.pdb"
    polymer = Polymer.from_pdb_file(fn)

    # alternatively, use PDB string (not file)
    with open(fn) as f:
        pdb_string = f.read()
    polymer = Polymer.from_pdb_string(pdb_string)


The parameterization is performed residue by residue, using the same code that
parameterizes ligands. To define how to parameterize a receptor, create an
instance of ``MoleculePreparation`` and pass it to the constructor:

.. code-block:: python

    from meeko import MoleculePreparation

    mk_prep = MoleculePreparation(
        load_atom_params=["openff-2.3.0"],
        charge_model="nagl",  # requires a recent version of OpenFF-toolkit
        merge_these_atom_types=[],
        rigid_macrocycles=True,
    )
    polymer = Polymer.from_pdb_string(pdb_string, mk_prep=mk_prep)


Flexible side chains
--------------------

Make a side chain flexible:

.. code-block:: python

    polymer.flexibilize_sidechain("A:3", mk_prep)


And make all rigid again, or one in particular:

.. code-block:: python

    polymer.rigidify_all(mk_prep)
    polymer.rigidify_sidechain("A:3", mk_prep)


The instance of ``mk_prep`` must be passed because flexibilizing and rigidifying side chains
involves setting bonds as rotatable or not. If the ``mk_prep`` instance is configured
differently than the one used to parameterize the polymer, than flexibilized and rigidified
residues will have different parameters. It is possible to reparameterize an entire polymer
or specific residues:

.. code-block:: python

    polymer.parameterize(mk_prep)
    polymer.monomers["A:3"].parameterize(mk_prep, "A:3")
