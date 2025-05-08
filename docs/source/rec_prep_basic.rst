Basic receptor preparation
========================

Command line script
-------------------

Meeko provides a command-line script ``mk_prepare_receptor.py`` for preparing receptor structures for docking and other downstream simulations. This script supports multiple input formats (PDB, mmCIF, and PQR) and can generate JSON, PDBQT and othe files needed for docking calculations with AutoDock-GPU or AutoDock-Vina. 

To display the help message, run:

.. code-block:: bash

    mk_receptor_ligand.py -h

Writing a single PDBQT file:

.. code-block:: bash

    mk_prepare_receptor.py -i receptor.pdb -o prepared_receptor -p

Supported input formats include PDB, mmCIF, and PQR. Use of PQR may be useful when receptor charges and radii have been precomputed, such as with PDB2PQR. 

Preparing a receptor from a PQR file:

.. code-block:: bash

    # reading both structure and charge from PQR
    mk_prepare_receptor.py --read_pqr receptor.pqr --charge_model read -o prepared_receptor_from_pqr -p
    # reading only structure from PQR
    mk_prepare_receptor.py --read_pqr receptor.pqr -o structure_only -p 

It is important to note that the precomputed charges are only valid with the input structure. Therefore, assignment of residue templates that have discrepancy with the input structure is forbidden with ``--read_pqr`` and ``--charge_model read`` option. 

.. code-block:: bash

    mk_prepare_receptor.py --read_pqr receptor.pqr -o structure_only -p --ignore_template_discrepancy
    # this will ignore the discrepancy and write the output, but it is not recommended
    # as it may lead to incorrect results in downstream calculations.

Regardless of the input format, the current default charge model for receptor preparation is gasteiger. Alternate charge models, for example, ``espaloma``, are accessible via the ``--charge_model`` option: 

.. code-block:: bash

    mk_prepare_receptor.py -i receptor.pdb --charge_model espaloma -o use_espaloma -p

During the preparation, the CLI script may fetch and build missing residue templates. When processing batches of structures, repeated fetching of the same templates can be avoided by using the ``--cache_templates`` option. This will create a cache file with the templates built in the runtime, which can be reused in subsequent runs. 

To createe, update cumulatively and use cached templates from the default location (``$HOME/.meeko_residue_chem_templates_cached.json``): 

.. code-block:: bash

    mk_prepare_receptor.py -i receptor.pdb -o some_output --cache_templates

This is like a dry run to create the cache file only without writing any receptor output. 

To specify a destination or re-uses an existing cache (must be a .JSON file): 

.. code-block:: bash

    mk_prepare_receptor.py -i receptor.pdb -o some_output --cache_templates path_to_existing_cache.json
