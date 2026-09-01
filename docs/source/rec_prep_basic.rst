Basic receptor preparation
========================

Command line script
-------------------

Meeko provides a command-line script ``mk_prepare_receptor.py`` for preparing receptor structures for docking and other downstream simulations. This script supports multiple input formats (PDB, mmCIF, and PQR) and can generate JSON, PDBQT and othe files needed for docking calculations with AutoDock-GPU or AutoDock-Vina. 

To display the help message, run:

.. code-block:: bash

    mk_receptor_receptor.py -h

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


See Also
--------

- Example usage for basic docking preparation with AutoDock-Vina and AutoDock-GPU: :ref:`Basic docking preparation <tutorial1>`
- Example usage for flexible docking: :ref:`Flexible docking <tutorial2>`
- Example usage for covalent docking: :ref:`Covalent docking <tutorial3>`


Python API
----------

The receptor preparation functionality is also accessible via Meeko’s Python API! This allows fine-grained control of preprocessing steps and integration into larger pipelines. The essential and general steps for receptor preparation are:

**1. Load and Clean Input Structure**

Choose a source of structural data (e.g., PDB ID) and preprocess it with a tool like ProDy or RDKit. 

**2. Instantiate the Preparation Method**

To instantiate the default preparation method, use the following code: 

.. code-block:: python

    from meeko import MoleculePreparation
    mk_prep = MoleculePreparation()

To use an established preparation method, you may instantiate ``MoleculePreparation`` with a configuration. This may be a dictionary or a JSON file: 

.. code-block:: python

    from meeko import MoleculePreparation
    # from a dictionary
    mk_prep_from_dict = MoleculePreparation.from_config(config_dict)
    # from a JSON file
    mk_prep_from_file = MoleculePreparation.from_config_file(config_file)

**3. Load Template Library**

To load the default library of ``ResidueChemTemplates``, use the following code: 

.. code-block:: python

    from meeko import ResidueChemTemplates
    chem_templates = ResidueChemTemplates.create_from_defaults()

See the :ref:`this post about templates <py_build_temp>` section for more information on how to create a custom template. See also the API reference on options to load the additional templates.  

**4. Build the Polymer Object**

To be constructed into a Polymer object, the receptor structure needs to be loaded into a ProDy object or a PDB/PQR string. 

.. code-block:: python

    from meeko import Polymer
    # from a ProDy object
    polymer_from_prody = Polymer.from_prody(pdb_str, chem_templates, mk_prep)
    # from a PDB string
    polymer_from_pdb = Polymer.from_pdb_string(pdb_str, chem_templates, mk_prep)

**5. Write Output Files**

The basic output files are PDBQT and JSON. To write a PDBQT file for basic, rigid docking, use the following code: 

.. code-block:: python

    from meeko import PDBQTWriterLegacy
    rigid_pdbqt_str, flex_pdbqt_str = PDBQTWriterLegacy.write_string_from_polymer(polymer)
    with open("receptor_rigid.pdbqt", "w") as f:
        f.write(rigid_pdbqt_str)

To write a JSON file, use the following code:

.. code-block:: python

    json_str = polymer.to_json()
    with open("receptor.json", "w") as f:
        f.write(json_str)

All ``BaseJSONParsable`` objects, including but not limited to ``Polymer``, can be read from/written to a JSON string using the ``from_json()/to_json()`` method. 