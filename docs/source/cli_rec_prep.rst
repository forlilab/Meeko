mk_prepare_receptor.py
======================

These are the docs for receptor preparation with the command line script.


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
