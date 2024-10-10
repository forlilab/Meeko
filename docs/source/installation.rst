Installation
============

We recommend using micromamba to manage Python environments and install Meeko.
Other similar package managers also work, like mamba, conda, or miniconda.
We prefer micromamba because it uses conda-forge as its default channel.
If you use other package managers, please use the ``-c conda-forge`` option.

To get micromamba, visit https://mamba.readthedocs.io/en/latest/


From conda-forge
----------------

.. code-block:: bash

    micromamba install meeko


From PyPI
------------------------

.. code-block:: bash

    pip install meeko

If using micromamba or a similar package manager, ``pip`` installs the package
in the active environment.


From source
-----------

Here, we will checkout the ``develop`` branch, as it is likely more recent than the
default ``release`` branch. Accessing features that are not in a release yet is one
of the reasons to use the develop branch, which requires installing from source.

.. code-block:: bash

    git clone https://github.com/forlilab/Meeko.git
    cd Meeko
    git checkout develop
    pip install .

Alternatively, it is possible to install with ``pip install -e .``. Then, changes to
the source files take immediate effect without requiring further ``pip install .``.
This is useful for developers. Changes to the command line scripts may still require
a re-installation.


Support for Python 3.12
-----------------------

Meeko runs on Python 3.12 as long as Prody is not installed. To run on 3.12,
install all dependencies except Prody and install Meeko from source.

Meeko uses Prody to parse PDB and mmCIF files. Without prody, PDB files
can be parsed with the command line option ``--read_pdb`` and with the Python
method ``LinkedRDKitChorizo.from_pdb_string()``. However, without ProDy it
won't be possible to read mmCIF files or use tethered docking. Prody developers
are working to support Python 3.12, so it is possible that Prody will work
on Python 3.12 soon.
