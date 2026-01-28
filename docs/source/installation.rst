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

    micromamba install -c conda-forge meeko

Other tools like ``mamba`` or ``conda`` can also be used. Note that ``conda``
is significantly slower.


Installing dependencies manually
--------------------------------

The conda-forge recipe includes the list of dependencies, and using `micromamba`
will install them automatically. But we didn't add them to the Python package
configuration, to allow the user to control dependencies manually. 

Meeko requires Python 3.10 or later, and `scipy`, `rdkit`, `gemmi`, and `tqdm`,
which can be installed from PyPI with `pip`:

.. code-block:: bash

   pip install scipy rdkit gemmi tqdm

If using Python 3.8 or 3.9, please see :ref:`suport-py38` for details. 


Optional dependencies
---------------------

Although optional, ProDy is installed by the conda-forge recipe.
Meeko uses Prody to parse PDB and mmCIF files. Without prody, PDB files
can be parsed with the command line option ``--read_pdb`` and with the Python
method ``Polymer.from_pdb_string()``. However, without ProDy it
won't be possible to read mmCIF files or use tethered docking.
Prody is available on PyPI and conda-forge for (at least) Python 3.11 and 3.12.

.. code-block:: bash

   pip install prody

Espaloma can be used to assign partal charges and proper torsion parameters.
These parameters are useful only for development.
Not to be confused with package `espaloma_charge`. Meeko uses the `espaloma` package.
See installation details at `espaloma readthedocs <https://espaloma.wangyq.net/install.html>`_

`OpenFF forcefields <https://github.com/openforcefield/openff-forcefields>`__
is used to assign vdW and proper torsion parameters. This is
useful only for development. It can be installed with:

.. code-block:: bash

   micromamba install -c conda-forge openff-forcefields

`OpenFF Toolkit <https://docs.openforcefield.org/projects/toolkit>`__ is used to assign
NAGL charges, useful only for development. See `installation instructions <https://docs.openforcefield.org/projects/toolkit/en/stable/installation.html>`__



From PyPI
---------

.. code-block:: bash

    pip install meeko

If using ``micromamba`` or a similar package manager, ``pip`` installs the
package in the active environment. The following dependencies required:
``python``, ``numpy``, ``scipy``, ``gemmi``, and ``rdkit``.

Installing with ``pip`` does not check for missing dependencies or attempt to
resolve conflicts between versions. That information is included on the
conda-forge recipe and used by a package manager like
``mamba`` or ``micromamba``.


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


.. _suport-py38:
Support for Python 3.8
-----------------------

As of v0.6.1, Meeko starts to use features that are only available in Python 3.9 or later. 
These features are: 

- Dictionary Union Operators
- Py39 Type Hints
- ``importlib.resources.files`` (Details in `PR #223 <https://github.com/forlilab/Meeko/pull/223>`_)

However, they are not essential for the code structure or function. To adapt Meeko to a Python 3.8 
environment, consider the respective workarounds, such as: 

- Replace dictionary union operators with dictionary unpacking (`{**dict1, **dict2}`), or the `update()` method.
- Use ``List``, ``Set``, ``Dict``, and ``Tuple`` from ``typing`` for type hints.
- Use ``Path`` from ``pathlib``, or ``importlib_resources`` backport package for ``importlib.resources.files``.

See a diff report of these changes in `Issue #313 <https://github.com/forlilab/Meeko/issues/313#issuecomment-2612000768>`_. 

Meeko v0.8 includes `match`/`case` statements, which require Python 3.10
