mk_export.py
============

Convert docking results to SDF
------------------------------

AutoDock-GPU and Vina write docking results in the PDBQT format. The DLG output
from AutoDock-GPU contains docked poses in PDBQT blocks, plus additional information.
Meeko generates RDKit molecules from PDBQT using the SMILES
string in the REMARK lines. The REMARK lines also have the mapping of atom indices
between SMILES and PDBQT. SD files with docked coordinates are written
from RDKit molecules.

.. code-block:: bash

    mk_export.py molecule.pdbqt -o molecule.sdf
    mk_export.py vina_results.pdbqt -o vina_results.sdf
    mk_export.py autodock-gpu_results.dlg -o autodock-gpu_results.sdf

Why this matters
----------------

Making RDKit molecules from SMILES is safer than guessing bond orders
from the coordinates, specially because the PDBQT lacks hydrogens bonded
to carbon. As an example, consider the following conversion, in which
OpenBabel adds an extra double bond, not because it has a bad algorithm,
but because this is a nearly impossible task.

.. code-block:: bash

    obabel -:"C1C=CCO1" -o pdbqt --gen3d | obabel -i pdbqt -o smi
    [C]1=[C][C]=[C]O1


Caveats
-------

If docking does not use explicit Hs, which it often does not, the
exported positions of hydrogens are calculated from RDKit. This can
be annoying if a careful forcefield minimization is employed before
docking, as probably rigorous Hs positions will be replaced by the
RDKit geometry rules, which are empirical and much simpler than most
force fields.
