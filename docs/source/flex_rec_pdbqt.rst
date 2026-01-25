PDBQT Format for Flexible Sidechains
=============================================


Flexible sidechains in the receptor are treated explicitly during AutoDock simulation. AutoDock
requires a separate PDBQT file with atomic coordinates of the sidechains that will be treated as
flexible. Atomic coordinates and branching keywords for each amino acid is placed between
`BEGIN_RES` and `END_RES` records. The atom linking the amino acid to the protein, which will
remain in fixed position during the simulation, is included as the root. The atoms included in the
flexible residue PDBQT must be omitted from the PDBQT for the rigid portions of the receptor.
For instance, in the example below, the CA atom of the PHE residue is used as the root of the
flexible residue. It is included in the flexible sidechain PDBQT file, and it will be omitted from
the rigid protein PDBQT file

Sample flexible residue file, with two flexible amino acids:

.. code-block:: 
    
    BEGIN_RES PHE A 53
    REMARK 2 active torsions:
    REMARK status: ('A' for Active; 'I' for Inactive)
    REMARK 1 A between atoms: CA and CB
    REMARK 2 A between atoms: CB and CG
    ROOT
    ATOM 1 CA PHE A 53 25.412 19.595 12.578 1.00 12.96 0.180 C
    ENDROOT
    BRANCH 1 2
    ATOM 2 CB PHE A 53 26.873 20.027 12.625 1.00 12.45 0.073 C
    BRANCH 2 3
    ATOM 3 CG PHE A 53 27.286 20.629 13.923 1.00 12.96 -0.056 A
    ATOM 4 CD1 PHE A 53 27.525 19.821 15.027 1.00 11.32 0.007 A
    ATOM 5 CE1 PHE A 53 27.919 20.380 16.242 1.00 13.77 0.001 A
    ATOM 6 CZ PHE A 53 28.108 21.754 16.360 1.00 13.84 0.000 A
    ATOM 7 CE2 PHE A 53 27.877 22.571 15.265 1.00 13.98 0.001 A
    ATOM 8 CD2 PHE A 53 27.470 22.001 14.050 1.00 12.47 0.007 A
    ENDBRANCH 2 3
    ENDBRANCH 1 2
    END_RES PHE A 53
    BEGIN_RES ILE A 54
    REMARK 2 active torsions:
    REMARK status: ('A' for Active; 'I' for Inactive)
    REMARK 3 A between atoms: CA and CB
    REMARK 4 A between atoms: CB and CG1
    ROOT
    ATOM 9 CA ILE A 54 24.457 20.591 9.052 1.00 12.30 0.180 C
    ENDROOT
    BRANCH 9 10
    ATOM 10 CB ILE A 54 22.958 20.662 8.641 1.00 11.82 0.013 C
    ATOM 11 CG2 ILE A 54 22.250 19.367 9.046 1.00 12.63 0.012 C
    BRANCH 10 12
    ATOM 12 CG1 ILE A 54 22.266 21.867 9.298 1.00 13.03 0.002 C
    ATOM 13 CD1 ILE A 54 20.931 22.246 8.670 1.00 14.42 0.005 C
    ENDBRANCH 10 12
    ENDBRANCH 9 10
    END_RES ILE A 54