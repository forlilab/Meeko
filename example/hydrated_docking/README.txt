HYDRATED DOCKING PROTOCOL
=========================

NOTES

- This protocol assumes all the files have been already prepared for a standard docking
  protocol (i.e. ligand and receptor structures; GPF and DPF parameter files).

- Help for all the scripts can be accessed by running them with no options.

- The directory 'example' contains a case study with both input and output files from a
  typical hydrated docking calculation.



1. Add W atoms to the ligands
------------------------------
The W atoms must be added to a PDBQT file. By default the hydrated ligand is saved with
the "_HYDRO" suffix added (i.e. ligand.pdbqt => ligand_HYDRO.pdbqt).

     $ wet.py -i ligand.pdbqt


2. Calculate grid maps
----------------------
Calculate the grid maps following the standard AutoDock protocol, checking that OA and HD 
types are present in the ligand atom set. If not, the GPF file must be modified to include 
them; i.e. :
       - add OA and HD to the line "ligand_types .... "
       - add lines "map protein.HD.map" and "map protein.OA.map"


3. Generate the W map
---------------------
Water maps are generated by combining OA and HD maps. If standard filenames are used for 
maps (i.e. receptor = protein.pdbqt >> maps = protein.OA.map, protein.HD.map), only the 
receptor name must be specified:

    $ mapwater.py -r protein.pdbqt -s protein.W.map


4. Run dockings
---------------
Prepare the DPF containing the keyword "parameter_file AD4_water_forcefield.dat", and add
the W type map ("protein.W.map"), then run the docking.



5. Extract and score the results
--------------------------------
Docking results are filtered by using the receptor to remove diplaced waters and the W
map file to rank the conserved ones. By default, the LELC pose is extracted as result.

    $ dry.py -c -r protein.pdbqt -c -m protein.W.map -i ligand_HYDRO_protein.dlg


Waters are ranked (STRONG, WEAK) and scored inside the output file ("*_LELC_DRY_SCORED.pdbqt") with the 
calculated energy.

    ...
    REMARK  STRONG water ( score: -0.91 )
    ...
