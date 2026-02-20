Options of mk_prepare_receptor.py
---------------------------------

Input/Output Options
~~~~~~~~~~~~~~~~~~~~

.. option:: --read_pdb <PDB_FILENAME>

   Read a PDB file using the PDB parser in RDKit.

.. option:: --read_pqr <PQR_FILENAME>

   Read PQR file and does not use ProDy

.. option:: -i, --read_with_prody <MACROMOL_FILENAME>

   Read a PDB or mmCIF file using ProDy (if installed). ProDy can be installed from PyPI or conda-forge.

.. option:: -o, --output_basename <basename>

   Specify a default basename for output files created by `--write` options when no filename is specified.

.. option:: -p, --write_pdbqt <PDBQT_FILENAME> [*]

   Output PDBQT files with `_rigid` or `_flex` suffixes for flexible residues. Defaults to `--output_basename` if no filename is provided.

.. option:: -j, --write_json <JSON_FILENAME> [*]

   Save the receptor's parameterized configuration to JSON format. Defaults to `--output_basename` if unspecified.

.. option:: --write_pdb [PDB_FILENAME ...]
   Write prepared receptor to pdb (must specify filename)

.. option:: -g, --write_gpf <GPF_FILENAME> [*]

   Output an AutoGrid input file (GPF). Defaults to `--output_basename` if not specified.

.. option:: -v, --write_vina_box <VINA_BOX_FILENAME> [*]

   Generate a configuration file for Vina with grid box dimensions. Defaults to `--output_basename` if not specified.

.. option:: --write_pdb <PDB_FILENAME> [*]

   Save the prepared receptor in PDB format. Must specify the filename.

.. option:: --debug_fn DEBUG_FN 
   Log debug elvel to filename DEBUG_FN

Receptor Perception Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. option:: -n, --set_template <template>

   Assign templates to residues, e.g., `A:5,7=CYX,B:17=HID`.

.. option:: -d, --delete_residues <residues>

   Specify residues to delete, e.g., `A:350,B:15,16,17`.

.. option:: -b, --blunt_ends <positions>

   Blunt end definitions, e.g., `A:123,200=2,A:1=0`.

.. option:: --add_templates <JSON_FILENAME> [*]

   Load additional templates from one or more JSON files.

.. option:: --cache_templates [CACHE_TEMPLATES] 
   Turns on caching of ResidueChemTemplates (default is OFF) by this option and (optionally) a provided JSON filename.
   Default cache filename is: $HOME/.meeko_residue_chem_templates_cached.json) When the caching is ON, the templates
   for polymer construction will be read from the specified cache file and updates may be made to the same file in a
   cumulative manner.

.. option:: --mk_config
   Can be a local json config file instead, or a the unsuffixed filename of a JSON file packaged
   in meeko/data/mk_configs, for example --mk_config scofu1

.. option:: -d, --delete_bad_res

   (Flag) Ignore residues with missing atoms instead of raising an error. 
   Replaces previous option -a or --allow_bad_res

.. option:: --default_altloc <location>

   Define a default alternate location for residues, overridden by `--wanted_altloc`.

.. option:: --wanted_altloc <location>
   Specify alternate locations for particular residues, e.g., `:5=B,B:17=A`.

.. option:: --compute_charges
   Flag to request that charges be computed from scratch. Default is to read 
   pre-calculated charges from template, which is much faster.

.. option:: --charge_model [gasteiger, nagl, espaloma, zero, read]
   Specify charge model to use. Charges will be computed or extracted from template depending 
   on the `--compute_charges` flag. Note that computing Nagl or Espaloma charges from scratch 
   can be slow. 
   Default is gasteiger, 'zero' sets all zeros, 'read' requires --read_pqr


Grid Box Options
~~~~~~~~~~~~~~~~

.. option:: --box_size <X Y Z>

   Set the size of the grid box in Angstroms (x, y, z).

.. option:: --box_center <X Y Z>

   Define the center of the grid box in Angstroms (x, y, z).

.. option:: --box_center_off_reactive_res

   (Flag) Shift the grid box center 5 Å along the CA-CB bond from CB. Applicable only when there is one reactive flexible residue.

.. option:: --box_enveloping <FILENAME>

   Adjust the grid box to enclose atoms in the specified file. Supported formats: `.sdf`, `.mol`, `.mol2`, `.pdb`, `.pdbqt`.

.. option:: --padding <value>

   Set padding around atoms specified in `--box_enveloping` (in Å).

Flexible and/or Reactive Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. option:: -f, --flexres <residues>

   Define flexible residues by chain ID and residue number, e.g., `-f ":42,B:23"`. 

.. option:: -r, --reactive_flexres <residues>

   Define reactive flexible residues by chain ID and residue number, e.g., `-r ":42,B:23"`. Maximum of 8 reactive residues.

.. option:: --reactive_name <residue:atom>

   Specify the reactive atom name for a residue type, e.g., `--reactive_name "TRP:NE1"`. Can be repeated for multiple assignments.

.. option:: -s, --reactive_name_specific <residue:atom>

   Specify the reactive atom for an individual residue by residue ID, e.g., `-s "A:42=NE2"`. The residue becomes reactive.

.. option:: -t, --rot_terminal_group <residues>

   specify the residues for which to make terminal functional group rotatable by the chain ID and residue number, 
   e.g. -t ":42,B:23" is equivalent to -t ":42" -t "B:23" (leave chain ID empty if omitted in input PDB or mmCIF)'

.. option:: --r_eq_12 <value>

   Set the equilibrium distance (r_eq) for reactive atoms in 1-2 interactions. Default is 1.8 Å.

.. option:: --eps_12 <value>

   Set the epsilon value for reactive atoms in 1-2 interactions. Default is 2.5.

.. option:: --r_eq_13_scaling <factor>

   Scale r_eq for 1-3 interactions across reactive atoms. Default is 0.5.

.. option:: --r_eq_14_scaling <factor>

   Scale r_eq for 1-4 interactions across reactive atoms. Default is 0.5.
