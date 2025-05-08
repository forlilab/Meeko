.. _tutorial4a:

================================================================
Preparation and Execution of Virtual Screening with AutoDock-GPU
================================================================

Introduction
============

In this tutorial, we will walk you through the basic steps required for virtual screening using Python as the primary control language during the preparation of ligand and receptor files. To execute the docking calculation, we will use AutoDock-GPU as the docking engine. This small virtual screening example is a docking calculation of a single receptor and ~1000 ligands (500 actives and 500 decoys). The calculation should be doable within 1-2 hours on a regular laptop. 


Ligand Preparation
==================

Acetylcholinesterase (AChE) is a well-studied target for drug discovery, particularly in the context of neurodegenerative diseases. In this section, we will prepare two sets of ligands for virtual screening against AChE using AutoDock-GPU. The process involves converting SMILES strings to molecular structures, applying protonation state assignment, generating conformers, and saving the prepared ligands in PDBQT format for docking. 

Specifically, we will retrieve the curated ligand set from the DUD-E dataset, where human AChE is listed under the target tag "aces/P22303". Docking results with both the active and decoy ligand sets may be used to evaluate the performance of the established docking-selection protocol. To achieve a relatively balanced combined dataset, only the first 500 ligands in the decoy ligand set will be processed. As a minimum requirement, you need to download "aces/actives_final.ism" and "aces/decoys_final.ism", which contain the SMILES strings and ligand names as the starting information of the ligands. 

Regarding the procedure, we will use the `molscrub` package to assign protonation states and generate ligand conformers. Then, we will use the respective writer functions in `meeko` to save the prepared ligand molecules into PDBQT files, which are the required input format for AutoDock-GPU. The ligand preparation process will be parallelized using the `multiprocessing` module to speed up the processing time. 

The following Python script demonstrates the ligand preparation process with possible multiprocessing. This is a complete Python script and you may use Python from terminal to execute this script. 

.. code-block:: python

   from rdkit import Chem
   try:
      from scrubber import Scrub  # Old name util v0.1.1
   except ImportError:
      from molscrub import Scrub  # New name
   from meeko import MoleculePreparation, PDBQTWriterLegacy
   from multiprocessing import Pool
   from time import sleep
   import os

   def process_ligand(args):
      """
      Process a single ligand by scrubbing and preparing it for docking.
      
      Args:
         args (tuple): A tuple containing the SMILES string, ligand name, output directory,
                        scrub instance, and maximum attempts for scrubbing.
      
      Returns:
         list: A list of output file paths generated during the process.
      """
      smi, name, outdir, scrub_instance, max_attempts = args
      output_paths = []

      # Convert SMILES string to molecule object
      mol = Chem.MolFromSmiles(smi)
      if mol is None:
         print(f"[SMILES] Failed to parse: {smi}")
         return []
      
      # Apply scrub with retry in case of failure, primarily with conformer generation
      for attempt in range(1, max_attempts + 1):
         try:
               mol_states = list(scrub_instance(mol))
               break
         except RuntimeError as e:
               print(f"[Scrub Run {attempt}/{max_attempts}] Failed on {name}, molecule Smiles {smi}: {e}")
               sleep(0.1)
      else:
         print(f"[Scrub Run] Gave up on {name}")
         return []

      # Initialize MoleculePreparation instance
      mk_prep = MoleculePreparation()
      counter = 0  # Counter for multiple outcomes from the same ligand SMILES

      # Prepare the molecules and generate the pdbqt files
      for mol_state in mol_states:
         molsetup_list = mk_prep.prepare(mol_state)
         for molsetup in molsetup_list:
               pdbqt_string, success, error_msg = PDBQTWriterLegacy.write_string(molsetup)
               if success:
                  path = os.path.join(outdir, f"{name}_{counter}.pdbqt")
                  with open(path, "w") as f:
                     f.write(pdbqt_string)
                  output_paths.append(path)
                  counter += 1
               else:
                  print(f"[PDBQT] Write failed for {name}: {error_msg}")

      return output_paths

   if __name__ == "__main__":

      # Multiprocessing options
      import multiprocessing
      n_processes = min(multiprocessing.cpu_count(), 8)  # Limit processes based on available cores

      # Scrubbing and ligand preparation options
      max_attempts = 5  # Maximum attempts for scrubbing each ligand
      max_ligands = 500  # Limit the number of ligands processed
      scrub = Scrub(ph_low=7.4, ph_high=7.4, skip_tautomers=True)  # Setup scrub instance with pH constraints

      # Directory creation for ligand sets
      for ligand_set in ["actives_final", "decoys_final"]:  # List of ligand sets
         os.makedirs(ligand_set, exist_ok=True)  # Create directories for ligands

         ligand_list = []
         with open(f"aces/{ligand_set}.ism", "r") as f:  # Input file with SMILES and ligand names
               for line in f:
                  if len(line.split()) >= 2:
                     ligand_smi, ligand_name = line.split()[0], line.split()[-1]
                     ligand_list.append((ligand_smi, ligand_name, ligand_set, scrub, max_attempts))

         print(f"Found {len(ligand_list)} ligands from {ligand_set}")
         print(f"Processing {min(max_ligands, len(ligand_list))} ligands with {n_processes} processes")

         # Multiprocessing to process ligands
         with Pool(processes=n_processes) as pool:
               for result in pool.imap_unordered(process_ligand, ligand_list[:max_ligands]):
                  pass  # Output files are written inside the function


Receptor Preparation
====================

When starting from a high-resolution crystal structure, the receptor preparation process is relatively straightforward. In this example, we will use a structure of human AChE from the Protein Data Bank (PDB ID: 4EY7), a homodimer in complex with Donepezil. Donepezil is FDA-approved for the treatment of Alzheimer's disease and is a well-studied ligand for AChE. The receptor preparation process involves removing water molecules, adding hydrogens, and saving the prepared receptor in PDBQT format. Additionally, we will define the grid box for docking calculations using the center of the Donepezil ligand in the PDB structure and compute the grid maps using AutoGrid, as the second part of the receptor preparation. 

Specifically, we will be choosing only chain A of the dimeric structure, choosing the first alternate location encountered, removing non-protein components, and saving the processed structure in PDBQT format. 

The following code snippet demonstrates how to retrieve the structure and perform basic structure processing with ProDy. The receptor preparation process is not parallelized, as it is relatively fast and does not require significant computational resources. This is only a part of a Python script and you may execute it in an interactive notebook or use it as a component in your own script. 

.. code-block:: python

   from prody import fetchPDB, parsePDB, writePDB, writePDBStream
   from meeko import MoleculePreparation, ResidueChemTemplates
   from meeko import Polymer
   from meeko import PDBQTWriterLegacy
   import numpy as np
   from meeko.gridbox import get_gpf_string
   from io import StringIO

   def get_clean_chainA(pdb_id: str, to_obj: bool = True, to_file: str = None) -> str:
      # Fetch PDB and parse structure
      pdb_path = fetchPDB(pdb_id, folder='.', compressed=False)
      structure = parsePDB(pdb_path, altloc='first')

      # Select protein atoms only in chain A, excluding hetatoms (e.g., ligands, water)
      # keeps only the first listed conformer for each atom
      chainA = structure.select('protein and chain A')

      if chainA is None:
         raise ValueError(f"No protein atoms found in chain A for PDB {pdb_id}")

      if to_obj:
         # Return the cleaned chain A object
         return chainA
      
      if to_file:
         writePDB(to_file, chainA)

      # Get PDB string as well
      pdb_io = StringIO()
      writePDBStream(pdb_io, chainA)
      return pdb_io.getvalue()

   def get_center_of_residue(pdb_id: str, chain_id: str, resname: str):
      # Fetch and parse structure with only the first altloc
      pdb_path = fetchPDB(pdb_id, folder='.', compressed=False)
      structure = parsePDB(pdb_path, altloc='first')

      # Find the residue
      sel_str = f"chain {chain_id} and resname {resname}"
      residue = structure.select(sel_str)
      
      if residue is None:
         raise ValueError(f"Residue {resname} in chain {chain_id} not found in PDB {pdb_id}")
      
      # Compute center of mass of the residue
      center = residue.getCoords().mean(axis=0)

      return center


   # Getting the PDB string for 4EY7 and preprocessing it with ProDy
   pdb_str = {}
   for pdb_id in ['4EY7']:
      pdb_str[pdb_id] = get_clean_chainA(pdb_id, to_file=f"{pdb_id}_chainA.pdb")

   # Constructing the polymer object
   mk_prep = MoleculePreparation()
   chem_templates = ResidueChemTemplates.create_from_defaults()
   mypol = Polymer.from_prody(pdb_str['4EY7'], chem_templates, mk_prep)

   # Writing the polymer object to a PDBQT file
   rigid_pdbqt_string, flex_pdbqt_string = PDBQTWriterLegacy.write_string_from_polymer(mypol)
   with open("4EY7_receptor.pdbqt", "w") as f:
      f.write(rigid_pdbqt_string) # here, we only write the rigid part of the receptor

   # Specifying the needed atom types for the grid map calculation
   # the following are consistent with the default options in the command line script: mk_prepare_receptor.py
   # including all possible atom types for the ligand and receptor
   any_lig_base_types = [
      "HD",
      "C",
      "A",
      "N",
      "NA",
      "OA",
      "F",
      "P",
      "SA",
      "S",
      "Cl",
      "Br",
      "I",
      "Si",
      "B",
   ]
   rec_types = [
      "HD",
      "C",
      "A",
      "N",
      "NA",
      "OA",
      "F",
      "P",
      "SA",
      "S",
      "Cl",
      "Br",
      "I",
      "Mg",
      "Ca",
      "Mn",
      "Fe",
      "Zn",
   ]

   # Calculating the center of the ligand in the PDB structure
   # E20 is the residue name for Donepezil in the PDB structure
   center = get_center_of_residue("4EY7", "A", "E20")

   # Writing out the grid parameter file (GPF) for AutoDock-GPU
   with open("4EY7_gpf.gpf", "w") as f:
      gpf_string, (npts_x, npts_y, npts_z) = get_gpf_string(center, [20.] * 3, "4EY7_receptor.pdbqt", rec_types, any_lig_base_types)
      f.write(gpf_string)


Now, with the prepared receptor PDBQT file ("4EY4_receptor.pdbqt") and the grid parameter file ("4EY7_gpf.gpf"), we are ready to run the grid calculation with AutoGrid: 

.. code-block:: bash

   autogrid4 -p 4EY7_gpf.gpf -l 4EY7_glg.glg


This will generate the grid maps for the receptor, which are the required inputs for AutoDock-GPU to perform the docking calculations. 


Execution of Docking Calculations
=================================

In this section, we will briefly describe how to execute the docking calculations with AutoDock-GPU. The docking calculation is performed using the prepared receptor PDBQT file and the ligand PDBQT files (located in the "actives_final" and "decoys_final" directories, respectively) generated in the previous section: 

.. code-block:: bash

   adgpu --ffile 4EY7_receptor.maps.fld --filelist actives_final
   adgpu --ffile 4EY7_receptor.maps.fld --filelist decoys_final


Please note, to proceed to the next tutorial which involves interaction analysis, you need to have the interactions written to the dlg file. To do this in the docking run time, use the `--contact_analysis/-C 1` option: 

.. code-block:: bash

   adgpu --ffile 4EY7_receptor.maps.fld --filelist actives_final -C 1

If you have obtained the docking outputs but did not include this argument in the docking run, you could still add the interaction analysis using the `--contact_analysis` argument:

.. code-block:: bash

   adgpu -C 1 -X *.xml

This in practice, will rewrite the dlg files with the contact analysis from the xml files, as implied by the long name of argument `--xml2dlg/-X`. 

