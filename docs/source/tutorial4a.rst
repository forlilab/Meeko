Tutorial 4A: Preparation and Execution of Virtual Screening
=========================================================

.. contents::
   :depth: 2
   :local:

In this tutorial, we will walk you through the basic steps required for virtual screening using Python as the primary control language during the preparation of ligand and receptor files. To execute the docking calculation, we will use AutoDock-GPU as the docking engine. This small virtual screening example is a docking calculation of a single receptor and ~1000 ligands (500 actives and 500 decoys). The calculation should be doable within an hour on a regular laptop. 

**Outline**
--------------------
1. Ligand Preparation
2. Receptor Preparation
3. Execution of Docking with AutoDock-GPU


### Ligand Preparation
-----------------------

Acetylcholinesterase (AChE) is a well-studied target for drug discovery, particularly in the context of neurodegenerative diseases. In this section, we will prepare two set of ligands for virtual screening against AChE using AutoDock-GPU. The process involves converting SMILES strings to molecular structures, applying protonation state assignment, generating conformers, and saving the prepared ligands in PDBQT format for docking. 

Specifically, we will retrieve the curated ligand set from the DUD-E dataset, where human AChE is listed under the target tage "aces/P22303". The ligand set includes both actives and decoys, which may be used to evaluate the performance of the established docking-selection protocol. As a minimum requirement, you need to download "aces/actives_final.ism" and "aces/decoys_final.ism", which contain the SMILES strings and ligand names. 

In regard to the precedure, we will use the `molscrub` package to assign protonation states and generate ligand conformers. Then, we will use the corresponding writer functions in `meeko` to save the prepared ligands into PDBQT files, which are the required input format for AutoDock-GPU. The ligand preparation process will be parallelized using the `multiprocessing` module to speed up the processing time. 

The following code block demonstrates the ligand preparation process:

```python
from rdkit import Chem
from scrubber import Scrub
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
