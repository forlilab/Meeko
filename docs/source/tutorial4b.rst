.. _tutorial4b:

=================================================
Retrospective Docking Analysis and Model Building
=================================================

Introduction
============

Using the outcomes from the preivous tutorial, this tutorial is intended to show the basic procedure of ligand reconstruction, computing of assessment metrics (ROC-AUC, EF) in a retrospective docking analysis, and the building of regression model involving interaction vectors from AutoDock-GPU. 

The main purpose of this tutorial is to show how to build the necessary interface between packages in retrospective docking analysis and model building via a practical example. But please note that the outcomes and conclusions should not be judged professionally. In practice, thorough profiling of the dataset and feature space, careful selection of the regression model and fine-tuning the coditions, are all necessary steps to achieve the optimal model. 

Additional Dependencies
=======================

Single-Pose Molecule Reconstruction
===================================

In this section, we demonstrate how to use Meeko's API functions to reconstruct RDKit molecules from docking results stored in a Ringtail database, which is essentially a SQLite databse. This reconstruction approach aligns with the internal method Ringtail uses when exporting filtered poses to SDF files. Instead of applying Ringtail’s built-in filtering tools, we will select poses using a custom SQL query. The selected poses will then be reconstructed using their SMILES strings, atom index mappings, hydrogen-parent relationships, and 3D coordinates. 

To begin with, we will use Ringtail to extract, transform and store the docking results from *.dlg files of both the actives and decoys ligand sets. Assuming that the *.dlg files are located in the folders named "4EY7_actives" and "4EY7_decoys" in the current directly, we will use the following Python code snippet to write all docking results to a Ringtail database. 

.. code-block:: python

    from ringtail import RingtailCore

    rtc = RingtailCore(db_file = "4EY7_combined.db", docking_mode = "dlg")
    rtc.add_results_from_files(file_path = ["4EY7_actives", "4EY7_decoys"])

The operations are committed instantly to the database. Consequently, a database file "4EY7_combined.db" will be created in the current directory. 

In the following Python code block, we have first defined a helper function `rebuild_mol_meeko` to reconstruct the RDKit molecule from the docking results. The function takes the SMILES string, 3D coordinates (JSON strings), atom index mapping (JSON strings), and hydrogen-parent relationships (JSON strings) as inputs. The function then uses Meeko's `RDKitMolCreate` class to add the pose to the molecule and update the hydrogen positions. 

.. code-block:: python

    import sqlite3
    import pandas as pd
    import json
    from rdkit import Chem
    from rdkit.Chem import SDWriter
    from meeko.rdkit_mol_create import RDKitMolCreate 

    def rebuild_mol_meeko(smiles, coordinates_json, atom_index_map_json, h_parent_json=None):
        """
        Rebuild an RDKit molecule with 3D pose coordinates using Meeko utilities.

        Parameters
        ----------
        smiles : str
            Canonical ligand SMILES.
        coordinates_json : str
            JSON-encoded list of 3D coordinates [['x', 'y', 'z'], ...].
        atom_index_map_json : str
            JSON-encoded flat list: [mol_idx_1, pose_idx_1, mol_idx_2, pose_idx_2, ...].
        h_parent_json : str, optional
            JSON-encoded list of [mol_idx_H, pose_idx_H, ...] for polar hydrogens.

        Returns
        -------
        Chem.Mol
            RDKit molecule with a conformer reconstructed from the pose.
        """

        # Create base molecule
        mol = Chem.MolFromSmiles(smiles)
        # Parse coordinates: [['x', 'y', 'z'], ...] → [(float, float, float), ...]
        coordinates = [tuple(map(float, coord)) for coord in json.loads(coordinates_json)]
        # Parse atom index map: [mol_idx1, pose_idx1, mol_idx2, pose_idx2, ...]
        index_map = [int(x) for x in json.loads(atom_index_map_json)]
        # Apply pose to molecule using Meeko logic
        mol = RDKitMolCreate.add_pose_to_mol(mol, coordinates, index_map)
        # Update hydrogen positions
        if h_parent_json:
            h_parent = [int(x) for x in json.loads(h_parent_json)]
            mol = RDKitMolCreate.add_hydrogens(mol, [coordinates], h_parent)

        return mol

Next, we will reconnect to the Ringtail database and use a custom SQL query to select the docking results. This sample SQL query selects the poses with best score per ligand. A list of reconstructed molecules is then created, where each molecule is assigned the ligand name and docking score as properties. Finally, the reconstructed molecules can be written to a single SDF file, or subject to further analysis with RDKit. 

.. code-block:: python

    # Connect to the Ringtail database
    conn = sqlite3.connect("4EY7_combined.db")
    cursor = conn.cursor()

    # SQL query to select the best pose per ligand
    best_pose_query = """
    SELECT LigName, Pose_ID, docking_score
    FROM Results AS R1
    WHERE docking_score = (
        SELECT MIN(docking_score)
        FROM Results AS R2
        WHERE R1.LigName = R2.LigName
    )
    """

    # Conviniently, created a table to store the best poses
    best_df = pd.read_sql(best_pose_query, conn)

    # Reconstruct molecules
    reconstructed_mols = []
    for _, row in best_df.iterrows():
        pose_id = row["Pose_ID"]
        ligand_name = row["LigName"]
        docking_score = row["docking_score"]

        # Retrieve SMILES, index map and hydrogen parents from the Ligands table, 
        # and pose coordinates from the Results table
        cursor.execute("""
        SELECT L.ligand_smile, R.ligand_coordinates, L.atom_index_map, L.hydrogen_parents
        FROM Ligands L
        JOIN Results R ON L.LigName = R.LigName
        WHERE R.Pose_ID = ?
        """, (pose_id,))
        row = cursor.fetchone()
        smiles, coords_json, index_map_json, h_parent_json = row
        
        mol = rebuild_mol_meeko(smiles, coords_json, index_map_json, h_parent_json)
        mol.SetProp("ligand_name", ligand_name)
        mol.SetProp("docking_score", f"{docking_score:.3f}")
        reconstructed_mols.append(mol)
    conn.close()

    # Optionally, write the reconstructed molecules to a single SDF file
    with SDWriter("4EY7_best_poses reconstructed.sdf") as writer:
        for mol in reconstructed_mols:
            writer.write(mol)
    
While the docking results are stored in the database as JSON-encoded fields, this section shows how they can be efficiently reconstituted into usable RDKit molecules. This "re-hydration" step complements the earlier ETL process by enabling further cheminformatics analysis and modeling. The reconstructed molecules serve as tangible, analyzable outputs ready for visualization, feature extraction, or machine learning workflows. 

Basic Metrics (ROC-AUC, EF) based on Single Metric
==================================================


Vectorization of Interactions, XGBoost Modeling and SHAP explanation
=====================================================================