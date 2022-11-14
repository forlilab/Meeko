mk_prepare_ligand.py\
    -i ligand_including_cys_sidechain.sdf\
    --receptor protein.pdb\
    --rec_residue ":CYS:6"\
    --tether_smarts "NC(=O)C(O)(C)SCC"\
    --tether_smarts_indices 9 8\
    -o prepared.pdbqt
