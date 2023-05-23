This is a draft

# 1. Prepare protein with waterkit
Follow `wk_prepare_receptor.py` instructions and run with `--pdb`.
The goal of this step is to perform essential fixes to the protein
(such as missing atoms), to add hydrogens, and to follow the Amber
naming scheme for atoms and residues, e.g., `HIE` or `HID`
instead of `HIS`.

# 2. Prepare protein pdbqt
Here, `wk.pdb` was written by waterkit.

```console
$ mk_prepare_receptor.py\
    --pdb wk.pdb\
    -o receptor.pdbqt\
    --flexres " :ARG:348"\
    --reactive_flexres " :SER:308"
```

We can't handle heteroatoms for the moment. Nor nucleic acids.
All parameters are tabulated according to Amber residue and atom names.
We are yet to add ions, nucleic acids, modified residues such as
phosphorilation, and a mechanism to support heteroatoms. 


# 3 Run autogrid

Make affinity maps for the `_rigid.pdbqt` part of the receptor.


# 4. Write ligand PDBQT

Assuming you have a sufex containing molecule in file named `sufex1.sdf`.
```sh
mk_prepare_ligand.py -i sufex1.sdf -o sufex1.pdbqt\
    --reactive_smarts "S(=O)(=O)F"\
    --reactive_smarts_idx 1
```

# 5. Configure AD-GPU for reactive docking

For reactive docking there are two options that need to be passed to AutoDock-GPU:
    ```console
    --derivtype
    --import_dpf
    ```

The `--derivtype` option, if needed, was written by `mk_prepare_receptor.py` to a file suffixed with `.derivtype`.

The filename to be passed to `--import_dpf` was also written by `mk_prepare_receptor.py`
and it is suffixed with `reactive_nbp`.
