This is a draft

# 1. Prepare protein with waterkit
Follow `mk_prepare_receptor.py` instructions and run with `--pdb`.
The goal of this step is to perform essential fixes to the protein
(such as missing atoms), to add hydrogens, and to follow the Amber
naming scheme for atoms and residues, e.g., `HIE` or `HID`
instead of `HIS`.

# 2. Prepare protein pdbqt
Here, `wk.pdb` was written by waterkit. The example below will center a gridbox of specified size on the given reactive residue.

```console
$ mk_prepare_receptor.py\
    --pdb wk.pdb\
    -o receptor.pdbqt\
    --flexres " :ARG:348"\
    --reactive_flexres " :SER:308"\
    --box_center_on_reactive_res\
    --box_size 40 40 40  # x y z (angstroms)
```
A manual box center can be specified with `--box_center`.
Reactive parameters can also be modified:
```sh
  --r_eq_12 R_EQ_12     r_eq for reactive atoms (1-2 interaction)
  --eps_12 EPS_12       epsilon for reactive atoms (1-2 interaction)
  --r_eq_13_scaling R_EQ_13_SCALING
                        r_eq scaling for 1-3 interaction across reactive atoms
  --r_eq_14_scaling R_EQ_14_SCALING
                        r_eq scaling for 1-4 interaction across reactive atoms
```

We can't handle heteroatoms for the moment. Nor nucleic acids.
All parameters are tabulated according to Amber residue and atom names.
We are yet to add ions, nucleic acids, modified residues such as
phosphorilation, and a mechanism to support heteroatoms. 


# 3 Run autogrid

Make affinity maps for the `_rigid.pdbqt` part of the receptor. `mk_prepare_receptor.py` will prepare the GPF for you.


# 4. Write ligand PDBQT

Assuming you have a sufex containing molecule in file named `sufex1.sdf`.
```sh
mk_prepare_ligand.py -i sufex1.sdf -o sufex1.pdbqt\
    --reactive_smarts "S(=O)(=O)F"\
    --reactive_smarts_idx 1
```

# 5. Configure AD-GPU for reactive docking

For reactive docking there are an additional option that needs to be passed to AutoDock-GPU:
    ```console
    --import_dpf
    ```

The filename to be passed to `--import_dpf` was written by `mk_prepare_receptor.py`
and it is suffixed with `reactive_config`.
```sh
ADGPU -I *.reactive_config -L sufex1.pdbqt -N sufex1_docked_ -F *_flex.pdbqt -C 1
```
