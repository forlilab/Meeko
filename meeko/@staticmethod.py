    @staticmethod
    def _build_padded_mols(monomers, bonds, padders):
        """

        Parameters
        ----------
        monomers
        bonds
        padders

        Returns
        -------

        """
        padded_mols = {}
        bond_use_count = {key: 0 for key in bonds}
        for (
            residue_id,
            monomer,
        ) in monomers.items():
            if monomer.rdkit_mol is None:
                continue
            padded_mol = monomer.rdkit_mol
            mapidx_pad = {
                atom.GetIdx(): atom.GetIdx() for atom in padded_mol.GetAtoms()
            }
            for atom_index, link_label in monomer.link_labels.items():
                adjacent_rid = None
                adjacent_mol = None
                adjacent_atom_index = None
                for (r1_id, r2_id), bond_list in bonds.items():
                    # TODO the second and subsequent bonds between a pair of
                    # residues will not update the padding atoms with the
                    # positions of the adjacent residues. This is OK, the same
                    # happens for blunt residues, because the adjacent residue
                    # is missing.
                    i1, i2 = bond_list[0]
                    if r1_id == residue_id and i1 == atom_index:
                        adjacent_rid = r2_id
                        adjacent_atom_index = i2
                        break
                    elif r2_id == residue_id and i2 == atom_index:
                        adjacent_rid = r1_id
                        adjacent_atom_index = i1
                        break
                
                if adjacent_rid is not None:
                    adjacent_mol = monomers[adjacent_rid].rdkit_mol
                    bond_use_count[(r1_id, r2_id)] += 1
                
                padded_mol, mapidx = padders[link_label](
                    padded_mol, adjacent_mol, atom_index, adjacent_atom_index
                )

                tmp = {}
                for i, j in enumerate(mapidx):
                    if j is None:
                        continue  # new padding atom
                    if j not in mapidx_pad:
                        continue  # padding atom from previous iteration for another link_label
                    tmp[i] = mapidx_pad[j]
                mapidx_pad = tmp

            # update position of hydrogens bonded to link atoms
            inv = {j: i for (i, j) in mapidx_pad.items()}
            padded_idxs_to_update = []
            no_pad_idxs_to_update = []
            for atom_index in monomer.link_labels:
                heavy_atom = monomer.rdkit_mol.GetAtomWithIdx(atom_index)
                for neighbor in heavy_atom.GetNeighbors():
                    if neighbor.GetAtomicNum() != 1:
                        continue
                    if neighbor.GetIdx() in monomer.mapidx_to_raw:
                        # index of H exists in mapidx_to_raw, which means that
                        # the raw_input_mol had the hydrogen. Thus, we do not
                        # want to update its coordiantes.
                        continue
                    no_pad_idxs_to_update.append(neighbor.GetIdx())
                    padded_idxs_to_update.append(inv[neighbor.GetIdx()])
            update_H_positions(padded_mol, padded_idxs_to_update)
            source = padded_mol.GetConformer()
            destination = monomer.rdkit_mol.GetConformer()
            for i, j in zip(no_pad_idxs_to_update, padded_idxs_to_update):
                destination.SetAtomPosition(i, source.GetAtomPosition(j))
                # can invert chirality in 3D positions

            padded_mols[residue_id] = (padded_mol, mapidx_pad)



