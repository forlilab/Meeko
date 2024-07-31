#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko flexibility typer
#

from copy import deepcopy
from .utils import pdbutils

from .molsetup import Bond


def _calc_max_weighted_depth(
    model: dict,
    seed_node: int,
    bonds_to_break: tuple[tuple],
    visited: list[int] = None,
    depth: int = 0,
) -> int:
    """
    Scores a flexibility model based on the depth of the nesting. The number of atoms in a rigid group increases the
    weight of the bonds.

    Parameters
    ----------
    model: dict
        The flexibility model being moved over.
    seed_node: int
        Starting node index.
    bonds_to_break: tuple[tuple]
        An immutable list of bonds to break.
    visited: list[int]
        Nodes that have been visited.
    depth: int

    Returns
    -------
    max_value: int
        The flexibility model score.
    """
    glue_atoms = []
    # Adds the indices from bonds to break to the list of glue atoms
    for i, j in bonds_to_break:
        glue_atoms.append(i)
        glue_atoms.append(j)
    # Pulls the graph and members from the given flexibility model
    graph = model["rigid_body_graph"]
    members = model["rigid_body_members"]
    if visited is None:
        visited = []
    nr_atoms = len(members[seed_node])
    compensation = -1  # First atom after rotatable bond doesn't move, so doesn't weigh
    # Atoms in breakable bonds count twice, because of added complexity
    compensation += int(sum([i in glue_atoms for i in members[seed_node]]))
    this_value = depth**2 * (nr_atoms + compensation)
    max_value = this_value
    visited.append(seed_node)
    # Recurse on nodes that have not been visited yet.
    for node in graph[seed_node]:
        if node not in visited:
            visited.append(node)
            new_value = _calc_max_weighted_depth(
                model, node, bonds_to_break, visited, depth + 1
            )
            max_value = max(max_value, new_value + this_value)
    return max_value


def merge_terminal_atoms(flex_model: dict, not_terminal_atoms: list[int] = ()) -> None:
    """
    Rotatable bonds that link to a rigid body group that contains one atom are removed because that one atom lies on the
    bond axis and rotating the bond does not result in any movement of the atom. The atom after the removed rotatable
    bond is merged with the rigid body group that is upstream of the removed bond.

    Parameters
    ----------
    flex_model: dict
        Flexibility model
    not_terminal_atoms: list
        A list of non-terminal atoms

    Returns
    -------
    None
    """
    # Get members and graph from the flexibility model
    members_dict = flex_model["rigid_body_members"]
    graph = flex_model["rigid_body_graph"]
    remove = {}
    # Loops over the members dict and adds members that can be deleted to the removal dict
    for group_index, members in members_dict.items():
        if (
            len(members) == 1
            and len(graph[group_index]) == 1
            and members[0] not in not_terminal_atoms
            and group_index != flex_model["root"]
        ):
            remove[group_index] = members[0]
    # Goes through the removal dict and removes its contents from the flexibility model
    for group_index, atom_index in remove.items():
        host_index = graph[group_index][0]
        flex_model["rigid_body_members"].pop(group_index)
        flex_model["rigid_body_members"][host_index].append(atom_index)
        flex_model["rigid_body_graph"].pop(group_index)
        flex_model["rigid_body_graph"][host_index].remove(group_index)
        flex_model["rigid_body_connectivity"].pop((host_index, group_index))
        flex_model["rigid_body_connectivity"].pop((group_index, host_index))
        flex_model["rigid_index_by_atom"][atom_index] = host_index
    # subtracts the number of members we are removing from the count
    flex_model["rigid_body_count"] -= len(remove)
    return


def get_flexibility_model(
    molsetup,
    root_atom_index: int = None,
    break_combo_data: dict = None,
):
    """
    Given a MoleculeSetup, creates a flexibility model for that MoleculeSetup. Breaks macrocyclic rings if bond break
    information is provided.

    Parameters
    ----------
    molsetup: RDKitMoleculeSetup
        The molecule setup to generate a flexibility model for.
    root_atom_index: int

    break_combo_data: dict
        Data about different bond break combinations from the macrocycle typer.

    Returns
    -------
    model: dict
        The best flexibility model that could be generated given this MoleculeSetup and break data.
    broken_bonds: list
        A list of the bonds broken corresponding to the returned flexibility model.

    Raises
    ------
    RuntimeError:
        If the flexibility model generated ignores atoms that are not explicitly marked as needing to be ignored in the
        MoleculeSetup.
    """
    # If no macrocyclic rings are to be broken, we simply build the flexibility model.
    if break_combo_data is None or len(break_combo_data["bond_break_combos"]) == 0:
        bonds_to_break = ()
        unbroken_rings_bonds = []
        # Gets all bonds from the MoleculeSetup
        for ring in molsetup.rings:
            for bond in molsetup.get_bonds_in_ring(ring):
                unbroken_rings_bonds.append(bond)
        # Instantiates the flexibility model
        flex_model = walk_rigid_body_graph(
            molsetup, bonds_to_break, unbroken_rings_bonds
        )
        # Gets the number of atoms that aren't marked to be ignored
        nr_not_ignored = sum(
            [not molsetup.get_is_ignore(i) for i in range(len(molsetup.atoms))]
        )
        # Checks the validity of the model by ensuring that all of the atoms that should have been visited were visited,
        # otherwise raises an error
        if len(flex_model["visited"]) != nr_not_ignored:
            molsetup.show()
            msg = f"{len(flex_model['visited'])=} differs from not-ignored atoms {nr_not_ignored}"
            raise RuntimeError(msg)
        # Sets the model root body index
        root_body_index = get_root_body_index(flex_model, root_atom_index)
        flex_model["root"] = root_body_index
        broken_bonds = []
        # Returns the model and the list of broken bonds
        return flex_model, broken_bonds

    # The macrocycle typer enumerated rings to break, and a number of lists
    # of bonds to break/delete that result in breaking of the enumerated rings.
    # to be deleted to break each ring. Deleting a bond results in the remaining
    # bonds within the same ring to become rotatable.
    bond_break_combos = break_combo_data["bond_break_combos"]
    bond_break_scores = break_combo_data["bond_break_scores"]
    unbroken_rings_list = break_combo_data["unbroken_rings"]
    best_model = None
    best_score = float("+inf")
    best_index = None
    # Loops through the potential bond break combinations, generates flexibility model information for each one, and
    # compares the flexibility models to determine the best model to return.
    for index in range(len(bond_break_combos)):
        bond_break_combo = bond_break_combos[index]
        bond_break_score = bond_break_scores[index]
        unbroken_rings_bonds = []
        for ring in unbroken_rings_list[index]:
            for bond in molsetup.get_bonds_in_ring(ring):
                unbroken_rings_bonds.append(bond)

        # Generates a model
        flex_model = walk_rigid_body_graph(
            molsetup, bond_break_combo, unbroken_rings_bonds
        )

        # Gets the number of atoms that aren't marked to be ignored
        nr_not_ignored = sum(
            [not molsetup.get_is_ignore(i) for i in range(len(molsetup.atoms))]
        )
        # Checks the validity of the model by ensuring that all of the atoms that should have been visited were visited,
        # otherwise raises an error
        if len(flex_model["visited"]) != nr_not_ignored:
            msg = f"{len(flex_model['visited'])=} differs from not-ignored atoms {nr_not_ignored}"
            raise RuntimeError(msg)
        # Sets the model root body index
        root_body_index = get_root_body_index(flex_model, root_atom_index)
        flex_model["root"] = root_body_index
        depth_weighted = _calc_max_weighted_depth(
            flex_model, flex_model["root"], bond_break_combo
        )
        # larger bond_break_score is better, larger depth is worse
        # bond break score kinda disappeared in another branch (bonds are either breakable or not)
        score = depth_weighted - 0.001 * bond_break_score
        if score < best_score:
            best_score = score
            best_model = flex_model
            best_index = index

    best_model["score"] = best_score
    broken_bonds = list(break_combo_data["bond_break_combos"][best_index])
    # Returns the best model and the list of broken bonds
    return best_model, broken_bonds


def get_root_body_index(model: dict, root_atom_index: int = None) -> int:
    """
    Gets the index of the rigid body group in the flexibility model that contains the given root_atom_index

    Parameters
    ----------
    model: dict
        Flexibility model
    root_atom_index: int


    Returns
    -------
    root_body_index: int
    """

    # find and return index of rigid body group that contains root_atom_index
    if root_atom_index is not None:
        for body_index in model["rigid_body_members"]:
            if root_atom_index in model["rigid_body_members"][body_index]:
                # 1-index atoms
                return body_index

    # find rigid group that minimizes weighted graph depth
    graph = deepcopy(model["rigid_body_graph"])
    while len(graph) > 2:  # remove leaves until 1 or 2 rigid groups remain
        leaves = []
        for vertex, edges in list(graph.items()):
            if len(edges) == 1:
                leaves.append(vertex)
        for leaf in leaves:
            for vertex, edges in list(graph.items()):
                if leaf in edges:
                    edges.remove(leaf)
                    graph[vertex] = edges
            del graph[leaf]
    if len(graph) == 0:
        root_body_index = 0
    elif len(graph) == 1:
        root_body_index = list(graph.keys())[0]
    else:
        r1, r2 = list(graph.keys())
        r1_size = len(model["rigid_body_members"][r1])
        r2_size = len(model["rigid_body_members"][r2])
        if r1_size >= r2_size:
            root_body_index = r1
        else:
            root_body_index = r2
    return root_body_index


def update_closure_atoms(
    molsetup, bonds_to_break: list[tuple], glue_pseudo_atoms: dict
) -> None:
    """
    Create pseudoatoms required by breaking bonds in the flexibility model

    Parameters
    ----------
    molsetup: MoleculeSetup
        Molecule Setup to modify and add pseudoatoms to
    bonds_to_break: tuple
        List of bonds to break
    glue_pseudo_atoms: dict

    Returns
    -------
    None
    """
    # Loops through the bonds to break and adds all the bonds to
    for i, bond in enumerate(bonds_to_break):
        molsetup.ring_closure_info.bonds_removed.append(
            bond
        )  # bond is a pair of atom indices

        # calculate position and parameters of the pseudoatoms for the closure
        for idx in (0, 1):
            target = bond[1 - idx]
            anchor = bond[0 - idx]
            if glue_pseudo_atoms is None or len(glue_pseudo_atoms) == 0:
                coord = molsetup.get_coord(target)
            else:
                coord = glue_pseudo_atoms[anchor]
            anchor_info = molsetup.get_pdbinfo(anchor)
            pdbinfo = pdbutils.PDBAtomInfo(
                "G",
                anchor_info.resName,
                anchor_info.resNum,
                anchor_info.icode,
                anchor_info.chain,
            )
            pseudo_index = molsetup.add_pseudoatom(
                coord=coord,
                charge=0.0,
                anchor_list=[anchor],
                atom_type=f"G{i}",
                rotatable=False,
                pdbinfo=pdbinfo,
            )
            if anchor in molsetup.ring_closure_info.pseudos_by_atom:
                raise RuntimeError("did not expect more than one G per atom")
            molsetup.ring_closure_info.pseudos_by_atom[anchor] = pseudo_index
        molsetup.set_atom_type(bond[0], "CG%d" % i)
        molsetup.set_atom_type(bond[1], "CG%d" % i)
    return


def walk_rigid_body_graph(
    molsetup,
    bonds_to_break: tuple,
    unbroken_rings_bonds: list[tuple],
    start: int = None,
    data: dict = None,
):
    """
    Recursively walks through the MoleculeSetup to build a graph of rigid bodies. Uses that graph to create and
    populate a flexibility model.

    Parameters
    ----------
    molsetup: RDKitMoleculeSetup
        MoleculeSetup to walk through
    bonds_to_break: tuple[tuple]
    unbroken_rings_bonds: list[tuple(int, int]
    start: int
    data: dict

    Returns
    -------
    data: dict
        A dictionary representing a flexibility model
    """
    # If start is none, uses the default start which is the first non-ignored atom
    if start is None:
        for atom in molsetup.atoms:
            if not atom.is_ignore:
                start = atom.index
                break
    # If the start atom is marked to be ignored, returns nothing
    if molsetup.get_is_ignore(start):
        return
    if data is None:
        data = {
            "visited": [],
            "rigid_body_count": 0,
            "rigid_index_by_atom": {},
            "rigid_body_members": {},
            "rigid_body_connectivity": {},
            "rigid_body_graph": {},
        }
    data["visited"].append(start)
    data["rigid_index_by_atom"][start] = data["rigid_body_count"]
    rigid_index = data["rigid_body_count"]
    sprouts_buffer = []
    # The while loop goes on until all atoms in `group_members` are queried
    # for their neighbors. `idx` is the index within `group_members`, and
    # `current` is the index of the atom being queried. When a neighbor is
    # bonded to `current` by a non-rotatable/rigid bond, it is added to
    # `group_members`. Remember that `idx` is not an index of an atom.
    idx = 0
    group_members = [start]
    while idx < len(group_members):
        current = group_members[idx]
        if molsetup.get_is_ignore(current):
            idx += 1
            continue
        for neigh in molsetup.get_neighbors(current):
            if molsetup.get_is_ignore(neigh):
                continue
            bond_id = Bond.get_bond_id(current, neigh)
            if bond_id in bonds_to_break:
                continue
            bond_info = None
            if bond_id in molsetup.bond_info:
                bond_info = molsetup.bond_info[bond_id]
            if neigh in data["visited"]:
                neigh_in_other_rigid_body = (
                    rigid_index != data["rigid_index_by_atom"][neigh]
                )
                if not bond_info.rotatable and neigh_in_other_rigid_body:
                    raise RuntimeError(
                        "Flexible bonds within rigid group. We have a problem."
                    )
                continue
            if bond_info.rotatable and bond_id not in unbroken_rings_bonds:
                sprouts_buffer.append((current, neigh))
            else:
                group_members.append(neigh)
                data["rigid_index_by_atom"][neigh] = rigid_index
                data["visited"].append(neigh)
        idx += 1
    data["rigid_body_members"][rigid_index] = list(group_members)
    data["rigid_body_graph"].setdefault(rigid_index, [])
    for i, (current, neigh) in enumerate(sprouts_buffer):
        if neigh in data["visited"]:
            continue
        data["rigid_body_count"] += 1
        next_rigid_index = data["rigid_body_count"]
        data["rigid_body_connectivity"][rigid_index, next_rigid_index] = current, neigh
        data["rigid_body_connectivity"][next_rigid_index, rigid_index] = neigh, current
        data["rigid_body_graph"].setdefault(next_rigid_index, [])
        data["rigid_body_graph"][rigid_index].append(next_rigid_index)
        data["rigid_body_graph"][next_rigid_index].append(rigid_index)
        walk_rigid_body_graph(
            molsetup, bonds_to_break, unbroken_rings_bonds, neigh, data
        )
    return data
