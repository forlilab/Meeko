#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko flexibility typer
#

from copy import deepcopy
from .utils import pdbutils


def _calc_max_weighted_depth(model, seed_node, bonds_to_break, visited=None, depth=0):
    # score a flexibility model based on the depth of the nesting. The number
    # of atoms in a rigid group increases the weight of bonds.

    glue_atoms = []
    for (i, j) in bonds_to_break:
        glue_atoms.append(i)
        glue_atoms.append(j)
    graph = model["rigid_body_graph"]
    members = model["rigid_body_members"]
    if visited is None:
        visited = []
    nr_atoms = len(members[seed_node])
    compensation = -1  # first atom after rotatable bond doesn't move, so doesn't weigh
    # atoms in breakable bonds count twice, because of added complexity
    compensation += int(sum([i in glue_atoms for i in members[seed_node]]))
    this_value = depth**2 * (nr_atoms + compensation)
    max_value = this_value
    visited.append(seed_node)
    for node in graph[seed_node]:
        if node not in visited:
            visited.append(node)
            new_value = _calc_max_weighted_depth(model, node, bonds_to_break, visited, depth + 1)
            max_value = max(max_value, new_value + this_value)
    return max_value


def merge_terminal_atoms(flex_model, not_terminal_atoms=()):
    # Rotatable bonds that link to a rigid body group that contains one atom
    # are removed because that one atom lies on the bond axis and rotating
    # the bond does not result in any movement of the atom. The atom after
    # the removed rotatable bond is merged with the rigid body group that is
    # upstream of the removed bond.

    members_dict = flex_model["rigid_body_members"]
    graph = flex_model["rigid_body_graph"]
    remove = {}
    for group_index, members in members_dict.items():
        if (
            len(members) == 1
            and len(graph[group_index]) == 1
            and members[0] not in not_terminal_atoms
            and group_index != flex_model["root"]
        ):
            remove[group_index] = members[0]
    for group_index, atom_index in remove.items():
        host_index = graph[group_index][0]
        flex_model["rigid_body_members"].pop(group_index)
        flex_model["rigid_body_members"][host_index].append(atom_index)
        flex_model["rigid_body_graph"].pop(group_index)
        flex_model["rigid_body_graph"][host_index].remove(group_index)
        flex_model["rigid_body_connectivity"].pop((host_index, group_index))
        flex_model["rigid_body_connectivity"].pop((group_index, host_index))
        flex_model["rigid_index_by_atom"][atom_index] = host_index
    flex_model["rigid_body_count"] -= len(remove)
    return


def get_flexibility_model(
    molsetup,
    root_atom_index=None,
    break_combo_data=None,
):
    # No macrocyclic rings are to be broken. We simply build the flexibility model.
    if break_combo_data is None or len(break_combo_data["bond_break_combos"]) == 0:
        bonds_to_break = ()
        unbroken_rings_bonds = []
        for ring in molsetup.rings:
            for bond in molsetup.get_bonds_in_ring(ring):
                unbroken_rings_bonds.append(bond)
        flex_model = walk_rigid_body_graph(molsetup, bonds_to_break, unbroken_rings_bonds)
        nr_not_ignored = sum(
            [not molsetup.atom_ignore[i] for i in range(len(molsetup.atom_ignore))]
        )
        if len(flex_model["visited"]) != nr_not_ignored:
            molsetup.show()
            msg = f"{len(flex_model['visited'])=} differs from not-ignored atoms {nr_not_ignored}"
            raise RuntimeError(msg)
        root_body_index = get_root_body_index(flex_model, root_atom_index)
        flex_model["root"] = root_body_index
        broken_bonds = []
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
    for index in range(len(bond_break_combos)):
        bond_break_combo = bond_break_combos[index]
        bond_break_score = bond_break_scores[index]
        unbroken_rings_bonds = []
        for ring in unbroken_rings_list[index]:
            for bond in molsetup.get_bonds_in_ring(ring):
                unbroken_rings_bonds.append(bond)

        flex_model = walk_rigid_body_graph(molsetup, bond_break_combo, unbroken_rings_bonds)
        nr_not_ignored = sum(
            [not molsetup.atom_ignore[i] for i in range(len(molsetup.atom_ignore))]
        )
        if len(flex_model["visited"]) != nr_not_ignored:
            msg = f"{len(flex_model['visited'])=} differs from not-ignored atoms {nr_not_ignored}"
            raise RuntimeError(msg)
        root_body_index = get_root_body_index(flex_model, root_atom_index)
        flex_model["root"] = root_body_index
        depth_weighted = _calc_max_weighted_depth(flex_model, flex_model["root"], bond_break_combo)
        # larger bond_break_score is better, larget depth is worse
        # bond break score kinda disapeared in another branch (bonds are either breakable or not)
        score = depth_weighted - 0.001 * bond_break_score
        if score < best_score:
            best_score = score
            best_model = flex_model
            best_index = index

    best_model["score"] = best_score
    broken_bonds = break_combo_data["bond_break_combos"][best_index]
    return best_model, broken_bonds


def get_root_body_index(model, root_atom_index=None):

    # find and return index of rigid body group that contains root_atom_index
    if root_atom_index is not None:
        for body_index in model["rigid_body_members"]:
            if (
                root_atom_index in model["rigid_body_members"][body_index]
            ):  # 1-index atoms
                root_body_index = body_index
                return body_index

    # find rigid group that minimizes weighted graph depth
    graph = deepcopy(model["rigid_body_graph"])
    while len(graph) > 2:  # remove leafs until 1 or 2 rigid groups remain
        leaves = []
        for vertex, edges in list(graph.items()):
            if len(edges) == 1:
                leaves.append(vertex)
        for l in leaves:
            for vertex, edges in list(graph.items()):
                if l in edges:
                    edges.remove(l)
                    graph[vertex] = edges
            del graph[l]
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


def update_closure_atoms(molsetup, bonds_to_break, glue_pseudo_atoms):
    """create pseudoatoms required by the flexible model with broken bonds"""

    for i, bond in enumerate(bonds_to_break):
        molsetup.ring_closure_info["bonds_removed"].append(
            bond
        )  # bond is pair of atom indices

        """ calculate position and parameters of the pseudoatoms for the closure"""
        for idx in (0, 1):
            target = bond[1 - idx]
            anchor = bond[0 - idx]
            if glue_pseudo_atoms is None or len(glue_pseudo_atoms) == 0:
                coord = molsetup.get_coord(target)
            else:
                coord = glue_pseudo_atoms[anchor]
            anchor_info = molsetup.pdbinfo[anchor]
            pdbinfo = pdbutils.PDBAtomInfo(
                "G", anchor_info.resName, anchor_info.resNum, anchor_info.icode, anchor_info.chain
            )
            pseudo_index = molsetup.add_pseudo(
                coord=coord,
                charge=0.0,
                anchor_list=[anchor],
                atom_type=f"G{i}",
                rotatable=False,
                pdbinfo=pdbinfo,
            )
            if anchor in molsetup.ring_closure_info:
                raise RuntimeError("did not expect more than one G per atom")
            molsetup.ring_closure_info["pseudos_by_atom"][anchor] = pseudo_index
        molsetup.set_atom_type(bond[0], "CG%d" % i)
        molsetup.set_atom_type(bond[1], "CG%d" % i)
    return


def walk_rigid_body_graph(molsetup, bonds_to_break, unbroken_rings_bonds, start=None, data=None):
    """recursive walk to build the graph of rigid bodies"""

    if start is None:
        for index, _ in enumerate(molsetup.atom_ignore):
            if not molsetup.atom_ignore[index]:
                start = index  # default start is 1st non-ignored atom
                break
    if molsetup.atom_ignore[start]:
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
    # for their neighbors. `idx` is the index whithin `group_members`, and
    # `current` is the index of the atom being queried. When a neighbor is
    # bonded to `current` by a non-rotatable/rigid bond, it is added to
    # `group_members`. Remember that `idx` is not an index of an atom.
    idx = 0
    group_members = [start]
    while idx < len(group_members):
        current = group_members[idx]
        if molsetup.atom_ignore[current]:
            idx += 1
            continue
        for neigh in molsetup.get_neigh(current):
            if molsetup.atom_ignore[neigh]:
                continue
            bond_id = molsetup.get_bond_id(current, neigh)
            if bond_id in bonds_to_break:
                continue
            bond_info = molsetup.get_bond(current, neigh)
            if neigh in data["visited"]:
                neigh_in_other_rigid_body = (
                    rigid_index != data["rigid_index_by_atom"][neigh]
                )
                if not bond_info["rotatable"] and neigh_in_other_rigid_body:
                    raise RuntimeError(
                        "Flexible bonds within rigid group. We have a problem."
                    )
                continue
            if bond_info["rotatable"] and bond_id not in unbroken_rings_bonds:
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
        walk_rigid_body_graph(molsetup, bonds_to_break, unbroken_rings_bonds, neigh, data)
    return data
