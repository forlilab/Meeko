"""Polymer-construction utility functions.

Top-level helpers that used to live near the top of ``meeko/polymer.py``:
graph-path search, inter-residue bond detection (KD-tree based and legacy),
MCS-based atom mapping, integer/charge rectification, hydrogen position
updates, and residue deletion.
"""

import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Geometry import Point3D

from ..utils.covalent_radius_table import covalent_radius
from .errors import PolymerCreationError

eol = "\n"

logger = logging.getLogger(__name__)
periodic_table = Chem.GetPeriodicTable()


def find_graph_paths(graph, start_node, end_nodes, current_path=(), paths_found=()):
    """Recursively find all paths between start and end nodes."""
    current_path = current_path + (start_node,)
    paths_found = list(paths_found)
    for node in graph[start_node]:
        if node in current_path:
            continue
        if node in end_nodes:
            paths_found.append(list(current_path) + [node])
        more_paths = find_graph_paths(graph, node, end_nodes, current_path)
        paths_found.extend(more_paths)
    return paths_found


def find_inter_mols_bonds(
    mols_dict,
    covalent_radius=covalent_radius,
    periodic_table=periodic_table,
    allowance=1.2,
):
    """Find inter-residue bonds within atom-specific covalent radii.

    Uses scipy.spatial.cKDTree for the initial pair search.
    mols_dict: dict[key] -> (rdkit_mol, ...)
    Returns dict[(key_i, key_j)] -> list[(atom_i, atom_j)].
    """
    keys = list(mols_dict.keys())

    all_xyz = []
    all_z = []
    all_mol_id = []
    all_atom_id = []

    missing = set()
    for mol_i, k in enumerate(keys):
        mol = mols_dict[k][0]
        conf = mol.GetConformer()
        xyz = conf.GetPositions()
        zs = np.array([a.GetSymbol() for a in mol.GetAtoms()])
        for z in np.unique(zs):
            if z not in covalent_radius:
                missing.add(z)
        n = xyz.shape[0]
        all_xyz.append(xyz)
        all_z.append(zs)
        all_mol_id.append(np.full(n, mol_i, dtype=np.int32))
        all_atom_id.append(np.arange(n, dtype=np.int32))

    if missing:
        syms = [periodic_table.GetElementSymbol(int(z)) for z in sorted(missing)]
        raise RuntimeError(f"Missing covalent radii for elements: {', '.join(syms)}")

    xyz = np.vstack(all_xyz)
    z = np.concatenate(all_z)
    mol_id = np.concatenate(all_mol_id)
    atom_id = np.concatenate(all_atom_id)

    rad = np.array([covalent_radius[zi] for zi in z], dtype=np.float64)
    max_possible_covalent_radius = (
        2.0 * allowance * float(max(covalent_radius.values()))
    )

    from scipy.spatial import cKDTree
    tree = cKDTree(xyz)
    cand = np.array(
        list(tree.query_pairs(r=max_possible_covalent_radius)), dtype=np.int64
    )
    if cand.size == 0:
        return {}

    i = cand[:, 0]
    j = cand[:, 1]
    inter = mol_id[i] != mol_id[j]
    i = i[inter]
    j = j[inter]
    if i.size == 0:
        return {}

    thresh2 = (allowance * (rad[i] + rad[j])) ** 2
    d = xyz[i] - xyz[j]
    dist2 = np.einsum("ij,ij->i", d, d)
    ok = dist2 < thresh2
    i = i[ok]
    j = j[ok]
    if i.size == 0:
        return {}

    bonds = {}
    mi = mol_id[i]
    mj = mol_id[j]
    swap = mi > mj
    i2 = i.copy()
    j2 = j.copy()
    mi2 = mi.copy()
    mj2 = mj.copy()
    i2[swap], j2[swap] = j2[swap], i2[swap]
    mi2[swap], mj2[swap] = mj2[swap], mi2[swap]
    for a_glob, b_glob, ma, mb in zip(i2, j2, mi2, mj2):
        key = (keys[int(ma)], keys[int(mb)])
        val = (int(atom_id[int(a_glob)]), int(atom_id[int(b_glob)]))
        bonds.setdefault(key, []).append(val)
    return bonds


def find_inter_mols_bonds_old(mols_dict):
    """O(N^2) reference implementation, kept for parity testing."""
    allowance = 1.2
    max_possible_covalent_radius = (
        2 * allowance * max([r for k, r in covalent_radius.items()])
    )
    cubes_min = []
    cubes_max = []
    for key, (mol, _) in mols_dict.items():
        positions = mol.GetConformer().GetPositions()
        cubes_min.append(np.min(positions, axis=0))
        cubes_max.append(np.max(positions, axis=0))
    tmp = np.array([0, 0, 1, 1])
    pairs_to_consider = []
    keys = list(mols_dict)
    for i in range(len(mols_dict)):
        for j in range(i + 1, len(mols_dict)):
            do_consider = True
            for d in range(3):
                x = (
                    cubes_min[i][d],
                    cubes_max[i][d],
                    cubes_min[j][d],
                    cubes_max[j][d],
                )
                idx = np.argsort(x)
                has_overlap = tmp[idx][0] != tmp[idx][1]
                close_enough = (
                    abs(x[idx[1]] - x[idx[2]]) < max_possible_covalent_radius
                )
                do_consider &= close_enough or has_overlap
            if do_consider:
                pairs_to_consider.append((i, j))

    bonds = {}
    for i, j in pairs_to_consider:
        p1 = mols_dict[keys[i]][0].GetConformer().GetPositions()
        p2 = mols_dict[keys[j]][0].GetConformer().GetPositions()
        for a1 in mols_dict[keys[i]][0].GetAtoms():
            for a2 in mols_dict[keys[j]][0].GetAtoms():
                vec = p1[a1.GetIdx()] - p2[a2.GetIdx()]
                distsqr = np.dot(vec, vec)
                for atom in [a1, a2]:
                    if atom.GetAtomicNum() not in covalent_radius:
                        raise RuntimeError(
                            f"Element {periodic_table.GetElementSymbol(atom.GetAtomicNum())} "
                            "doesn't have an implemented covalent radius, which was required "
                            "for the perception of intermolecular bonds. "
                        )
                cov_dist = (
                    covalent_radius[a1.GetAtomicNum()]
                    + covalent_radius[a2.GetAtomicNum()]
                )
                if distsqr < (allowance * cov_dist) ** 2:
                    key = (keys[i], keys[j])
                    value = (a1.GetIdx(), a2.GetIdx())
                    bonds.setdefault(key, [])
                    bonds[key].append(value)
    return bonds


def find_inter_mols_bonds_kdtree_fast(
    mols_dict, covalent_radius, periodic_table, allowance=1.2
):
    keys = list(mols_dict.keys())
    all_xyz = []
    all_z = []
    all_mol_id = []
    all_atom_id = []
    missing = set()
    for mol_i, k in enumerate(keys):
        mol = mols_dict[k][0]
        xyz = mol.GetConformer().GetPositions()
        zs = np.fromiter(
            (a.GetAtomicNum() for a in mol.GetAtoms()), dtype=np.int32
        )
        for z0 in np.unique(zs):
            if int(z0) not in covalent_radius:
                missing.add(int(z0))
        n = xyz.shape[0]
        all_xyz.append(xyz)
        all_z.append(zs)
        all_mol_id.append(np.full(n, mol_i, dtype=np.int32))
        all_atom_id.append(np.arange(n, dtype=np.int32))

    if missing:
        syms = [periodic_table.GetElementSymbol(z) for z in sorted(missing)]
        raise RuntimeError(f"Missing covalent radii for elements: {', '.join(syms)}")

    xyz = np.vstack(all_xyz).astype(np.float64, copy=False)
    z = np.concatenate(all_z)
    mol_id = np.concatenate(all_mol_id)
    atom_id = np.concatenate(all_atom_id)
    rad = np.array([covalent_radius[int(zi)] for zi in z], dtype=np.float64)
    max_r = 2.0 * allowance * float(max(covalent_radius.values()))

    from scipy.spatial import cKDTree
    tree = cKDTree(xyz)
    coo = tree.sparse_distance_matrix(tree, max_r, output_type="coo_matrix")

    i = coo.row.astype(np.int64, copy=False)
    j = coo.col.astype(np.int64, copy=False)
    keep = i < j
    i = i[keep]
    j = j[keep]
    dist2 = np.square(coo.data[keep]).astype(np.float64, copy=False)

    inter = mol_id[i] != mol_id[j]
    i = i[inter]
    j = j[inter]
    dist2 = dist2[inter]
    if i.size == 0:
        return {}

    thresh2 = np.square(allowance * (rad[i] + rad[j]))
    ok = dist2 < thresh2
    i = i[ok]
    j = j[ok]
    if i.size == 0:
        return {}

    mi = mol_id[i]
    mj = mol_id[j]
    swap = mi > mj
    if np.any(swap):
        i_s = i.copy()
        j_s = j.copy()
        mi_s = mi.copy()
        mj_s = mj.copy()
        i_s[swap], j_s[swap] = j_s[swap], i_s[swap]
        mi_s[swap], mj_s[swap] = mj_s[swap], mi_s[swap]
        i, j, mi, mj = i_s, j_s, mi_s, mj_s

    bonds = {}
    for a_glob, b_glob, ma, mb in zip(i, j, mi, mj):
        key = (keys[int(ma)], keys[int(mb)])
        val = (int(atom_id[int(a_glob)]), int(atom_id[int(b_glob)]))
        bonds.setdefault(key, []).append(val)
    return bonds


def find_inter_mols_bonds_kdtree(
    mols_dict, covalent_radius, periodic_table, allowance=1.2
):
    """Alternative KD-tree implementation using query_pairs."""
    keys = list(mols_dict.keys())
    all_xyz = []
    all_z = []
    all_mol_id = []
    all_atom_id = []
    missing = set()
    for mol_i, k in enumerate(keys):
        mol = mols_dict[k][0]
        conf = mol.GetConformer()
        xyz = conf.GetPositions()
        zs = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32)
        for z in np.unique(zs):
            if z not in covalent_radius:
                missing.add(z)
        n = xyz.shape[0]
        all_xyz.append(xyz)
        all_z.append(zs)
        all_mol_id.append(np.full(n, mol_i, dtype=np.int32))
        all_atom_id.append(np.arange(n, dtype=np.int32))

    if missing:
        syms = [periodic_table.GetElementSymbol(int(z)) for z in sorted(missing)]
        raise RuntimeError(f"Missing covalent radii for elements: {', '.join(syms)}")

    xyz = np.vstack(all_xyz)
    z = np.concatenate(all_z)
    mol_id = np.concatenate(all_mol_id)
    atom_id = np.concatenate(all_atom_id)
    rad = np.array([covalent_radius[int(zi)] for zi in z], dtype=np.float64)
    max_possible_covalent_radius = (
        2.0 * allowance * float(max(covalent_radius.values()))
    )

    from scipy.spatial import cKDTree
    tree = cKDTree(xyz)
    cand = np.array(
        list(tree.query_pairs(r=max_possible_covalent_radius)), dtype=np.int64
    )
    if cand.size == 0:
        return {}

    i = cand[:, 0]
    j = cand[:, 1]
    inter = mol_id[i] != mol_id[j]
    i = i[inter]
    j = j[inter]
    if i.size == 0:
        return {}

    thresh2 = (allowance * (rad[i] + rad[j])) ** 2
    d = xyz[i] - xyz[j]
    dist2 = np.einsum("ij,ij->i", d, d)
    ok = dist2 < thresh2
    i = i[ok]
    j = j[ok]
    if i.size == 0:
        return {}

    bonds = {}
    mi = mol_id[i]
    mj = mol_id[j]
    swap = mi > mj
    i2 = i.copy()
    j2 = j.copy()
    mi2 = mi.copy()
    mj2 = mj.copy()
    i2[swap], j2[swap] = j2[swap], i2[swap]
    mi2[swap], mj2[swap] = mj2[swap], mi2[swap]
    for a_glob, b_glob, ma, mb in zip(i2, j2, mi2, mj2):
        key = (keys[int(ma)], keys[int(mb)])
        val = (int(atom_id[int(a_glob)]), int(atom_id[int(b_glob)]))
        bonds.setdefault(key, []).append(val)
    return bonds


def mapping_by_mcs(mol, ref):
    mcs_result = rdFMCS.FindMCS([mol, ref], bondCompare=rdFMCS.BondCompare.CompareAny)
    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
    mol_idxs = mol.GetSubstructMatch(mcs_mol)
    ref_idxs = ref.GetSubstructMatch(mcs_mol)
    return {i: j for (i, j) in zip(mol_idxs, ref_idxs)}


def _snap_to_int(value, tolerance=0.12):
    for inc in [-1, 0, 1]:
        if abs(value - int(value) - inc) <= tolerance:
            return int(value) + inc
    return None


def divide_int_gracefully(integer, weights, allow_equal_weights_to_differ=False):
    for weight in weights:
        if type(weight) not in [int, float] or weight < 0:
            raise ValueError("weights must be numeric and non-negative")
    if type(integer) is not int:
        raise ValueError("integer must be integer")
    if sum(weights) < np.finfo(np.float32).eps:
        shares = [1.0 / len(weights) for _ in weights]
    else:
        inv_total_weight = 1.0 / sum(weights)
        shares = [w * inv_total_weight for w in weights]
    result = [_snap_to_int(integer * s, tolerance=0.5) for s in shares]
    surplus = integer - sum(result)
    if surplus == 0:
        return result
    data = [(i, w) for (i, w) in enumerate(weights)]
    data = sorted(data, key=lambda x: x[1], reverse=True)
    idxs = [i for (i, _) in data]
    if allow_equal_weights_to_differ:
        groups = [1 for _ in weights]
    else:
        groups = []
        last_weight = None
        for i in idxs:
            if weights[i] == last_weight:
                groups[-1] += 1
            else:
                groups.append(1)
            last_weight = weights[i]

    nr_groups = len(groups)
    for j in range(1, 2**nr_groups):
        n_changes = 0
        combo = []
        for grpidx in range(nr_groups):
            is_changed = bool(j & 2**grpidx)
            combo.append(is_changed)
            n_changes += is_changed * groups[grpidx]
        if n_changes == abs(surplus):
            break

    increment = surplus / abs(surplus)
    index = 0
    for i, is_changed in enumerate(combo):
        if is_changed:
            for j in range(groups[i]):
                result[idxs[index]] += increment
                index += 1
    return result


def rectify_charges(q_list, net_charge=None, decimals=3) -> list[float]:
    """Round to ``decimals`` and rebalance so they sum to an integer."""
    fstr = "%%.%df" % decimals
    charges_dec = [float(fstr % q) for q in q_list]

    if net_charge is None:
        net_charge = _snap_to_int(sum(charges_dec), tolerance=0.15)
        if net_charge is None:
            msg = (
                "net charge could not be predicted from input q_list. "
                "(residual is beyond tolerance) "
                "Please set the net_charge argument directly"
            )
            raise RuntimeError(msg)
    elif type(net_charge) != int:
        raise TypeError("net charge must be an integer")

    surplus = net_charge - sum(charges_dec)
    surplus_int = _snap_to_int(10**decimals * surplus)
    if surplus_int == 0:
        return charges_dec

    weights = [abs(q) for q in q_list]
    surplus_int_splits = divide_int_gracefully(surplus_int, weights)
    for i, increment in enumerate(surplus_int_splits):
        charges_dec[i] += 10**-decimals * increment
    return charges_dec


def get_updated_positions(monomer, new_positions: dict):
    """Apply ``new_positions`` to a copy of ``monomer.rdkit_mol`` and let RDKit
    refresh any Hs one or two bonds away."""
    h_to_update = set()
    mol = Chem.Mol(monomer.rdkit_mol)
    conformer = mol.GetConformer()

    for n1 in (mol.GetAtomWithIdx(idx) for idx in new_positions):
        for n2 in n1.GetNeighbors():
            if n2.GetAtomicNum() == 1:
                h_to_update.add(n2.GetIdx())
            else:
                if n2.GetIdx() not in new_positions:
                    h_to_update.update(
                        set(
                            n2h.GetIdx()
                            for n2h in n2.GetNeighbors()
                            if n2h.GetAtomicNum() == 1
                        )
                    )
    h_to_update -= set(new_positions)

    for index in new_positions:
        x, y, z = new_positions[index]
        p = Point3D(float(x), float(y), float(z))
        conformer.SetAtomPosition(index, p)
    if h_to_update:
        update_H_positions(mol, list(h_to_update))
    return mol.GetConformer().GetPositions()


def update_H_positions(mol: Chem.Mol, indices_to_update: list[int]) -> None:
    """Re-compute positions of selected hydrogens already in ``mol`` (no
    chirality guarantee)."""
    conf = mol.GetConformer()
    tmpmol = Chem.RWMol(mol)
    to_del = {}
    to_add_h = []
    for h_index in indices_to_update:
        atom = tmpmol.GetAtomWithIdx(h_index)
        if atom.GetAtomicNum() != 1:
            raise RuntimeError("only H positions can be updated")
        heavy_neighbors = [
            neigh_atom
            for neigh_atom in atom.GetNeighbors()
            if neigh_atom.GetAtomicNum() != 1
        ]
        if len(heavy_neighbors) != 1:
            raise RuntimeError(
                f"hydrogens must have 1 non-H neighbor, got {len(heavy_neighbors)}"
            )
        to_add_h.append(heavy_neighbors[0])
        to_del[h_index] = heavy_neighbors[0]
    for i in sorted(to_del, reverse=True):
        tmpmol.RemoveAtom(i)
        to_del[i].SetNumExplicitHs(to_del[i].GetNumExplicitHs() + 1)
    to_add_h = list(set([atom.GetIdx() for atom in to_add_h]))
    tmpmol = tmpmol.GetMol()
    tmpmol.UpdatePropertyCache()
    Chem.SanitizeMol(tmpmol)
    tmpmol = Chem.AddHs(tmpmol, onlyOnAtoms=to_add_h, addCoords=True)
    tmpconf = tmpmol.GetConformer()
    used_h = set()
    to_del = {k: atom.GetIdx() for k, atom in to_del.items()}
    for h_index, parent in to_del.items():
        for atom in tmpmol.GetAtomWithIdx(parent).GetNeighbors():
            has_new_position = atom.GetIdx() >= mol.GetNumAtoms() - len(to_del)
            if atom.GetAtomicNum() == 1 and has_new_position:
                if atom.GetIdx() not in used_h:
                    conf.SetAtomPosition(
                        h_index, tmpconf.GetAtomPosition(atom.GetIdx())
                    )
                    used_h.add(atom.GetIdx())
                    break

    if len(used_h) != len(to_del):
        raise RuntimeError(
            f"Updated {len(used_h)} H positions but deleted {len(to_del)}"
        )


def _delete_residues(res_to_delete, raw_input_mols):
    """In-place delete entries from ``raw_input_mols`` keyed by residue id."""
    if res_to_delete is None:
        return
    missing = set()
    for res in res_to_delete:
        if res not in raw_input_mols:
            missing.add(res)
        raw_input_mols.pop(res, None)
    if len(missing) > 0:
        msg = "can't find the following residues to delete: " + " ".join(missing)
        raise ValueError(msg)


def handle_parsing_situations(
    unmatched_res,
    unparsed_res,
    allow_bad_res,
    res_missed_altloc,
    res_needed_altloc,
):
    err = ""
    if unparsed_res:
        msg = f"- Parsing failed for: {unparsed_res}."
        if not allow_bad_res:
            err += msg + eol
        else:
            msg += " Ignored due to allow_bad_res."
            logger.warning(msg)

    if unmatched_res:
        msg = f"- Template matching failed for: {list(unmatched_res)}"
        if not allow_bad_res:
            err += msg + eol
        else:
            msg += " Ignored due to allow_bad_res."
            logger.warning(msg)

    if err:
        err += "These residues can be ignored with option allow_bad_res." + eol

    if res_needed_altloc:
        msg = f"- Residues with alternate location: {res_needed_altloc}" + eol
        msg += "Either specify an altloc for each with option wanted_altloc" + eol
        msg += "or a general default altloc with option default_altloc."
        err += msg

    if res_missed_altloc:
        msg = f"- Requested altlocs not found for: {res_missed_altloc}." + eol
        err += msg

    if err:
        recs = (
            "1. (for batch processing) Use -a/--allow_bad_res to automatically remove residues"
            + eol
            + "that do not match templates, and --default_altloc to set"
            + eol
            + "a default altloc variant. Use these at your own risk."
            + eol
            + "" + eol
            + "2. (processing individual structure) Inspecting and fixing the input structure is recommended."
            + eol
            + "Use --wanted_altloc to set variants for specific residues."
        )
        raise PolymerCreationError(err, recs)
