"""RDKit-coupled operations on MoleculeSetup.

This module contains the chemistry functions that need an RDKit
``Chem.Mol`` to do their work. They previously lived as methods on
``RDKitMoleculeSetup`` (which also required a ``MoleculeSetupExternalToolkit``
ABC mixin to declare the contract). Pulling them out as free functions
removes the inheritance gymnastics; ``RDKitMoleculeSetup`` keeps thin
method wrappers so external callers see no API change.
"""

from __future__ import annotations

import json
import logging
import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np
from rdkit import Chem

from ..utils import rdkitutils, utils
from .ring import Ring

if TYPE_CHECKING:  # avoid circular import at module load
    from .setup import RDKitMoleculeSetup

try:
    from misctools import StereoIsomorphism  # noqa: F401
    _has_misctools = True
except ImportError as _import_misctools_error:
    _has_misctools = False
    _stored_import_error = _import_misctools_error

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Pure RDKit utilities (don't touch the setup at all)
# --------------------------------------------------------------------------

def has_implicit_hydrogens(mol: Chem.Mol) -> bool:
    """Return True if ``mol`` has any atom whose total H count exceeds its
    explicit H neighbor count (mirrors RDKit's internal ``needsHs``)."""
    for atom in mol.GetAtoms():
        nr_H_neighbors = 0
        for neighbor in atom.GetNeighbors():
            nr_H_neighbors += int(neighbor.GetAtomicNum() == 1)
        if atom.GetTotalNumHs(includeNeighbors=False) > nr_H_neighbors:
            return True
    return False


def get_symmetries_for_rmsd(mol: Chem.Mol, max_matches: int = 17):
    mol_noHs = Chem.RemoveHs(mol)
    matches = mol.GetSubstructMatches(
        mol_noHs, uniquify=False, maxMatches=max_matches
    )
    if len(matches) == max_matches:
        molname = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        warnings.warn(
            "Found the maximum nr of matches (%d) in get_symmetries_for_rmsd"
            % max_matches,
            RuntimeWarning,
        )
        warnings.warn(
            'Maybe this molecule is "too" symmetric? %s %s'
            % (molname, Chem.MolToSmiles(mol_noHs)),
            RuntimeWarning,
        )
    return matches


# --------------------------------------------------------------------------
# Setup factory + per-setup operations
# --------------------------------------------------------------------------

def from_rdkit_mol(
    cls,
    mol: Chem.Mol,
    keep_chorded_rings: bool = False,
    keep_equivalent_rings: bool = False,
    charge_model: str = "gasteiger",
    read_charges_from_prop: Optional[str] = None,
    conformer_id: int = -1,
    compute_charges: bool = False,
    template_key: Optional[str] = None,
    template_charge: Optional[dict] = None,
):
    """Build a fresh ``RDKitMoleculeSetup`` from an RDKit ``Chem.Mol``.

    The ``cls`` argument is the ``RDKitMoleculeSetup`` class itself
    (passed in by the method wrapper) so this function does not need to
    import it directly and trigger a circular import.
    """
    if has_implicit_hydrogens(mol):
        raise ValueError("RDKit molecule has implicit Hs. Need explicit Hs.")
    if mol.GetNumConformers() == 0:
        raise ValueError(
            "RDKit molecule does not have a conformer. Need 3D coordinates."
        )
    rdkit_conformer = mol.GetConformer(conformer_id)
    if not rdkit_conformer.Is3D():
        warnings.warn(
            "RDKit molecule not labeled as 3D. This warning won't show again.",
            RuntimeWarning,
        )
        cls.warned_not3D = True
    if mol.GetNumConformers() > 1 and conformer_id == -1:
        warnings.warn(
            "RDKit molecule has multiple conformers. Considering only the first one.",
            RuntimeWarning,
        )
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError(
            f"RDKit molecule has {len(Chem.GetMolFrags(mol))} fragments. Must have 1."
        )
    if mol.HasQuery():
        raise ValueError(
            "RDKit molecule has query. Check exotic fields (atom or bond) in SDF."
        )

    setup = cls()
    setup.mol = mol
    setup.atom_true_count = mol.GetNumAtoms()
    setup.compute_charges = compute_charges
    setup.name = get_mol_name(setup)
    coords = rdkit_conformer.GetPositions()
    init_atom(
        setup,
        charge_model,
        read_charges_from_prop,
        coords,
        template_key=template_key,
        template_charge=template_charge,
    )
    init_bond(setup)
    perceive_rings(setup, keep_chorded_rings, keep_equivalent_rings)
    setup.modified_atom_positions = []
    return setup


def init_atom(
    setup: "RDKitMoleculeSetup",
    charge_model: str,
    read_charges_from_prop: Optional[str],
    coords,
    template_key: Optional[str] = None,
    template_charge: Optional[dict] = None,
) -> None:
    if template_key is None and setup.compute_charges is False:
        raise ValueError(
            "Template key is none and compute_charges is false. Something has gone terribly wrong. "
        )

    temp_compute_charges = None
    if charge_model == "read":
        temp_compute_charges = setup.compute_charges
        setup.compute_charges = True

    if setup.compute_charges:
        charges = calculate_charges(setup, charge_model, read_charges_from_prop)
    else:
        charges = get_charges_from_template(setup, charge_model, template_charge)

    if temp_compute_charges is not None:
        setup.compute_charges = temp_compute_charges

    for a in setup.mol.GetAtoms():
        idx = a.GetIdx()
        setup.add_atom(
            atom_index=idx,
            pdbinfo=rdkitutils.getPdbInfoNoNull(a),
            charge=charges[idx],
            coord=coords[idx],
            atomic_num=a.GetAtomicNum(),
            is_ignore=False,
        )


def calculate_charges(
    setup: "RDKitMoleculeSetup",
    charge_model: str,
    read_charges_from_prop: Optional[str],
):
    if charge_model == "gasteiger":
        if read_charges_from_prop is not None:
            raise ValueError(
                "Conflicting options: charge_model cannot be gasteiger and read_charges_from_prop cannot both be set."
            )
        try:
            charges = rdkitutils.compute_gasteiger_charges(setup.mol)
        except Exception as e:
            print("gasteiger charge computation failed with: ")
            print(e)
    elif charge_model == "nagl":
        if read_charges_from_prop is not None:
            raise ValueError(
                "Conflicting options: charge_model cannot be nagl and read_charges_from_prop cannot both be set."
            )
        try:
            from openff.toolkit import Molecule
        except ImportError:
            print("A recent version of OpenFF is required for NAGL charges")
        mol_off = Molecule.from_rdkit(
            setup.mol, allow_undefined_stereo=True, hydrogens_are_explicit=True
        )
        try:
            mol_off.assign_partial_charges(
                partial_charge_method="openff-gnn-am1bcc-1.0.0.pt"
            )
            charges = mol_off.partial_charges.magnitude.tolist()
        except Exception as e:
            print("NAGL charge computation failed with with exception:")
            print(e)
            print("Make sure you've installed the latest version of openff")
    elif read_charges_from_prop is not None:
        if not isinstance(read_charges_from_prop, str) or not read_charges_from_prop:
            raise ValueError(
                f"Invalid atom property name for read_charges_from_prop: expected a nonempty string (str), but got {type(read_charges_from_prop).__name__} instead. "
            )
        charges = [
            float(atom.GetProp(read_charges_from_prop))
            if atom.HasProp(read_charges_from_prop)
            else None
            for atom in setup.mol.GetAtoms()
        ]
        if None in charges:
            for idx, charge in enumerate(charges):
                if charge is None:
                    logger.error(f"Charge at index {idx} is None.")
            raise ValueError(
                f"The list of charges based on atom property name {read_charges_from_prop} contains None. "
            )
    else:
        charges = [0.0] * setup.mol.GetNumAtoms()
    return charges


def get_charges_from_template(
    setup: "RDKitMoleculeSetup",
    charge_model: str,
    template_charge: dict,
):
    if setup.mol is None:
        raise ValueError("No rdkit mol generated for current residue. ")
    template_mol = Chem.MolFromMolBlock(template_charge["molblock"], removeHs=False)
    setup.template_mol = template_mol
    match_indices = list(template_mol.GetSubstructMatch(setup.mol))
    if len(match_indices) != setup.mol.GetNumAtoms():
        l1 = len(match_indices)
        l2 = setup.mol.GetNumAtoms()
        raise ValueError(
            f"Mismatch between template mol ({l1} atoms) and padded mol ({l2} atoms). Abandoning prep!"
        )
    match charge_model:
        case "nagl":
            charges = template_charge["nagl_charges"]
        case "espaloma":
            charges = template_charge["espaloma_charges"]
        case "gasteiger":
            charges = template_charge["gasteiger_charges"]
        case "zero":
            charges = [0.0] * setup.mol.GetNumAtoms()
        case _:
            raise ValueError(
                "Incompatible charge model requested from charge template. Use --recompute_charges"
            )
    charges = np.array(charges)
    charges = [float(x) for x in charges[match_indices]]
    return charges


def init_bond(setup: "RDKitMoleculeSetup") -> None:
    for b in setup.mol.GetBonds():
        idx1 = b.GetBeginAtomIdx()
        idx2 = b.GetEndAtomIdx()
        rotatable = int(b.GetBondType()) == 1
        setup.add_bond(idx1, idx2, rotatable=rotatable)


def find_pattern(setup: "RDKitMoleculeSetup", smarts: str, uniquify: bool = False,
                 max_matches: int = int(1e7)):
    p = Chem.MolFromSmarts(smarts)
    return setup.mol.GetSubstructMatches(
        p, uniquify=uniquify, maxMatches=max_matches
    )


def get_mol_name(setup: "RDKitMoleculeSetup"):
    if setup.mol.HasProp("_Name"):
        return setup.mol.GetProp("_Name")
    return None


def get_num_mol_atoms(setup: "RDKitMoleculeSetup") -> int:
    return setup.mol.GetNumAtoms()


def get_equivalent_atoms(setup: "RDKitMoleculeSetup"):
    return list(Chem.CanonicalRankAtoms(setup.mol, breakTies=False))


def perceive_rings(
    setup: "RDKitMoleculeSetup", keep_chorded_rings: bool, keep_equivalent_rings: bool
) -> None:
    old_graph = {atom.index: atom.graph for atom in setup.atoms}
    hjk_ring_detection = utils.HJKRingDetection(old_graph)
    rings = hjk_ring_detection.scan(keep_chorded_rings, keep_equivalent_rings)
    for ring_atom_indices in rings:
        setup.rings[ring_atom_indices] = Ring(ring_atom_indices)


def get_conformer_with_modified_positions(
    setup: "RDKitMoleculeSetup", new_atom_positions
):
    new_mol = Chem.Mol(setup.mol)
    new_conformer = Chem.Conformer(setup.mol.GetConformer())
    is_set_list = [False] * setup.mol.GetNumAtoms()
    for atom_index, new_position in new_atom_positions.items():
        new_conformer.SetAtomPosition(atom_index, new_position)
        is_set_list[atom_index] = True
    new_mol.RemoveAllConformers()
    new_mol.AddConformer(new_conformer, assignId=True)
    for atom_index, is_set in enumerate(is_set_list):
        if not is_set and new_mol.GetAtomWithIdx(atom_index).GetAtomicNum() == 1:
            neighbors = new_mol.GetAtomWithIdx(atom_index).GetNeighbors()
            if len(neighbors) != 1:
                raise RuntimeError("Expected H to have one neighbors")
            Chem.SetTerminalAtomCoords(new_mol, atom_index, neighbors[0].GetIdx())
    return new_conformer


def get_mol_with_modified_positions(
    setup: "RDKitMoleculeSetup", new_atom_positions_list=None
):
    if new_atom_positions_list is None:
        new_atom_positions_list = setup.modified_atom_positions
    new_mol = Chem.Mol(setup.mol)
    new_mol.RemoveAllConformers()
    for new_atom_positions in new_atom_positions_list:
        conformer = get_conformer_with_modified_positions(setup, new_atom_positions)
        new_mol.AddConformer(conformer, assignId=True)
    return new_mol


def get_smiles_and_order(setup: "RDKitMoleculeSetup"):
    """Return ``(smiles, atom_index_map)`` for ``setup.mol`` after removing Hs."""
    mol_no_ignore = setup.mol
    ps = Chem.RemoveHsParameters()
    ps.removeWithQuery = True
    mol_noH = Chem.RemoveHs(mol_no_ignore, ps)
    atomic_num_mol_noH = [atom.GetAtomicNum() for atom in mol_noH.GetAtoms()]
    noH_to_H: list = []
    parents_of_hs: dict = {}
    for index, atom in enumerate(mol_no_ignore.GetAtoms()):
        if atom.GetAtomicNum() == 1:
            continue
        for i in range(len(noH_to_H), len(atomic_num_mol_noH)):
            if atomic_num_mol_noH[i] > 1:
                break
            h_atom = mol_noH.GetAtomWithIdx(len(noH_to_H))
            assert h_atom.GetAtomicNum() == 1
            neighbors = h_atom.GetNeighbors()
            assert len(neighbors) == 1
            parents_of_hs[len(noH_to_H)] = neighbors[0].GetIdx()
            noH_to_H.append("H")
        noH_to_H.append(index)
    extra_hydrogens = len(atomic_num_mol_noH) - len(noH_to_H)
    if extra_hydrogens > 0:
        assert set(atomic_num_mol_noH[len(noH_to_H) :]) == {1}
    for i in range(extra_hydrogens):
        h_atom = mol_noH.GetAtomWithIdx(len(noH_to_H))
        assert h_atom.GetAtomicNum() == 1
        neighbors = h_atom.GetNeighbors()
        assert len(neighbors) == 1
        parents_of_hs[len(noH_to_H)] = neighbors[0].GetIdx()
        noH_to_H.append("H")

    hs_by_parent: dict = {}
    for hidx, pidx in parents_of_hs.items():
        hs_by_parent.setdefault(pidx, [])
        hs_by_parent[pidx].append(hidx)
    for pidx, hidxs in hs_by_parent.items():
        siblings_of_h = [
            atom
            for atom in mol_no_ignore.GetAtomWithIdx(noH_to_H[pidx]).GetNeighbors()
            if atom.GetAtomicNum() == 1
        ]
        sortidx = [
            i
            for i, j in sorted(
                list(enumerate(siblings_of_h)), key=lambda x: x[1].GetIdx()
            )
        ]
        if len(hidxs) == len(siblings_of_h):
            for i, hidx in enumerate(hidxs):
                noH_to_H[hidx] = siblings_of_h[sortidx[i]].GetIdx()
        elif len(hidxs) < len(siblings_of_h):
            sibling_isotopes = [
                siblings_of_h[sortidx[i]].GetIsotope()
                for i in range(len(siblings_of_h))
            ]
            matches = []
            for i, sibling_isotope in enumerate(sibling_isotopes):
                for hidx in hidxs[len(matches) :]:
                    if mol_noH.GetAtomWithIdx(hidx).GetIsotope() == sibling_isotope:
                        matches.append(i)
                        break
            if len(matches) != len(hidxs):
                raise RuntimeError(
                    "Number of matched isotopes %d differs from query Hs: %d"
                    % (len(matches), len(hidxs))
                )
            for hidx, i in zip(hidxs, matches):
                noH_to_H[hidx] = siblings_of_h[sortidx[i]].GetIdx()
        else:
            raise RuntimeError(
                "nr of Hs in mol_noH bonded to an atom exceeds nr of Hs in mol_no_ignore"
            )

    smiles = Chem.MolToSmiles(mol_noH)
    order_string = mol_noH.GetProp("_smilesAtomOutputOrder")
    order_string = order_string.replace(",]", "]")
    order = json.loads(order_string)
    order = list(np.argsort(order))
    order = {noH_to_H[i]: order[i] + 1 for i in range(len(order))}

    for atom in mol_noH.GetAtoms():
        if atom.GetAtomicNum() == 1 and atom.GetIsotope() > 0:
            order.pop(atom.GetIdx())
    return smiles, order


def restrain_to(
    setup: "RDKitMoleculeSetup",
    target_mol,
    kcal_per_angstrom_square: float = 1.0,
    delay_angstroms: float = 2.0,
) -> None:
    if not _has_misctools:
        raise ImportError(_stored_import_error)
    from .restraint import Restraint
    stereo_isomorphism = StereoIsomorphism()
    mapping, idx = stereo_isomorphism(target_mol, setup.mol)
    lig_to_drive = {b: a for (a, b) in mapping}
    target_positions = target_mol.GetConformer().GetPositions()
    for atom_index in range(len(mapping)):
        target_xyz = target_positions[lig_to_drive[atom_index]]
        restraint = Restraint(
            atom_index, target_xyz, kcal_per_angstrom_square, delay_angstroms
        )
        setup.restraints.append(restraint)


# --------------------------------------------------------------------------
# Dihedral interaction helpers (used to live on MoleculeSetupExternalToolkit)
# --------------------------------------------------------------------------

def are_fourier_series_identical(series1: list, series2: list) -> bool:
    index_by_periodicity1 = {
        series1[index]["periodicity"]: index for index in range(len(series1))
    }
    index_by_periodicity2 = {
        series2[index]["periodicity"]: index for index in range(len(series2))
    }
    if index_by_periodicity1 != index_by_periodicity2:
        return False
    for periodicity in index_by_periodicity1:
        index1 = index_by_periodicity1[periodicity]
        index2 = index_by_periodicity2[periodicity]
        for key in ["k", "phase", "periodicity"]:
            if series1[index1][key] != series2[index2][key]:
                return False
    return True


def add_dihedral_interaction(setup: "RDKitMoleculeSetup", fourier_series) -> int:
    index = 0
    for existing_fs in setup.dihedral_interactions:
        if are_fourier_series_identical(existing_fs, fourier_series):
            return index
        index += 1
    safe_copy = json.loads(json.dumps(fourier_series))
    setup.dihedral_interactions.append(safe_copy)
    return index
