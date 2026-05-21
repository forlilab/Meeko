"""ResiduePadder: pads a monomer with atoms from adjacent residues.

Also hosts the small atom-map helpers (``get_molAtomMapNumbers``,
``remove_unmapped_atoms_from_mol``, ``apply_atom_mappings``,
``remove_atoms_with_mapping``) and the RDKit-warning filter
``NoAtomMapWarning`` used internally.
"""

import logging
from typing import Any, Optional

from rdkit import Chem
from rdkit.Chem import rdChemReactions, rdFMCS

from ..utils.jsonutils import BaseJSONParsable, serialize_optional
from ..utils.rdkitutils import react_and_map

logger = logging.getLogger(__name__)
rdkit_logger = logging.getLogger("rdkit")


# ---------------------------------------------------------------------------
# Atom-map helpers
# ---------------------------------------------------------------------------

def get_molAtomMapNumbers(mol: Chem.Mol) -> set[int]:
    """Return the set of mapping numbers in a molecule."""
    return {
        atom.GetIntProp("molAtomMapNumber")
        for atom in mol.GetAtoms()
        if atom.HasProp("molAtomMapNumber")
    }


def remove_unmapped_atoms_from_mol(mol: Chem.Mol) -> Chem.Mol:
    """Remove atoms without mapping numbers from a molecule."""
    atoms_to_remove = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if not atom.HasProp("molAtomMapNumber")
    ]
    if len(atoms_to_remove) > 0:
        mol = Chem.RWMol(mol)
        for idx in sorted(atoms_to_remove, reverse=True):
            mol.RemoveAtom(idx)
        mol = mol.GetMol()
    return mol


def apply_atom_mappings(
    mcs_mol: Chem.Mol, original_mol: Chem.Mol
) -> list[Chem.Mol]:
    """Copy atom-map numbers from ``original_mol`` onto each substructure
    match of ``mcs_mol``; returns one decorated MCS mol per match."""
    matches = original_mol.GetSubstructMatches(mcs_mol)
    mapped_mcs_molecules = []
    for match in matches:
        rw_mcs_mol = Chem.RWMol(mcs_mol)
        for i, mcs_atom in enumerate(rw_mcs_mol.GetAtoms()):
            original_atom_idx = match[i]
            original_atom = original_mol.GetAtomWithIdx(original_atom_idx)
            if original_atom.HasProp("molAtomMapNumber"):
                mcs_atom.SetProp(
                    "molAtomMapNumber", original_atom.GetProp("molAtomMapNumber")
                )
        mapped_mcs_molecules.append(rw_mcs_mol.GetMol())
    return mapped_mcs_molecules


def remove_atoms_with_mapping(product: Chem.Mol, mapping_numbers: set) -> Chem.Mol:
    """Drop atoms whose mapping number is in ``mapping_numbers``."""
    editable_product = Chem.RWMol(product)
    atoms_to_remove = [
        atom.GetIdx()
        for atom in editable_product.GetAtoms()
        if atom.HasProp("molAtomMapNumber")
        and int(atom.GetProp("molAtomMapNumber")) in mapping_numbers
    ]
    for idx in sorted(atoms_to_remove, reverse=True):
        editable_product.RemoveAtom(idx)
    return editable_product.GetMol()


# ---------------------------------------------------------------------------
# RDKit warning suppression filter
# ---------------------------------------------------------------------------

class NoAtomMapWarning(logging.Filter):
    def filter(self, record):
        fields = record.getMessage().split()
        a = " ".join(fields[1:4]) == "product atom-mapping number"
        b = " ".join(fields[5:]) == "not found in reactants."
        is_atom_map_warning = a and b
        return not is_atom_map_warning


# ---------------------------------------------------------------------------
# ResiduePadder
# ---------------------------------------------------------------------------

class ResiduePadder(BaseJSONParsable):
    """Pad an RDKit molecule of a residue with parts from adjacent residues.

    Attributes
    ----------
    rxn : rdChemReactions.ChemicalReaction
        Single-reactant, single-product padding reaction.
    adjacent_smartsmol : Chem.Mol
        Mapped SMARTS for copying atom positions from the adjacent residue.
    adjacent_smartsmol_mapidx : dict
        Mapping number → atom index inside ``adjacent_smartsmol``.
    """

    def __init__(
        self,
        rxn_smarts: str,
        adjacent_res_smarts: str = None,
        auto_blunt: bool = False,
    ):
        self.rxn = self._validate_rxn_smarts(rxn_smarts)
        self.auto_blunt = auto_blunt

        if adjacent_res_smarts is None:
            self.adjacent_smartsmol = None
            self.adjacent_smartsmol_mapidx = None
            return

        self.adjacent_smartsmol = self._initialize_adj_smartsmol(adjacent_res_smarts)
        self._check_adj_smarts(self.rxn, self.adjacent_smartsmol)
        self.adjacent_smartsmol_mapidx = {
            atom.GetIntProp("molAtomMapNumber"): atom.GetIdx()
            for atom in self.adjacent_smartsmol.GetAtoms()
            if atom.HasProp("molAtomMapNumber")
        }

    @staticmethod
    def _validate_rxn_smarts(rxn_smarts: str) -> rdChemReactions.ChemicalReaction:
        rxn = rdChemReactions.ReactionFromSmarts(rxn_smarts)
        if rxn.GetNumReactantTemplates() != 1:
            raise ValueError(
                f"Expected 1 reactant, got {rxn.GetNumReactantTemplates()}."
            )
        if rxn.GetNumProductTemplates() != 1:
            raise ValueError(
                f"Expected 1 product, got {rxn.GetNumProductTemplates()}."
            )
        return rxn

    @staticmethod
    def _initialize_adj_smartsmol(adjacent_res_smarts: str) -> Chem.Mol:
        adjacent_smartsmol = Chem.MolFromSmarts(adjacent_res_smarts)
        if adjacent_smartsmol is None:
            raise RuntimeError("Invalid SMARTS pattern in adjacent_res_smarts")
        return adjacent_smartsmol

    @staticmethod
    def _check_adj_smarts(
        rxn: rdChemReactions.ChemicalReaction, adjacent_smartsmol: Chem.Mol
    ):
        reactant_ids = get_molAtomMapNumbers(rxn.GetReactantTemplate(0))
        product_ids = get_molAtomMapNumbers(rxn.GetProductTemplate(0))
        adjacent_ids = get_molAtomMapNumbers(adjacent_smartsmol)
        padding_ids = product_ids.difference(reactant_ids)
        is_ok = padding_ids == adjacent_ids
        if not is_ok:
            raise ValueError(
                f"SMARTS labels in adjacent_smartsmol ({adjacent_ids}) differ from "
                f"unmapped product labels in reaction ({padding_ids})"
            )

    def __call__(
        self,
        target_mol: Chem.Mol,
        adjacent_mol=None,
        target_required_atom_index=None,
        adjacent_required_atom_index=None,
    ):
        rxn = self.rxn
        if not self._check_target_mol(target_mol):
            logger.info(
                f"target_mol ({Chem.MolToSmiles(target_mol)}) is not fully compliant "
                f"with the template rxn ({rdChemReactions.ReactionToSmarts(self.rxn)})..."
            )
            reactant_smartsmol = rxn.GetReactantTemplate(0)
            reactant_ids = get_molAtomMapNumbers(reactant_smartsmol)

            fallback_reactant_smartsmol = Chem.MolFromSmarts(
                rdFMCS.FindMCS([reactant_smartsmol, target_mol]).smartsString
            )
            if fallback_reactant_smartsmol is None:
                raise RuntimeError(
                    "There is no common substructure between target_mol and the expected reactant. "
                )

            fallback_reactants = [
                reactant_mol
                for reactant_mol in apply_atom_mappings(
                    fallback_reactant_smartsmol, reactant_smartsmol
                )
                if any(
                    target_required_atom_index in match
                    for match in target_mol.GetSubstructMatches(reactant_mol)
                )
            ]
            if len(fallback_reactants) == 0:
                raise RuntimeError(
                    "The maximum common substructure between target_mol and the expected "
                    "reactant does not contain the expected linker atom with "
                    "target_required_atom_index."
                )

            fallback_reactant = fallback_reactants[0]
            fallback_reactant_ids = get_molAtomMapNumbers(fallback_reactant)
            skipping_ids = reactant_ids.difference(fallback_reactant_ids)
            fallback_product = remove_atoms_with_mapping(
                rxn.GetProductTemplate(0), skipping_ids
            )
            fallback_rxnsmarts = (
                f"{Chem.MolToSmarts(fallback_reactant)}>>"
                f"{Chem.MolToSmarts(fallback_product)}"
            )
            rxn = rdChemReactions.ReactionFromSmarts(fallback_rxnsmarts)
            logger.info(
                f"Switched from Template rxn ({rdChemReactions.ReactionToSmarts(self.rxn)}) "
                f"to Fallback rxn ({fallback_rxnsmarts})"
            )

        if adjacent_mol is not None:
            if self._check_adjacent_mol(
                self.adjacent_smartsmol, adjacent_mol, adjacent_required_atom_index
            ):
                adjacent_smartsmol = self.adjacent_smartsmol
            else:
                logger.info(
                    f"adjacent_mol ({Chem.MolToSmiles(adjacent_mol)}) is not fully compliant "
                    f"with the template adjacent_smarts ({Chem.MolToSmarts(self.adjacent_smartsmol)})..."
                )
                adjacent_smartsmol = remove_unmapped_atoms_from_mol(self.adjacent_smartsmol)
                if self._check_adjacent_mol(
                    adjacent_smartsmol, adjacent_mol, adjacent_required_atom_index
                ):
                    logger.info(
                        f"Switched from Template adjacent mol ({Chem.MolToSmarts(self.adjacent_smartsmol)}) "
                        f"to Fallback adjacent mol ({Chem.MolToSmarts(adjacent_smartsmol)})"
                    )
                else:
                    raise RuntimeError(
                        "adjacent_mol doesn't contain the mapped atoms in adjacent_smartsmol."
                    )

            hit = adjacent_mol.GetSubstructMatches(adjacent_smartsmol)[0]
            adjacent_smartsmol_mapidx = {
                atom.GetIntProp("molAtomMapNumber"): atom.GetIdx()
                for atom in adjacent_smartsmol.GetAtoms()
                if atom.HasProp("molAtomMapNumber")
            }

        filtr = NoAtomMapWarning()
        rdkit_logger.addFilter(filtr)
        outcomes = react_and_map((target_mol,), rxn)
        rdkit_logger.removeFilter(filtr)

        if target_required_atom_index is not None:
            outcomes = [
                (product, index_map)
                for (product, index_map) in outcomes
                if target_required_atom_index in index_map["atom_idx"]
            ]

        if len(outcomes) == 0:
            raise RuntimeError(
                "The padding reaction of target_mol has no outcome that contains the atom "
                "with target_required_atom_index"
            )
        elif len(outcomes) > 1:
            raise RuntimeError(
                "The padding reaction of target_mol has multiple outcomes that contain the "
                "atom with target_required_atom_index"
            )
        padded_mol, idxmap = outcomes[0]

        padding_heavy_atoms = [
            i
            for i, j in enumerate(idxmap["atom_idx"])
            if j is None and padded_mol.GetAtomWithIdx(i).GetAtomicNum() != 1
        ]
        mapidx = idxmap["atom_idx"]

        if adjacent_mol is None:
            padded_mol.UpdatePropertyCache()
            Chem.SanitizeMol(padded_mol)
            padded_h = Chem.AddHs(padded_mol, onlyOnAtoms=padding_heavy_atoms)
            mapidx += [None] * (padded_h.GetNumAtoms() - padded_mol.GetNumAtoms())
        else:
            adjacent_coords = adjacent_mol.GetConformer().GetPositions()
            for atom in adjacent_smartsmol.GetAtoms():
                if not atom.HasProp("molAtomMapNumber"):
                    continue
                j = atom.GetIntProp("molAtomMapNumber")
                k = idxmap["new_atom_label"].index(j)
                l = adjacent_smartsmol_mapidx[j]
                padded_mol.GetConformer().SetAtomPosition(
                    k, adjacent_coords[hit[l]]
                )
            padded_mol.UpdatePropertyCache()
            Chem.SanitizeMol(padded_mol)
            padded_h = Chem.AddHs(
                padded_mol, onlyOnAtoms=padding_heavy_atoms, addCoords=True
            )

        return padded_h, mapidx

    @staticmethod
    def _check_adjacent_mol(
        expected_adjacent_smartsmol: Chem.Mol,
        adjacent_mol: Chem.Mol,
        adjacent_required_atom_index: str,
    ):
        if expected_adjacent_smartsmol is None:
            raise RuntimeError(
                "adjacent_res_smarts must be initialized to support adjacent_mol."
            )
        hits = adjacent_mol.GetSubstructMatches(expected_adjacent_smartsmol)
        if adjacent_required_atom_index is not None:
            hits = [hit for hit in hits if adjacent_required_atom_index in hit]
            if len(hits) > 1:
                raise RuntimeError(
                    "adjacent_mol has multiple matches for adjacent_smartsmol."
                )
            elif len(hits) == 0:
                return False
        return True

    def _check_target_mol(self, target_mol: Chem.Mol):
        if target_mol.GetSubstructMatches(self.rxn.GetReactantTemplate(0)):
            return True
        return False

    @classmethod
    def json_encoder(cls, obj: "ResiduePadder") -> Optional[dict[str, Any]]:
        return {
            "rxn_smarts": rdChemReactions.ReactionToSmarts(obj.rxn),
            "adjacent_res_smarts": serialize_optional(
                Chem.MolToSmarts, obj.adjacent_smartsmol
            ),
            "auto_blunt": obj.auto_blunt,
        }

    expected_json_keys = {
        "rxn_smarts",
        "adjacent_res_smarts",
        "auto_blunt",
    }

    @classmethod
    def _decode_object(cls, obj: dict[str, Any]):
        adjacent_res_smarts = cls.access_with_deprecated_key(
            obj, old_key="adjacent_smarts", new_key="adjacent_res_smarts"
        )
        return cls(
            obj["rxn_smarts"],
            adjacent_res_smarts,
            obj.get("auto_blunt", False),
        )
