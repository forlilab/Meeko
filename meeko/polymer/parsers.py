"""Per-source parsers that produce ``raw_input_mols`` for ``Polymer``.

The PDB-string and PQR-string parsers share most of their pipeline:
both stream lines, accumulate ``AtomField`` records per residue key,
verify reskey→resname uniqueness, then build one RDKit mol per residue
via ``_aux_altloc_mol_build``. The deduplicated stages live in
``_build_residue_mols_from_blocks``.

The ProDy parser is structurally different (it walks the ProDy
hierarchy and lets ProDy yield one mol per residue) so it doesn't share
the common loop, but it lives here for cohesion.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from ..utils.rdkitutils import AtomField, _aux_altloc_mol_build

eol = "\n"
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def _add_if_new(to_dict, key, value, repeat_log) -> None:
    """Add ``key → value`` to ``to_dict``; if ``key`` already exists,
    record it in ``repeat_log`` (a set of "interrupted residues")."""
    if key in to_dict:
        repeat_log.add(key)
    else:
        to_dict[key] = value


def _build_residue_mols_from_blocks(
    blocks_by_residue: dict,
    reskey_to_resname: dict,
    wanted_altloc: Optional[dict],
    default_altloc: Optional[str],
    per_residue_postprocess: Optional[Callable] = None,
):
    """Given ``reskey → [AtomField, ...]`` blocks, build one RDKit mol per
    residue and assemble ``raw_input_mols``.

    ``per_residue_postprocess(reskey, mol)`` is invoked just after each mol
    is built, useful for attaching PQR charges/radii.
    """
    violations = {k: v for k, v in reskey_to_resname.items() if len(v) != 1}
    if violations:
        msg = "each residue key must have exactly 1 resname" + eol
        msg += f"but got {violations=}"
        raise ValueError(msg)

    if wanted_altloc is None:
        wanted_altloc = {}

    raw_input_mols = {}
    for reskey, atom_field_list in blocks_by_residue.items():
        resname = list(reskey_to_resname[reskey])[0]
        requested_altloc = wanted_altloc.get(reskey, None)
        try:
            mol, _, missed_altloc, needed_altloc = _aux_altloc_mol_build(
                atom_field_list, requested_altloc, default_altloc,
            )
        except Exception:
            raise RuntimeError(
                f"unable to build rdkit mol for residue {resname} corresponding to key {reskey}"
            )
        if per_residue_postprocess is not None:
            per_residue_postprocess(reskey, mol)
        raw_input_mols[reskey] = (mol, resname, missed_altloc, needed_altloc)
    return raw_input_mols


# ---------------------------------------------------------------------------
# PDB parser
# ---------------------------------------------------------------------------

def pdb_to_residue_mols(
    pdb_string: str,
    wanted_altloc: Optional[dict[str, str]] = None,
    default_altloc: Optional[str] = None,
) -> dict:
    """Parse a PDB string into ``raw_input_mols``."""
    blocks_by_residue: dict = {}
    reskey_to_resname: dict = {}
    reskey = None
    buffered_reskey = None
    interrupted_residues: set = set()
    pdb_block: list = []

    for line in pdb_string.splitlines(True):
        if line.startswith("TER") and reskey is not None:
            _add_if_new(blocks_by_residue, reskey, pdb_block, interrupted_residues)
            blocks_by_residue[reskey] = pdb_block
            pdb_block = []
            reskey = None
            buffered_reskey = None
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atomname = line[12:16].strip()
            altloc = line[16:17].strip()
            resname = line[17:20].strip()
            chainid = line[21:22].strip()
            resnum = int(line[22:26].strip())
            icode = line[26:27].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            element = line[76:78].strip()
            reskey = f"{chainid}:{resnum}{icode}"
            reskey_to_resname.setdefault(reskey, set())
            reskey_to_resname[reskey].add(resname)
            atom = AtomField(
                atomname, altloc, resname, chainid,
                resnum, icode, x, y, z, element,
            )

            if reskey == buffered_reskey:
                pdb_block.append(atom)
            else:
                if buffered_reskey is not None:
                    _add_if_new(
                        blocks_by_residue,
                        buffered_reskey,
                        pdb_block,
                        interrupted_residues,
                    )
                buffered_reskey = reskey
                pdb_block = [atom]

    if pdb_block:
        _add_if_new(blocks_by_residue, reskey, pdb_block, interrupted_residues)

    if interrupted_residues:
        raise ValueError(f"interrupted residues in PDB: {interrupted_residues}")

    return _build_residue_mols_from_blocks(
        blocks_by_residue,
        reskey_to_resname,
        wanted_altloc,
        default_altloc,
    )


# ---------------------------------------------------------------------------
# PQR parser
# ---------------------------------------------------------------------------

_PQR_NON_ATOM_TOKENS = frozenset({
    "REMARK", "TER", "END", "HEADER", "TITLE", "COMPND", "SOURCE",
    "KEYWDS", "EXPDTA", "AUTHOR", "REVDAT", "JRNL",
})


def _get_pqr_atom_items(pqr_line: str):
    """Tokenize a PQR line; return None for non-atom lines."""
    items = [w.strip() for w in pqr_line.split()]
    if not items:
        return None
    token = items.pop(0)
    if token in _PQR_NON_ATOM_TOKENS:
        return None
    if token in ("ATOM", "HETATM"):
        return items
    if token[:4] == "ATOM":
        return token[4:] + items
    if token[:6] == "HETATM":
        return token[6:] + items
    raise ValueError(f"Unable to parse PQR line: {pqr_line}")


def _atom_from_pqr_items(atom_pqr_items: list[str]):
    """Build an ``AtomField`` plus charge/radius from PQR-format token list."""
    if not atom_pqr_items:
        return None

    atom_pqr_items.pop(0)  # serial — not used downstream
    atomname = atom_pqr_items.pop(0)
    element = next((c for c in atomname if c.isalpha()), None)
    if element is None:
        raise ValueError(f"Unable to parse element from PQR atomname: {atomname}")
    element = element.upper()

    altloc = ""
    resname = atom_pqr_items.pop(0)

    token = atom_pqr_items.pop(0)
    chainid = ""
    try:
        resnum = int(token)
    except ValueError:
        chainid = token
        resnum = int(atom_pqr_items.pop(0))

    token = atom_pqr_items.pop(0)
    icode = ""
    try:
        x = float(token)
    except ValueError:
        icode = token
        x = float(atom_pqr_items.pop(0))

    y = float(atom_pqr_items.pop(0))
    z = float(atom_pqr_items.pop(0))
    charge = float(atom_pqr_items.pop(0))
    radius = float(atom_pqr_items.pop(0))

    return (
        AtomField(
            atomname, altloc, resname, chainid,
            resnum, icode, x, y, z, element,
        ),
        charge,
        radius,
    )


def pqr_to_residue_mols(pqr_string: str) -> dict:
    """Parse a PQR string into ``raw_input_mols``, also attaching
    ``PQRCharge`` and ``PQRRadius`` properties to each atom of the
    resulting RDKit mols.
    """
    blocks_by_residue: dict = {}
    blocks_qr: dict = {}
    reskey_to_resname: dict = {}
    reskey = None
    buffered_reskey = None
    interrupted_residues: set = set()
    pdb_block: list = []
    block_qr: list = []

    for line in pqr_string.splitlines(True):
        pqr_items = _get_pqr_atom_items(line)
        if pqr_items is None and reskey is not None:
            _add_if_new(blocks_by_residue, reskey, pdb_block, interrupted_residues)
            blocks_by_residue[reskey] = pdb_block
            blocks_qr[reskey] = block_qr
            pdb_block = []
            block_qr = []
            reskey = None
            buffered_reskey = None
        if pqr_items:
            atom, pqr_charge, pqr_radius = _atom_from_pqr_items(pqr_items)
            reskey = f"{atom.chain}:{atom.resnum}{atom.icode}"
            resname = atom.resname
            reskey_to_resname.setdefault(reskey, set())
            reskey_to_resname[reskey].add(resname)

            if reskey == buffered_reskey:
                pdb_block.append(atom)
                block_qr.append((pqr_charge, pqr_radius))
            else:
                if buffered_reskey is not None:
                    _add_if_new(
                        blocks_by_residue,
                        buffered_reskey,
                        pdb_block,
                        interrupted_residues,
                    )
                    blocks_qr[buffered_reskey] = block_qr
                buffered_reskey = reskey
                pdb_block = [atom]
                block_qr = [(pqr_charge, pqr_radius)]

    if pdb_block:
        _add_if_new(blocks_by_residue, reskey, pdb_block, interrupted_residues)
        blocks_qr[reskey] = block_qr

    if interrupted_residues:
        raise ValueError(f"interrupted residues in PDB: {interrupted_residues}")

    def attach_pqr_props(reskey, mol):
        for atom, (charge, radius) in zip(mol.GetAtoms(), blocks_qr[reskey]):
            atom.SetDoubleProp("PQRCharge", charge)
            atom.SetDoubleProp("PQRRadius", radius)

    return _build_residue_mols_from_blocks(
        blocks_by_residue,
        reskey_to_resname,
        wanted_altloc=None,    # PQR has no altloc concept
        default_altloc="",
        per_residue_postprocess=attach_pqr_props,
    )


# ---------------------------------------------------------------------------
# ProDy parser
# ---------------------------------------------------------------------------

def prody_to_residue_mols(
    prody_obj,
    wanted_altloc_dict: Optional[dict] = None,
    default_altloc: Optional[str] = None,
) -> dict:
    """Walk a ProDy AtomGroup/Selection and build one RDKit mol per residue.

    Imports ProDy adapters lazily so meeko still imports cleanly without
    ProDy installed (the call itself will fail if ProDy is missing).
    """
    from ..utils.prodyutils import prody_to_rdkit

    if wanted_altloc_dict is None:
        wanted_altloc_dict = {}
    raw_input_mols: dict = {}
    reskey_to_resname: dict = {}
    hierarchy = prody_obj.getHierView()
    for chain in hierarchy.iterChains():
        for res in chain.iterResidues():
            chain_id = str(res.getChid()).strip()
            res_name = str(res.getResname()).strip()
            res_num = int(res.getResnum())
            icode = str(res.getIcode()).strip()
            reskey = f"{chain_id}:{res_num}{icode}"
            reskey_to_resname.setdefault(reskey, set())
            reskey_to_resname[reskey].add(res_name)
            requested_altloc = wanted_altloc_dict.get(reskey, None)
            prody_mol, missed_altloc, needed_altloc = prody_to_rdkit(
                res,
                sanitize=False,
                requested_altloc=requested_altloc,
                default_altloc=default_altloc,
            )
            raw_input_mols[reskey] = (
                prody_mol, res_name, missed_altloc, needed_altloc,
            )
    return raw_input_mols
