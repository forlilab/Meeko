"""Monomer: one subunit of a polymer (often called a residue)."""

from typing import Any, Optional

from rdkit.Chem import rdMolInterchange

from ..molsetup import MoleculeSetup, RDKitMoleculeSetup
from ..utils.jsonutils import (
    BaseJSONParsable,
    convert_to_int_keyed_dict,
    rdkit_mol_from_json,
    serialize_optional,
)
from ..utils.pdbutils import PDBAtomInfo
from .templates import ResidueTemplate
from .utils import find_graph_paths, rectify_charges


class Monomer(BaseJSONParsable):
    """Subunit of a polymer (residue).

    Attributes
    ----------
    raw_rdkit_mol: Chem.Mol
        Defines element + connectivity. Bond orders / formal charges may be
        wrong, hydrogens may be missing. Carries the input atom positions.
    rdkit_mol: Chem.Mol
        Copy of the molecule from a ``ResidueTemplate``, with positions from
        raw_rdkit_mol. All hydrogens are real atoms except those at links
        to adjacent residues.
    mapidx_to_raw: dict (int -> int)
        Atom-index map: rdkit_mol → raw_rdkit_mol.
    input_resname: str
    template_key: str
    atom_names: list[str]
    padded_mol: Chem.Mol
        Molecule padded with ``ResiduePadder``.
    molsetup: RDKitMoleculeSetup
    molsetup_mapidx: dict (int -> int)
        padded_mol → rdkit_mol atom-index map.
    template: ResidueTemplate
    """

    def __init__(
        self,
        raw_input_mol,
        rdkit_mol,
        mapidx_to_raw,
        input_resname=None,
        template_key=None,
        atom_names=None,
    ):
        self.raw_rdkit_mol = raw_input_mol
        self.rdkit_mol = rdkit_mol
        self.mapidx_to_raw = mapidx_to_raw
        self.residue_template_key = template_key
        self.input_resname = input_resname
        self.atom_names = atom_names

        self.padded_mol = None
        self.molsetup = None
        self.molsetup_mapidx = None
        self.is_flexres_atom = None
        self.is_movable = False
        self.mapidx_from_raw = self._invert_mapping(self.mapidx_to_raw)

        self.template = None
        self.template_charge = None

    @staticmethod
    def _invert_mapping(mapping):
        if mapping is None:
            return None
        inverted = {}
        for key, value in mapping.items():
            if value in inverted:
                raise RuntimeError(f"Mapping is not invertible: {mapping}")
            inverted[value] = key
        return inverted

    @classmethod
    def json_encoder(cls, obj: "Monomer") -> Optional[dict[str, Any]]:
        try:
            molsetup = serialize_optional(RDKitMoleculeSetup.json_encoder, obj.molsetup)
        except KeyError:
            molsetup = serialize_optional(MoleculeSetup.json_encoder, obj.molsetup)

        return {
            "raw_rdkit_mol": serialize_optional(
                rdMolInterchange.MolToJSON, obj.raw_rdkit_mol
            ),
            "rdkit_mol": serialize_optional(
                rdMolInterchange.MolToJSON, obj.rdkit_mol
            ),
            "mapidx_to_raw": obj.mapidx_to_raw,
            "residue_template_key": obj.residue_template_key,
            "input_resname": obj.input_resname,
            "atom_name": obj.atom_names,
            "mapidx_from_raw": obj.mapidx_from_raw,
            "padded_mol": serialize_optional(
                rdMolInterchange.MolToJSON, obj.padded_mol
            ),
            "molsetup": molsetup,
            "is_flexres_atom": obj.is_flexres_atom,
            "is_movable": obj.is_movable,
            "molsetup_mapidx": obj.molsetup_mapidx,
            "template": serialize_optional(ResidueTemplate.json_encoder, obj.template),
        }

    expected_json_keys = frozenset(
        {
            "raw_rdkit_mol",
            "rdkit_mol",
            "mapidx_to_raw",
            "residue_template_key",
            "input_resname",
            "atom_name",
            "padded_mol",
            "molsetup",
            "molsetup_mapidx",
            "is_flexres_atom",
            "is_movable",
            "mapidx_from_raw",
            "template",
        }
    )

    @classmethod
    def _decode_object(cls, obj: dict[str, Any]):
        raw_rdkit_mol = rdkit_mol_from_json(obj["raw_rdkit_mol"])
        rdkit_mol = rdkit_mol_from_json(obj["rdkit_mol"])
        padded_mol = rdkit_mol_from_json(obj["padded_mol"])

        molsetup = RDKitMoleculeSetup.from_dict(obj["molsetup"])
        if not isinstance(molsetup, RDKitMoleculeSetup):
            molsetup = MoleculeSetup.from_dict(obj["molsetup"])

        mapidx_to_raw = convert_to_int_keyed_dict(obj["mapidx_to_raw"])
        molsetup_mapidx = convert_to_int_keyed_dict(obj["molsetup_mapidx"])
        mapidx_from_raw = convert_to_int_keyed_dict(obj["mapidx_from_raw"])

        atom_name = cls.access_with_deprecated_key(
            obj, old_key="atom_names", new_key="atom_name"
        )
        monomer = cls(
            raw_input_mol=raw_rdkit_mol,
            rdkit_mol=rdkit_mol,
            mapidx_to_raw=mapidx_to_raw,
            input_resname=obj["input_resname"],
            template_key=obj["residue_template_key"],
            atom_names=atom_name,
        )

        monomer.padded_mol = padded_mol
        monomer.molsetup = molsetup
        monomer.molsetup_mapidx = molsetup_mapidx
        monomer.is_flexres_atom = obj["is_flexres_atom"]
        monomer.is_movable = obj["is_movable"]
        monomer.mapidx_from_raw = mapidx_from_raw
        if "template" in obj:
            monomer.template = ResidueTemplate.from_dict(obj["template"])
        else:
            monomer.template = None
        return monomer

    def set_atom_names(self, atom_names_list):
        if self.rdkit_mol is None:
            raise RuntimeError("can't set atom_names if rdkit_mol is not set yet")
        if len(atom_names_list) != self.rdkit_mol.GetNumAtoms():
            raise ValueError(
                f"{len(atom_names_list)=} differs from {self.rdkit_mol.GetNumAtoms()=}"
            )
        name_types = set([type(name) for name in atom_names_list])
        if name_types != {str}:
            raise ValueError(f"atom names must be str but {name_types=}")
        self.atom_names = atom_names_list

    def parameterize(self, mk_prep, residue_id, get_atomprop_from_raw: dict = None):
        if get_atomprop_from_raw:
            if any(
                not isinstance(prop_name, str)
                for prop_name in get_atomprop_from_raw.keys()
            ):
                raise ValueError(
                    f"Atom property name must be str. Got {prop_name} ({type(prop_name)}) instead! "
                )
            raw_mol = self.raw_rdkit_mol
            atoms_in_raw_mol = [atom for atom in raw_mol.GetAtoms()]
            mapidx_to_raw = self.mapidx_to_raw
            molsetup_mapidx = self.molsetup_mapidx
            for atom in self.padded_mol.GetAtoms():
                atom_idx_in_raw = mapidx_to_raw.get(
                    molsetup_mapidx.get(atom.GetIdx(), None), None
                )
                for prop_name, default_value in get_atomprop_from_raw.items():
                    if atom_idx_in_raw is not None:
                        prop_value = atoms_in_raw_mol[atom_idx_in_raw].GetProp(prop_name)
                    else:
                        prop_value = str(default_value)
                    atom.SetProp(prop_name, prop_value)

        molsetups = mk_prep(
            mol=self.padded_mol,
            template_key=self.residue_template_key,
            template_charge=self.template_charge,
        )
        if len(molsetups) != 1:
            raise NotImplementedError(f"need 1 molsetup but got {len(molsetups)}")
        molsetup = molsetups[0]
        self.molsetup = molsetup
        self.is_flexres_atom = [False for _ in molsetup.atoms]

        for atom in molsetup.atoms:
            if atom.index not in self.molsetup_mapidx:
                atom.is_ignore = True

        if mk_prep.charge_model == "zero":
            net_charge = 0
        else:
            rdkit_mol = self.rdkit_mol
            net_charge = sum(
                [atom.GetFormalCharge() for atom in rdkit_mol.GetAtoms()]
            )
        not_ignored_idxs = []
        charges = []
        for atom in molsetup.atoms:
            if atom.index in self.molsetup_mapidx:
                charges.append(atom.charge)
                not_ignored_idxs.append(atom.index)
        charges = rectify_charges(charges, net_charge, decimals=3)

        for i, j in enumerate(not_ignored_idxs):
            molsetup.atoms[j].charge = charges[i]
        self._set_pdbinfo(residue_id)

        if self.is_movable:
            self.flexibilize(mk_prep)

    def flexibilize(self, mk_prep):
        inv = {j: i for i, j in self.molsetup_mapidx.items()}
        link_atoms = [inv[i] for i in self.template.link_labels]
        if len(link_atoms) == 0:
            raise RuntimeError(
                "can't define a sidechain without bonds to other residues"
            )
        graph = {atom.index: atom.graph for atom in self.molsetup.atoms}
        for i in range(len(link_atoms) - 1):
            start_node = link_atoms[i]
            end_nodes = [k for (j, k) in enumerate(link_atoms) if j != i]
            backbone_paths = find_graph_paths(graph, start_node, end_nodes)
            for path in backbone_paths:
                for x in range(len(path) - 1):
                    idx1 = min(path[x], path[x + 1])
                    idx2 = max(path[x], path[x + 1])
                    self.molsetup.bond_info[(idx1, idx2)].rotatable = False
        self.is_movable = True

        mk_prep.calc_flex(self.molsetup, root_atom_index=link_atoms[0])

        molsetup = self.molsetup
        graph = molsetup.flexibility_model["rigid_body_graph"]
        root_body_idx = molsetup.flexibility_model["root"]
        conn = molsetup.flexibility_model["rigid_body_connectivity"]
        rigid_index_by_atom = molsetup.flexibility_model["rigid_index_by_atom"]
        for other_body_idx in graph[root_body_idx]:
            root_link_atom_idx = conn[(root_body_idx, other_body_idx)][0]
            for atom_idx, body_idx in rigid_index_by_atom.items():
                if body_idx != root_body_idx or atom_idx == root_link_atom_idx:
                    self.is_flexres_atom[atom_idx] = True

    def rigidify(self, mk_prep, residue_id):
        self.is_movable = False
        self.parameterize(mk_prep, residue_id)
        self.is_flexres_atom = [False for _ in self.molsetup.atoms]

    def _set_pdbinfo(self, residue_id):
        not_ignored_idxs = []
        for atom in self.molsetup.atoms:
            if atom.index in self.molsetup_mapidx:
                not_ignored_idxs.append(atom.index)
        chain, resnum = residue_id.split(":")
        if resnum[-1].isalpha():
            icode = resnum[-1]
            resnum = resnum[:-1]
        else:
            icode = ""
        if self.atom_names is None:
            atom_names = ["" for _ in not_ignored_idxs]
        else:
            atom_names = self.atom_names
        for i, j in enumerate(not_ignored_idxs):
            atom_name = atom_names[self.molsetup_mapidx[j]]
            self.molsetup.atoms[j].pdbinfo = PDBAtomInfo(
                atom_name, self.input_resname, int(resnum), icode, chain
            )
