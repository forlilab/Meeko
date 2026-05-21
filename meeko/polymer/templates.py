"""Residue chemistry templates: ``ResidueTemplate`` and ``ResidueChemTemplates``."""

import json
import logging
import pathlib
from importlib.resources import files
from typing import Any, Optional

import rdkit.Chem
from rdkit import Chem
from rdkit.Chem import rdMolInterchange

from ..utils.jsonutils import (
    BaseJSONParsable,
    convert_to_int_keyed_dict,
    rdkit_mol_from_json,
)
from ..utils.rdkitutils import getPdbInfoNoNull
from .padder import ResiduePadder
from .utils import mapping_by_mcs

logger = logging.getLogger(__name__)
data_path = files("meeko") / "data"


class ResidueTemplate(BaseJSONParsable):
    """Template molecule for one residue: atoms, names, and link labels.

    Attributes
    ----------
    mol: RDKit Mol
        Explicit-H molecule. Atoms bonded to adjacent residues miss an H.
    link_labels: dict (int -> string)
        Indices of link atoms → label strings keying ``ResiduePadder`` choice.
    atom_names: list[str]
    """

    def __init__(self, smiles, link_labels=None, atom_names=None):
        self.link_labels = link_labels
        self.atom_names = atom_names

        ps = Chem.SmilesParserParams()
        ps.removeHs = False
        mol = Chem.MolFromSmiles(smiles, ps)
        self.check(mol, link_labels, atom_names)
        self.mol = mol

    @classmethod
    def json_encoder(cls, obj: "ResidueTemplate") -> Optional[dict[str, Any]]:
        return {
            "mol": rdMolInterchange.MolToJSON(obj.mol),
            "link_labels": obj.link_labels,
            "atom_name": obj.atom_names,
        }

    expected_json_keys = {"mol", "link_labels", "atom_name"}

    @classmethod
    def _decode_object(cls, obj: dict[str, Any]):
        deserialized_mol = rdkit_mol_from_json(obj.get("mol"))
        if deserialized_mol:
            mol_smiles = rdkit.Chem.MolToSmiles(deserialized_mol, canonical=False)
        else:
            mol_smiles = obj.get("smiles")
        link_labels = convert_to_int_keyed_dict(obj.get("link_labels"))
        atom_name = cls.access_with_deprecated_key(
            obj, old_key="atom_names", new_key="atom_name"
        )
        residue_template = cls(mol_smiles, None, atom_name)
        residue_template.link_labels = link_labels
        return residue_template

    def check(self, mol, link_labels, atom_names):
        have_implicit_hs = set()
        for atom in mol.GetAtoms():
            if atom.GetTotalNumHs() > 0:
                have_implicit_hs.add(atom.GetIdx())
        if link_labels is not None and set(link_labels) != have_implicit_hs:
            raise ValueError(
                f"expected any atom with non-real Hs ({have_implicit_hs}) to be in {link_labels=}"
            )
        if atom_names is None:
            return
        if len(atom_names) != mol.GetNumAtoms():
            raise ValueError(f"{len(atom_names)=} differs from {mol.GetNumAtoms()=}")

    def match(self, input_mol):
        mapping = mapping_by_mcs(self.mol, input_mol)
        mapping_inv = {value: key for (key, value) in mapping.items()}
        if len(mapping_inv) != len(mapping):
            raise RuntimeError(
                f"bug in atom indices, repeated value different keys? {mapping=}"
            )
        result = {
            "H": {"found": 0, "missing": 0, "excess": []},
            "heavy": {"found": 0, "missing": 0, "excess": 0},
        }
        for atom in self.mol.GetAtoms():
            element = "H" if atom.GetAtomicNum() == 1 else "heavy"
            key = "found" if atom.GetIdx() in mapping else "missing"
            result[element][key] += 1
        for atom in input_mol.GetAtoms():
            element = "H" if atom.GetAtomicNum() == 1 else "heavy"
            if atom.GetIdx() not in mapping_inv:
                if element == "H":
                    if atom.GetNeighbors():
                        nei_idx = atom.GetNeighbors()[0].GetIdx()
                        if nei_idx in mapping_inv:
                            result[element]["excess"].append(mapping_inv[nei_idx])
                        else:
                            result[element]["excess"].append(-1)
                    else:
                        monomer_info = getPdbInfoNoNull(atom)
                        if monomer_info:
                            logger.warning(
                                f"WARNING: Lone hydrogen is ignored: \n"
                                f"  {monomer_info} \n"
                            )
                        else:
                            logger.warning(
                                "WARNING: A lone hydrogen is ignored during monomer-template matching. \n"
                            )
                else:
                    result[element]["excess"] += 1
        return result, mapping


class ResidueChemTemplates(BaseJSONParsable):
    """Holds template data required to initialize ``Polymer``.

    Attributes
    ----------
    residue_templates: dict[str, ResidueTemplate]
    padders: dict[str, ResiduePadder]
        Keyed by link_label.
    ambiguous: dict[str, list[str]]
        Input residue name → candidate ``ResidueTemplate`` IDs.
    """

    def __init__(self, residue_templates, padders, ambiguous):
        self._check_missing_padders(residue_templates, padders)
        self._check_ambiguous_reskeys(residue_templates, ambiguous)
        self.residue_templates = residue_templates
        self.padders = padders
        self.ambiguous = ambiguous
        self.template_charges = ResidueChemTemplates._read_template_charge(
            "template_charges"
        )

    @classmethod
    def _read_template_charge(cls, filename):
        json_file = ResidueChemTemplates.lookup_filename(filename, data_path)
        template_charge = {}
        try:
            with open(json_file, "r") as file:
                template_charge = json.load(file)
        except FileNotFoundError:
            print("Error: The file 'template_charges.json' was not found.")
        except json.JSONDecodeError as e:
            print(
                f"Error: Failed to decode template_charges.json from the file: {e}"
            )
        return template_charge

    @classmethod
    def json_encoder(cls, obj: "ResidueChemTemplates") -> Optional[dict[str, Any]]:
        return {
            "residue_templates": {
                k: ResidueTemplate.json_encoder(v)
                for k, v in obj.residue_templates.items()
            },
            "ambiguous": obj.ambiguous,
            "padders": {
                k: ResiduePadder.json_encoder(v) for k, v in obj.padders.items()
            },
        }

    expected_json_keys = {
        "residue_templates",
        "ambiguous",
        "padders",
    }

    @classmethod
    def _decode_object(cls, obj: dict[str, Any]):
        templates = {
            k: ResidueTemplate.from_dict(v)
            for k, v in obj["residue_templates"].items()
        }
        padders = {k: ResiduePadder.from_dict(v) for k, v in obj["padders"].items()}
        return cls(templates, padders, obj["ambiguous"])

    def add_dict(self, data, overwrite=False):
        bad_keys = set(data) - {"ambiguous", "residue_templates", "padders"}
        if bad_keys:
            logging.warning(f"Ignore unexpected keys: {bad_keys}")
        new_ambiguous = data.get("ambiguous", {})
        if overwrite:
            self.ambiguous.update(new_ambiguous)
        else:
            new_ambiguous = {k: v.copy() for k, v in new_ambiguous.items()}
            new_ambiguous.update(self.ambiguous)
            self.ambiguous = new_ambiguous
        for key, value in data.get("residue_templates", {}).items():
            if overwrite or key not in self.residue_templates:
                res_template = ResidueTemplate.from_dict(value)
                self.residue_templates[key] = res_template
        for link_label, value in data.get("padders", {}).items():
            if overwrite or key not in self.padders:
                padder = ResiduePadder.from_dict(data)
                self.padders[link_label] = padder

    @staticmethod
    def lookup_filename(filename, data_path):
        p = pathlib.Path(filename)
        if not p.exists():
            if (data_path / p).exists():
                filename = str(data_path / p)
            elif (data_path / (p.name + ".json")).exists():
                filename = str(data_path / (p.name + ".json"))
            else:
                raise ValueError(
                    f"can't find {filename} in current dir or {data_path}"
                )
        return filename

    @classmethod
    def from_json_file(cls, filename):
        filename = cls.lookup_filename(filename, data_path)
        with open(filename) as f:
            jsonstr = f.read()
        alldata = json.loads(jsonstr)
        ambiguous = {k: v.copy() for k, v in alldata.get("ambiguous", {}).items()}
        residue_templates = {}
        padders = {}
        for key, data in alldata.get("residue_templates", {}).items():
            residue_templates[key] = ResidueTemplate.from_dict(data)
        for link_label, data in alldata.get("padders", {}).items():
            padders[link_label] = ResiduePadder.from_dict(data)
        return cls(residue_templates, padders, ambiguous)

    @classmethod
    def create_from_defaults(cls):
        return cls.from_json_file("residue_chem_templates")

    def add_json_file(self, filename):
        filename = self.lookup_filename(filename, data_path)
        with open(filename) as f:
            jsonstr = f.read()
        data = json.loads(jsonstr)
        self.add_dict(data)

    @staticmethod
    def _check_missing_padders(residue_templates, padders):
        link_labels_in_residues = set()
        for reskey, res_template in residue_templates.items():
            for _, link_label in res_template.link_labels.items():
                link_labels_in_residues.add(link_label)
        link_labels_in_padders = set([label for label in padders])
        missing = link_labels_in_residues.difference(link_labels_in_padders)
        if missing:
            raise RuntimeError(f"missing padders for {missing}")

    @staticmethod
    def _check_ambiguous_reskeys(residue_templates, ambiguous):
        missing = {}
        for input_resname, reskeys in ambiguous.items():
            for reskey in reskeys:
                if reskey not in residue_templates:
                    missing.setdefault(input_resname, set())
                    missing[input_resname].add(reskey)
        if len(missing):
            raise ValueError(f"missing residue templates for ambiguous: {missing}")
