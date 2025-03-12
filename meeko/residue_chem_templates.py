import pathlib
import json
import logging
import traceback
from importlib.resources import files
from os import linesep as eol
from sys import exc_info
from typing import Union
from typing import Optional

import rdkit.Chem
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdChemReactions
from rdkit.Chem import rdMolInterchange
from rdkit.Geometry import Point3D

from .molsetup import RDKitMoleculeSetup
from .molsetup import MoleculeSetupEncoder
from .utils.jsonutils import rdkit_mol_from_json
from .utils.rdkitutils import mini_periodic_table
from .utils.rdkitutils import react_and_map
from .utils.rdkitutils import AtomField
from .utils.rdkitutils import build_one_rdkit_mol_per_altloc
from .utils.rdkitutils import _aux_altloc_mol_build
from .utils.rdkitutils import covalent_radius
from .utils.pdbutils import PDBAtomInfo
from .chemtempgen import export_chem_templates_to_json
from .chemtempgen import build_noncovalent_CC
from .chemtempgen import build_linked_CCs

import numpy as np

data_path = files("meeko") / "data"
periodic_table = Chem.GetPeriodicTable()

try:
    import prody
except ImportError as _prody_import_error:
    ALLOWED_PRODY_TYPES = None
    AtomGroup = None
    Selection = None
    def prody_to_rdkit(*args):
        raise ImportError(_prody_import_error)
else:
    from .utils.prodyutils import prody_to_rdkit, ALLOWED_PRODY_TYPES
    from prody.atomic.atomgroup import AtomGroup
    from prody.atomic.selection import Selection


logger = logging.getLogger(__name__)
rdkit_logger = logging.getLogger("rdkit")

residues_rotamers = {
    "SER": [("C", "CA", "CB", "OG")],
    "THR": [("C", "CA", "CB", "CG2")],
    "CYS": [("C", "CA", "CB", "SG")],
    "VAL": [("C", "CA", "CB", "CG1")],
    "HIS": [("C", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD2")],
    "ASN": [("C", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND2")],
    "ASP": [("C", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "ILE": [("C", "CA", "CB", "CG2"), ("CA", "CB", "CG2", "CD1")],
    "LEU": [("C", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "PHE": [("C", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD2")],
    "TYR": [("C", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD2")],
    "TRP": [("C", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD2")],
    "GLU": [
        ("C", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "OE1"),
    ],
    "GLN": [
        ("C", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "OE1"),
    ],
    "MET": [
        ("C", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "SD"),
        ("CB", "CG", "SD", "CE"),
    ],
    "ARG": [
        ("C", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "NE"),
        ("CG", "CD", "NE", "CZ"),
    ],
    "LYS": [
        ("C", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "CE"),
        ("CG", "CD", "CE", "NZ"),
    ],
}


class ResidueChemTemplates:
    """Holds template data required to initialize Polymer

    Attributes
    ----------
    residue_templates: dict (string -> ResidueTemplate)
        keys are the ID of an instance of ResidueTemplate
    padders: dict
        instances of ResiduePadder keyed by a link_label (a string)
        link_labels establish the relationship between ResidueTemplates
        and ResiduePadders, determining which padder is to be used to
        pad each atom of an instance of Monomer that needs padding.
    ambiguous: dict
        mapping between input residue names (e.g. the three-letter residue
        name from PDB files) and IDs (strings) of ResidueTemplates
    """

    def __init__(self, residue_templates, padders, ambiguous):
        self._check_missing_padders(residue_templates, padders)
        self._check_ambiguous_reskeys(residue_templates, ambiguous)
        self.residue_templates = residue_templates
        self.padders = padders
        self.ambiguous = ambiguous

    @classmethod
    def from_dict(cls, alldata):
        """
        constructs ResidueTemplates and ResiduePadders from a dictionary
        with raw data such as that in data/residue_chem_templates.json
        This is pretty much a JSON deserializer that takes a dictionary
        as input to allow users to modify the input dict in Python
        """

        ambiguous = {k: v.copy() for k, v in alldata["ambiguous"].items()}
        residue_templates = {}
        padders = {}
        for key, data in alldata["residue_templates"].items():
            res_template = cls.residue_template_from_dict(data)
            residue_templates[key] = res_template
        for link_label, data in alldata["padders"].items():
            padders[link_label] = cls.padder_from_dict(data)
        return cls(residue_templates, padders, ambiguous)

    @staticmethod
    def residue_template_from_dict(data):
        if "link_labels" in data:
            link_labels = {int(k): v for k, v in data["link_labels"].items()}
        else:
            link_labels = None
        atom_names = data.get("atom_name", None)
        return ResidueTemplate(data["smiles"], link_labels, atom_names)

    @staticmethod
    def padder_from_dict(data):
        rxn_smarts = data["rxn_smarts"]
        adjacent_res_smarts = data.get("adjacent_res_smarts", None)
        auto_blunt = data.get("auto_blunt", False)
        padder = ResiduePadder(rxn_smarts, adjacent_res_smarts, auto_blunt)
        return padder

    def add_dict(self, data, overwrite=False):
        bad_keys = set(data) - {"ambiguous", "residue_templates", "padders"}
        if bad_keys:
            raise ValueError("unexpected keys: {bad_keys}")
        new_ambiguous = data.get("ambiguous", {}) 
        if overwrite:
            self.ambiguous.update(new_ambiguous)
        else:
            new_ambiguous = {k: v.copy() for k, v in new_ambiguous.items()}
            new_ambiguous.update(self.ambiguous)
            self.ambiguous = new_ambiguous
        for key, value in data.get("residue_templates", {}).items():
            if overwrite or key not in self.residue_templates:
                res_template = self.residue_template_from_dict(value)
                self.residue_templates[key] = res_template
        for link_label, value in data.get("padders", {}).items():
            if overwrite or key not in self.padders:
                padder = self.padder_from_dict(data)
                self.padders[link_label] = padder
        return

    @staticmethod
    def lookup_filename(filename, data_path):
        p = pathlib.Path(filename)
        if not p.exists():
            if (data_path / p).exists():
                filename = str(data_path / p)
            elif (data_path / (p.name + ".json")).exists():
                filename = str(data_path / (p.name + ".json"))
            else:
                raise ValueError(f"can't find {filename} in current dir or {data_path}")
        return filename

    @classmethod
    def from_json_file(cls, filename):
        filename = cls.lookup_filename(filename, data_path)
        with open(filename) as f:
            jsonstr = f.read()
        data = json.loads(jsonstr)
        return cls.from_dict(data)

    @classmethod
    def create_from_defaults(cls):
        return cls.from_json_file("residue_chem_templates")

    def add_json_file(self, filename):
        filename = self.lookup_filename(filename, data_path)
        with open(filename) as f:
            jsonstr = f.read()
        data = json.loads(jsonstr)
        self.add_dict(data)
        return

    @staticmethod
    def _check_missing_padders(residue_templates, padders):

        # can't guarantee full coverage because the topology that is passed
        # to the Polymer may contain bonds between residues that are not
        # anticipated to be bonded, for example, protein N-term bonded to
        # nucleic acid 5 prime.

        # collect labels from residues
        link_labels_in_residues = set()
        for reskey, res_template in residue_templates.items():
            for _, link_label in res_template.link_labels.items():
                link_labels_in_residues.add(link_label)

        # and check we have padders for all of them
        link_labels_in_padders = set([label for label in padders])
        # for link_label in padders:
        #    for (link_labels) in padder.link_labels:
        #        print(link_key, link_labels)
        #        for (label, _) in link_labels: # link_labels is a list of pairs
        #            link_labels_in_padders.add(label)

        missing = link_labels_in_residues.difference(link_labels_in_padders)
        if missing:
            raise RuntimeError(f"missing padders for {missing}")

        return

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
        return
