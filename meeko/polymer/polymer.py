"""The ``Polymer`` class: represents proteins / nucleic acids as a graph of
``Monomer`` instances joined by inter-residue bonds.

The orchestration here is large; helper utilities, the residue/padder/template
data classes, and the rotamer machinery have been split off into sibling
modules. This module holds only the class.
"""

import json
import logging
import warnings
from importlib.resources import files
from typing import Any, Optional, Union

from rdkit import Chem
from rdkit.Chem import rdMolInterchange
from rdkit.Geometry import Point3D

from ..chemtempgen import (
    build_linked_CCs,
    build_noncovalent_CC,
    export_chem_templates_to_json,
)
from ..molsetup import RDKitMoleculeSetup
from ..preparation import MoleculePreparation
from ..utils.covalent_radius_table import covalent_radius
from ..utils.jsonutils import (
    BaseJSONParsable,
    convert_to_int_keyed_dict,
    rdkit_mol_from_json,
    serialize_optional,
)
from ..utils.pdbutils import PDBAtomInfo
from ..utils.rdkitutils import AtomField, _aux_altloc_mol_build, mini_periodic_table

from .errors import PolymerCreationError
from .monomer import Monomer
from .padder import ResiduePadder
from .templates import ResidueChemTemplates, ResidueTemplate
from .utils import (
    _delete_residues,
    find_graph_paths,
    find_inter_mols_bonds,
    find_inter_mols_bonds_kdtree,
    find_inter_mols_bonds_kdtree_fast,
    find_inter_mols_bonds_old,
    get_updated_positions,
    handle_parsing_situations,
    mapping_by_mcs,
    rectify_charges,
    update_H_positions,
)

import numpy as np

eol = "\n"
data_path = files("meeko") / "data"
periodic_table = Chem.GetPeriodicTable()

logger = logging.getLogger(__name__)
rdkit_logger = logging.getLogger("rdkit")

try:
    import prody
except ImportError as _prody_import_error:
    ALLOWED_PRODY_TYPES = None
    AtomGroup = None
    Selection = None

    def prody_to_rdkit(*args):
        raise ImportError(_prody_import_error)
else:
    from ..utils.prodyutils import prody_to_rdkit, ALLOWED_PRODY_TYPES
    from prody.atomic.atomgroup import AtomGroup
    from prody.atomic.selection import Selection


class Polymer(BaseJSONParsable):
    """Represents polymer with its subunits as individual RDKit molecules.

    Used for proteins and nucleic acids. The key class is Monomer,
    which contains, a padded RDKit molecule containing part of the adjacent
    residues to enable chemically meaningful parameterizaion.
    Instances of ResidueTemplate make sure that the input, which may originate
    from a PDB string, matches the RDKit molecule of the template, even if
    hydrogens are missing.

    Attributes
    ----------
    monomers: dict (string -> Monomer) #TODO: figure out exact SciPy standard for dictionary key/value notation
    termini: dict (string (representing residue id) -> string (representing what we want the capping to look like))
    mutate_res_dict: dict (string (representing starting residue id) -> string (representing the desired mutated id))
    res_templates: dict (string -> dict (rdkit_mol and atom_data))
    ambiguous:
    disulfide_bridges:
    suggested_mutations:
    """

    def __init__(
        self,
        raw_input_mols: dict[str, tuple[Chem.Mol, str]],
        bonds: dict[tuple[str, str], tuple[int, int]],
        residue_chem_templates: ResidueChemTemplates,
        mk_prep=None,
        set_template: dict[str, str] = None,
        blunt_ends: list[tuple[str, int]] = None,
        get_atomprop_from_raw: dict = None,
        ignore_https_cert = False,
        forgive_extra_bonds: bool = False
    ):
        """
        Parameters
        ----------
        raw_input_mols: dict (string -> (Chem.Mol, string))
            A dictionary of raw input mols where keys are residue IDs in the format <chain>:<resnum> such as "A:42" and
            values are tuples of an RDKit Mols and input resname.
            RDKit Mols will be matched to instances of ResidueTemplate, and may contain none, all, or some of the
            Hydrogens.
        bonds: dict ((string, string) -> (int, int))
        residue_chem_templates: ResidueChemTemplates
            An instance of the ResidueChemTemplates class.
        mk_prep: MoleculePreparation
            An instance of the MoleculePreparation class to parameterize the padded molecules.
        set_template: dict (string -> string)
            A dict mapping residue IDs in the format <chain>:<resnum> such as "A:42" to ResidueTemplate instances.
        blunt_ends: list (tuple (string, int))
            A list of tuples where each tuple is residue IDs and 0-based atom index, e.g.; ("A:42", 0)
        ignore_https_cert: Ignore https cert of PDB database (rcsb.org) when True
        forgive_extra_bonds: bool
            allows processing clashed structures because templates match even with excess bonds to other residues
            at the expense of causing unpredictable problems and potentially matching incorrect templates

        Returns
        -------
        None

        Raises
        ------
        ValueError:
        """
        

        # TODO simplify SMARTS for adjacent res in padders

        if type(raw_input_mols) != dict:
            msg = f"expected raw_input_mols to be dict, got {type(raw_input_mols)}"
            if type(raw_input_mols) == str:
                msg += eol
                msg += (
                    "consider Polymer.from_pdb_string(pdbstr)" + eol
                )
            raise ValueError(msg)
        self.residue_chem_templates = residue_chem_templates
        residue_templates = residue_chem_templates.residue_templates
        padders = residue_chem_templates.padders
        ambiguous = residue_chem_templates.ambiguous

        # store a copy of bonds.
        self.bonds = bonds.copy()

        if set_template is None:
            set_template = {}
        else:  # make sure all resiude_id in set_template exist
            missing = set(
                [
                    residue_id
                    for residue_id in set_template
                    if residue_id not in raw_input_mols
                ]
            )
            if len(missing):
                raise ValueError(
                    f"Residue IDs in set_template not found: {missing} {raw_input_mols.keys()}"
                )

        # check if input assigned residue name in residue_templates
        err = ""
        supported_resnames = residue_templates.keys() | ambiguous.keys()
        unknown_res_from_input = {res_id: raw_input_mols[res_id][1] 
                                  for res_id in raw_input_mols 
                                  if res_id not in set_template and raw_input_mols[res_id][1] not in supported_resnames
                                  }
        
        if unknown_res_from_input:
            unknown_valid_res_from_input = {k: v for k, v in unknown_res_from_input.items() if v != "UNL"}
            if unknown_valid_res_from_input: 
                err += f"Input residues {unknown_valid_res_from_input} not in residue_templates" + eol
            UNL_from_input = {k: v for k, v in unknown_res_from_input.items() if v == "UNL"}
            if UNL_from_input: 
                err += f"Input residues {UNL_from_input} do not have a concrete definition" + eol
        
        unknown_res_from_assign = {}
        if set_template:
            unknown_res_from_assign = {res_id: resn for res_id, resn in set_template.items() if resn not in supported_resnames}
            unknown_valid_res_from_assign = {k: v for k, v in unknown_res_from_assign.items() if v != "UNL"}
            if unknown_valid_res_from_assign: 
                err += f"Input residues {unknown_valid_res_from_assign} not in residue_templates" + eol
            UNL_from_assign = {k: v for k, v in unknown_res_from_assign.items() if v == "UNL"}
            if UNL_from_assign: 
                err += f"Input residues {UNL_from_assign} do not have a concrete definition" + eol
        
        if err:
            if "UNL" in err: 
                err += "Resdiues that are named UNL can't be parameterized. " + eol
                rec = "1. (to parameterize the residues) Use --set_template to specify valid residue names, " + eol
                rec += "2. (to skip the residues) Use --delete_residues to ignore them. Residues will be deleted from the prepared receptor. "
                raise PolymerCreationError(err, rec)

            warnings.warn(err, RuntimeWarning)
            warnings.warn("Trying to resolve unknown residues by building chemical templates... ", RuntimeWarning)

            all_unknown_res = unknown_res_from_input.copy()
            all_unknown_res.update(unknown_res_from_assign)

            bonded_unknown_res = {res_id: all_unknown_res[res_id] for res_id in all_unknown_res 
                                  if any(res_id in respair for respair in bonds)}

            unbound_unknown_res = all_unknown_res.copy()
            for key in bonded_unknown_res:
                unbound_unknown_res.pop(key, None) 

            if unbound_unknown_res: 
                for resname in set(unbound_unknown_res.values()): 
                    try: 
                        cc = build_noncovalent_CC(resname, ignore_https_cert=ignore_https_cert)
                        fetch_template_dict = json.loads(export_chem_templates_to_json([cc]))['residue_templates'][cc.resname]
                        residue_templates.update({resname: ResidueTemplate(
                                                    smiles = fetch_template_dict['smiles'],
                                                    atom_names = fetch_template_dict['atom_name'],
                                                    link_labels = fetch_template_dict['link_labels'])})
                        ambiguous[resname] = [cc.resname]
                    except Exception as e: 
                        logger.warning(f"Failed building template from CCD for {resname=}")
                        raise PolymerCreationError(str(e))

            if bonded_unknown_res: 
                failed_build = set()
                try: 
                    for resname in set(bonded_unknown_res.values()): 
                        cc_list = build_linked_CCs(resname, ignore_https_cert=ignore_https_cert)
                        if not cc_list: 
                            failed_build.add(resname)
                        else:
                            for cc in cc_list:
                                fetch_template_dict = json.loads(export_chem_templates_to_json([cc]))['residue_templates'][cc.resname]
                                residue_templates.update({cc.resname: ResidueTemplate(
                                                            smiles = fetch_template_dict['smiles'],
                                                            atom_names = fetch_template_dict['atom_name'],
                                                            link_labels = convert_to_int_keyed_dict(fetch_template_dict['link_labels']))})
                                if resname in ambiguous: 
                                    ambiguous[resname].append(cc.resname)
                                else:
                                    ambiguous[resname] = [cc.resname]
                except Exception as e: 
                    raise PolymerCreationError(str(e))
                            
                if failed_build: 
                    raise PolymerCreationError(f"Template generation failed for unknown residues: {failed_build}, which appear to be linking fragments. " + eol
                                            + "Generation of chemical templates with modified backbones, which involves guessing of linker positions and types, are not currently supported. ", 
                                            "1. (to parameterize the residues) Use --add_templates to pass the additional templates with valid linker_labels, " + eol
                                            + "2. (to skip the residues) Use --delete_residues to ignore them. Residues will be deleted from the prepared receptor. ")

        self.monomers, self.log = self._get_monomers(
            raw_input_mols,
            ambiguous,
            residue_chem_templates,
            set_template,
            bonds,
            blunt_ends,
            forgive_extra_bonds,
        )

        _bonds = {}
        for key, bond_list in bonds.items():
            monomer1 = self.monomers[key[0]]
            monomer2 = self.monomers[key[1]]
            if monomer1.rdkit_mol is None or monomer2.rdkit_mol is None:
                continue
            invmap1 = {j: i for i, j in monomer1.mapidx_to_raw.items()}
            invmap2 = {j: i for i, j in monomer2.mapidx_to_raw.items()}
            _bonds[key] = [(invmap1[b[0]], invmap2[b[1]]) for b in bond_list]
        bonds = _bonds

        # padding may seem overkill but we had to run a reaction anyway for h_coord_from_dipep
        padded_mols = self._build_padded_mols(self.monomers, bonds, padders)
        for residue_id, (padded_mol, mapidx_from_pad) in padded_mols.items():
            monomer = self.monomers[residue_id]
            monomer.padded_mol = padded_mol
            monomer.molsetup_mapidx = mapidx_from_pad

        if mk_prep is not None:
            self.parameterize(mk_prep, get_atomprop_from_raw = get_atomprop_from_raw)

        return
    
    # region JSON-interchange functions
    @classmethod
    def json_encoder(cls, obj: "Polymer") -> Optional[dict[str, Any]]:
        
        output_dict = {
            "residue_chem_templates": ResidueChemTemplates.json_encoder(
                obj.residue_chem_templates
            ),
            "monomers": {
                k: Monomer.json_encoder(v)
                for k, v in obj.monomers.items()
            },
            "log": obj.log,
        }
        return output_dict
    
    # Keys to check for deserialized JSON 
    expected_json_keys = {
        "residue_chem_templates",
        "monomers",
        "log",
    }

    @classmethod
    def _combine_many_mols_tree(cls, mols):
        r"""tree-like compbination of mols iterable (nlog(n) cost).

            mols: 
            (a,b)   (c,d)
              \       /
               (ab, cd)
                   |
                 abcd   

        """
        mols = list(mols)
        if not mols:
            return Chem.Mol()
        while len(mols) > 1:
            nxt = []
            it = iter(mols)
            for a in it:
                b = next(it, None)
                nxt.append(Chem.CombineMols(a, b) if b is not None else a)
            mols = nxt
        return mols[0]

    def to_rdkit_mol(self, residues_to_add: Optional[set[str]] = None, 
               bonds_to_use: Optional[dict[tuple[str], list[tuple[int]]]] = None):
        """returns a single rdkit molecule that results from adding bonds
            between every monomer residue. It may contain multiple fragments
            if there are multiple chains or gaps. 

            Optionally, specify a set of residue IDs for stitching.
            Defaults to stitching all monomers. 

            Optionally, specify a dict for bonds to use, 
            Defaults to stitching using all available bonds in polymer. 
            key format: (res_id_1, res_id_2)
            value format: [(atom_idx_1, atom_idx_2), ]
            same format as output from function find_inter_mols_bonds, 
            but the indices need to based on rdkit_mol. 
        """
        
        # stitching all valid monomers by default
        valid_monomers = set(self.get_valid_monomers().keys())
        residues_to_add = residues_to_add or valid_monomers
        residues_to_add = set(residues_to_add)

        # verify if requested monomers are valid (have rdkit_mol)
        invalid_monomers = residues_to_add - valid_monomers
        if invalid_monomers: 
            raise ValueError(f"Residue IDs not in valid monomers: {invalid_monomers}")

        if bonds_to_use is None:
            bonds_to_use = {}
            resid_to_rawmols = {res_id: (self.monomers[res_id].raw_rdkit_mol, self.monomers[res_id].input_resname) for res_id in residues_to_add}

            # check if bonds is None or empty. 
            if self.bonds is None or not self.bonds: 
                bonds_indexed_in_raw = find_inter_mols_bonds(resid_to_rawmols)
            else:
                bonds_indexed_in_raw = self.bonds

            invmaps = {
                res_id: {j: i for i, j in self.monomers[res_id].mapidx_to_raw.items()}
                for res_id in residues_to_add
            }
            for (res1, res2), bond_list in bonds_indexed_in_raw.items():
                invmap1, invmap2 = invmaps[res1], invmaps[res2]
                bonds_to_use[(res1, res2)] = [(invmap1[b[0]], invmap2[b[1]]) for b in bond_list]
        
        # initialize mol and residue/bond tracking
        mol = Chem.Mol()
        residues_added = {}
        bonds_spent = set()
        
        # add residues and get offset in order
        offset = 0
        mols = []
        for r_id in residues_to_add:
            res = self.monomers[r_id]
            m = res.rdkit_mol
            residues_added[r_id] = offset
            offset += m.GetNumAtoms()
            # mol = Chem.CombineMols(mol, res.rdkit_mol)
            mols.append(m)

        mol = Polymer._combine_many_mols_tree(mols)

        # add bonds
        edit_mol = Chem.EditableMol(mol)
        for bond_key, bond_list in bonds_to_use.items():
            if bond_key in bonds_spent:
                continue
            r1, r2 = bond_key
            if r1 in residues_added and r2 in residues_added:
                bonds_spent.add(bond_key)
                for bond in bond_list: 
                    i, j = bond
                    edit_mol.AddBond(
                        i + residues_added[r1],
                        j + residues_added[r2],
                        order=Chem.rdchem.BondType.SINGLE
                    )
        mol = edit_mol.GetMol()

        
        # review added bonds and residues
        if len(bonds_spent) != len(bonds_to_use):
            raise RuntimeError("nr of bonds added differs from bonds to use")
        if len(residues_added) != len(residues_to_add):
            raise RuntimeError("nr of residues added differs from residues to add")
        
        return mol
    
    # for backwards compatibility. 
    def stitch(self, residues_to_add = None, 
               bonds_to_use = None):
        """ Alias of polymer.to_rdkit_mol."""
        return self.to_rdkit_mol(residues_to_add, bonds_to_use)

    @classmethod
    def _decode_object(cls, obj: dict[str, Any]): 

        # Deserializes ResidueChemTemplates from the dict to use as an input, then constructs a Polymer object
        # and sets its values using deserialized JSON values.
        residue_chem_templates = ResidueChemTemplates.from_dict(
            obj["residue_chem_templates"]
        )

        polymer = cls({}, {}, residue_chem_templates)

        polymer.monomers = {}
        templates = residue_chem_templates.residue_templates
        for k, v in obj["monomers"].items():
            monomer = Monomer.from_dict(v)
            if monomer.template is None:  # JSON-bound only from v0.7.0
                # try to recover template from stored templates
                residue_key = monomer.residue_template_key
                monomer.template = templates.get(residue_key, None)
            polymer.monomers[k] = monomer
        polymer.log = obj["log"]

        return polymer
    # endregion

    @classmethod
    def from_pdb_file(cls, filename, *args, **kwargs):
        with open(filename) as f:
            pdb_string = f.read()
        return cls.from_pdb_string(pdb_string, *args, **kwargs)
    
    @classmethod
    def from_pdb_string(
        cls,
        pdb_string,
        chem_templates=None,
        mk_prep=None,
        set_template=None,
        residues_to_delete=None,
        ignore_https_cert=False,
        allow_bad_res=False,
        bonds_to_delete=None,
        blunt_ends=None,
        wanted_altloc=None,
        default_altloc=None,
        forgive_extra_bonds=False,
    ):
        """

        Parameters
        ----------
        pdb_string
        chem_templates
        mk_prep
        set_template
        residues_to_delete
        ignore_https_cert
        allow_bad_res
        bonds_to_delete
        blunt_ends
        wanted_altloc
        default_altloc
        forgive_extra_bonds

        Returns
        -------

        """

        #Set default mk_prep and chem_templates if not available. 
        if chem_templates is None:
            chem_templates = ResidueChemTemplates.create_from_defaults()

        if mk_prep is None:
            mk_prep = MoleculePreparation()

        tmp_raw_input_mols = cls._pdb_to_residue_mols(
            pdb_string,
            wanted_altloc,
            default_altloc,
        )

        # from here on it duplicates self.from_prody(), but extracting
        # this out into a function felt like it sacrificed readibility
        # so I decided to keep the duplication.
        _delete_residues(residues_to_delete, tmp_raw_input_mols)
        raw_input_mols = {}
        res_needed_altloc = []
        res_missed_altloc = []
        unparsed_res = []
        for res_id, stuff in tmp_raw_input_mols.items():
            mol, resname, missed_altloc, needed_altloc = stuff
            if mol is None and missed_altloc:
                res_missed_altloc.append(res_id)
            elif mol is None and needed_altloc:
                res_needed_altloc.append(res_id)
            elif mol is None:
                unparsed_res.append(res_id)
            else:
                raw_input_mols[res_id] = (mol, resname)

        # bonds_old = find_inter_mols_bonds_old(raw_input_mols)

        bonds = find_inter_mols_bonds(raw_input_mols)

        if bonds_to_delete is not None:
            for res1, res2 in bonds_to_delete:
                popped = ()
                if (res1, res2) in bonds:
                    popped = bonds.pop((res1, res2))
                elif (res2, res1) in bonds:
                    popped = bonds.pop((res2, res1))
                if len(popped) >= 2:
                    msg = (
                        "can't delete bonds for residue pairs that have more"
                        " than one bond between them"
                    )
                    raise NotImplementedError(msg)

        polymer = cls(
            raw_input_mols,
            bonds,
            chem_templates,
            mk_prep,
            set_template,
            blunt_ends,
            None,
            ignore_https_cert,
            forgive_extra_bonds=forgive_extra_bonds
        )

        unmatched_res = polymer.get_ignored_monomers()
        handle_parsing_situations(
            unmatched_res,
            unparsed_res,
            allow_bad_res,
            res_missed_altloc,
            res_needed_altloc,
        )

        return polymer

    # region adapted from from_pdb_string
    @classmethod
    def from_pqr_string(
        cls,
        pqr_string,
        chem_templates=None,
        mk_prep=None,
        set_template=None,
        residues_to_delete=None,
        ignore_https_cert=False,
        allow_bad_res=False,
        bonds_to_delete=None,
        blunt_ends=None,
        forgive_extra_bonds=False,
    ):
        """

        Parameters
        ----------
        pdb_string
        chem_templates
        mk_prep
        set_template
        residues_to_delete
        ignore_https_cert
        allow_bad_res
        bonds_to_delete
        blunt_ends
        forgive_extra_bonds

        Returns
        -------

        """

        #Set default mk_prep and chem_templates if not available. 
        if chem_templates is None:
            chem_templates = ResidueChemTemplates.create_from_defaults()
        if mk_prep is None:
            mk_prep = MoleculePreparation()

        tmp_raw_input_mols = cls._pqr_to_residue_mols(
            pqr_string,
        )

        # from here on it duplicates self.from_prody(), but extracting
        # this out into a function felt like it sacrificed readibility
        # so I decided to keep the duplication.
        _delete_residues(residues_to_delete, tmp_raw_input_mols)
        raw_input_mols = {}
        res_needed_altloc = []
        res_missed_altloc = []
        unparsed_res = []
        for res_id, stuff in tmp_raw_input_mols.items():
            mol, resname, missed_altloc, needed_altloc = stuff
            if mol is None and missed_altloc:
                res_missed_altloc.append(res_id)
            elif mol is None and needed_altloc:
                res_needed_altloc.append(res_id)
            elif mol is None:
                unparsed_res.append(res_id)
            else:
                raw_input_mols[res_id] = (mol, resname)
        bonds = find_inter_mols_bonds(raw_input_mols)
        if bonds_to_delete is not None:
            for res1, res2 in bonds_to_delete:
                popped = ()
                if (res1, res2) in bonds:
                    popped = bonds.pop((res1, res2))
                elif (res2, res1) in bonds:
                    popped = bonds.pop((res2, res1))
                if len(popped) >= 2:
                    msg = (
                        "can't delete bonds for residue pairs that have more"
                        " than one bond between them"
                    )
                    raise NotImplementedError(msg)
                
        polymer = cls(
            raw_input_mols,
            bonds,
            chem_templates,
            mk_prep,
            set_template,
            blunt_ends,
            get_atomprop_from_raw = {"PQRCharge": 0.},
            ignore_https_cert=ignore_https_cert,
            forgive_extra_bonds=forgive_extra_bonds,
        )

        if polymer.log["matched_with_H_anomaly"]:
            msg = ""
            for res_id, (template_name, h_info) in polymer.log["matched_with_H_anomaly"].items():
                h_miss = h_info.get('H_miss', 0)
                h_excess = h_info.get('H_excess', 0)
                msg += f"Residue {res_id} matched with template '{template_name}' has H discrepancy: {h_miss} missing, {h_excess} excess. \n"
            raise PolymerCreationError(msg + "These discrepancies may compromise the validity of the charge assignment from PQR, making the charges inapplicable to the processed receptor. \n")

        unmatched_res = polymer.get_ignored_monomers()
        handle_parsing_situations(
            unmatched_res,
            unparsed_res,
            allow_bad_res,
            res_missed_altloc,
            res_needed_altloc,
        )

        return polymer
    # endregion

            
    @classmethod
    def from_prody(
        cls,
        prody_obj: Union[Selection, AtomGroup],
        chem_templates=None,
        mk_prep=None,
        set_template=None,
        residues_to_delete=None,
        ignore_https_cert=False,
        allow_bad_res=False,
        bonds_to_delete=None,
        blunt_ends=None,
        wanted_altloc: Optional[dict]=None,
        default_altloc: Optional[str]=None,
        forgive_extra_bonds: bool=False,
    ):
        """

        Parameters
        ----------
        prody_obj
        chem_templates
        mk_prep
        set_template
        residues_to_delete
        ignore_https_cert
        allow_bad_res
        bonds_to_delete
        blunt_ends
        wanted_altloc
        default_altloc
        forgive_extra_bonds

        Returns
        -------

        """

        #Set default mk_prep and chem_templates if not available. 
        if chem_templates is None:
            chem_templates = ResidueChemTemplates.create_from_defaults()
        if mk_prep is None:
            mk_prep = MoleculePreparation()

        tmp_raw_input_mols = cls._prody_to_residue_mols(
            prody_obj,
            wanted_altloc,
            default_altloc,
        )

        # from here on it duplicates self.from_pdb_string(), but extracting
        # this out into a function felt like it sacrificed readibility
        # so I decided to keep the duplication.
        _delete_residues(residues_to_delete, tmp_raw_input_mols)
        raw_input_mols = {}
        res_needed_altloc = []
        res_missed_altloc = []
        unparsed_res = []
        for res_id, stuff in tmp_raw_input_mols.items():
            mol, resname, missed_altloc, needed_altloc = stuff
            if mol is None and missed_altloc:
                res_missed_altloc.append(res_id)
            elif mol is None and needed_altloc:
                res_needed_altloc.append(res_id)
            elif mol is None:
                unparsed_res.append(res_id)
            else:
                raw_input_mols[res_id] = (mol, resname)

        bonds = find_inter_mols_bonds(raw_input_mols)
        if bonds_to_delete is not None:
            for res1, res2 in bonds_to_delete:
                popped = ()
                if (res1, res2) in bonds:
                    popped = bonds.pop((res1, res2))
                elif (res2, res1) in bonds:
                    popped = bonds.pop((res2, res1))
                if len(popped) >= 2:
                    msg = (
                        "can't delete bonds for residue pairs that have more"
                        " than one bond between them"
                    )
                    raise NotImplementedError(msg)
        polymer = cls(
            raw_input_mols,
            bonds,
            chem_templates,
            mk_prep,
            set_template,
            blunt_ends,
            None,
            ignore_https_cert,
            forgive_extra_bonds=forgive_extra_bonds
        )
        unmatched_res = polymer.get_ignored_monomers()
        handle_parsing_situations(
            unmatched_res,
            unparsed_res,
            allow_bad_res,
            res_missed_altloc,
            res_needed_altloc,
        )

        return polymer

    def parameterize(self, mk_prep, get_atomprop_from_raw = None):
        """

        Parameters
        ----------
        mk_prep

        Returns
        -------

        """

        for residue_id, monomer in self.get_valid_monomers().items():
            monomer.parameterize(mk_prep, residue_id, get_atomprop_from_raw = get_atomprop_from_raw)

    def flexibilize_sidechain(self, residue_id, mk_prep):
        if residue_id not in self.get_valid_monomers():
            raise ValueError(f"{residue_id=} not in valid monomers")
        return self.monomers[residue_id].flexibilize(mk_prep)

    def rigidify_sidechain(self, residue_id, mk_prep):
        if residue_id not in self.get_valid_monomers():
            raise ValueError(f"{residue_id=} not in valid monomers")
        return self.monomers[residue_id].rigidify(mk_prep, residue_id)

    def rigidify_all(self, mk_prep):
        for residue_id, monomer in self.get_valid_monomers().items():
            if monomer.is_movable:
                monomer.rigidify(mk_prep, residue_id)
        return


    @staticmethod
    def _build_rdkit_mol(raw_mol, template, mapping, nr_missing_H):
        """

        Parameters
        ----------
        raw_mol
        template
        mapping
        nr_missing_H

        Returns
        -------

        """
        rdkit_mol = Chem.Mol(template.mol)  # making a copy
        conf = Chem.Conformer(rdkit_mol.GetNumAtoms())
        input_conf = raw_mol.GetConformer()
        for i, j in mapping.items():
            conf.SetAtomPosition(i, input_conf.GetAtomPosition(j))

        rdkit_mol.AddConformer(conf, assignId=True)

        if nr_missing_H:  # add positions to Hs missing in raw_mol
            if rdkit_mol.GetNumAtoms() != len(mapping) + nr_missing_H:
                raise RuntimeError(
                    f"nr of atoms ({rdkit_mol.GetNumAtoms()}) != "
                    f"{len(mapping)=} + {nr_missing_H=}"
                )
            idxs = [i for i in range(rdkit_mol.GetNumAtoms()) if i not in mapping]
            update_H_positions(rdkit_mol, idxs)

        return rdkit_mol

    @staticmethod
    def _get_best_missing_Hs(results):
        """

        Parameters
        ----------
        results

        Returns
        -------

        """
        min_missing_H = 999999
        best_idxs = []
        fail_log = []
        for i, result in enumerate(results):
            fail_log.append([])
            if result["heavy"]["missing"] > 0:
                fail_log[-1].append("heavy missing")
            if result["heavy"]["excess"] > 0:
                fail_log[-1].append("heavy excess")
            if len(result["H"]["excess"]) > 0:
                fail_log[-1].append("H excess")
            if len(result["bonds"]["excess"]) > 0:
                fail_log[-1].append("bonds excess")
            if len(result["bonds"]["missing"]) > 0:
                fail_log[-1].append(f"bonds missing at {result['bonds']['missing']}")
            if len(fail_log[-1]):
                continue
            if result["H"]["missing"] < min_missing_H:
                best_idxs = []
                min_missing_H = result["H"]["missing"]
            if result["H"]["missing"] == min_missing_H:
                best_idxs.append(i)
        return best_idxs, fail_log

    @classmethod
    def _get_monomers(
        cls,
        raw_input_mols,
        ambiguous,
        residue_chem_templates,
        set_template,
        bonds,
        blunt_ends,
        forgive_extra_bonds=False,
    ):
        """

        Parameters
        ----------
        raw_input_mols
        ambiguous
        residue_chem_templates
        set_template
        bonds
        blunt_ends

        Returns
        -------

        """

        residue_templates = residue_chem_templates.residue_templates
        template_charges = residue_chem_templates.template_charges
        monomers = {}
        log = {
            "chosen_by_fewest_missing_H": {},
            "chosen_by_default": {},
            "matched_with_H_anomaly": {},
            "matched_with_excess_bond": [],
            "no_match": [],
            "no_mol": [],
            "msg": "",
        }
        for residue_key, (raw_mol, input_resname) in raw_input_mols.items():
            if raw_mol is None:
                monomers[residue_key] = Monomer(
                    None, None, None, input_resname, None
                )
                log["no_mol"].append(residue_key)
                logger.warning(f"molecule for {residue_key=} is None")
                continue

            raw_mol_has_H = sum([a.GetAtomicNum() == 1 for a in raw_mol.GetAtoms()]) > 0
            excess_H_ok = False
            if set_template is not None and residue_key in set_template:
                excess_H_ok = True  # e.g. allow set LYN (NH2) from LYS (NH3+)
                template_key = set_template[residue_key]  # e.g. HID, NALA
                if template_key not in residue_templates: 
                    if template_key in ambiguous: 
                        raise RuntimeError(f"Can't assign an ambiguous tamplate_key ({template_key}) to residue ({residue_key}). ")
                    raise RuntimeError(f"Assigned tamplate_key ({template_key}) for residue ({residue_key}) is not in residue_templates. ")
                template = residue_templates[template_key]
                candidate_template_keys = [set_template[residue_key]]
                candidate_templates = [template]

            elif input_resname not in ambiguous:
                template_key = input_resname
                template = residue_templates[template_key]
                candidate_template_keys = [template_key]
                candidate_templates = [template]
            elif len(ambiguous[input_resname]) == 1:
                template_key = ambiguous[input_resname][0]
                template = residue_templates[template_key]
                candidate_template_keys = [template_key]
                candidate_templates = [template]
            else:
                candidate_template_keys = []
                candidate_templates = []
                for key in ambiguous[input_resname]:
                    template = residue_templates[key]
                    candidate_templates.append(template)
                    candidate_template_keys.append(key)

            # gather raw_mol atoms that have bonds or blunt ends
            if blunt_ends is None:
                blunt_ends = []
            raw_atoms_with_bonds = []
            for (r1, r2), bond_list in bonds.items():
                for i, j in bond_list:
                    if r1 == residue_key:
                        raw_atoms_with_bonds.append(i)
                    if r2 == residue_key:
                        raw_atoms_with_bonds.append(j)

            all_stats = {
                "heavy_missing": [],
                "heavy_excess": [],
                "H_excess": [],
                "H_missing": [],
                "bonded_atoms_missing": [],
                "bonded_atoms_excess": [],
            }
            mappings = []
            for index, template in enumerate(candidate_templates):

                # match intra-residue graph
                match_stats, mapping = template.match(raw_mol)
                mappings.append(mapping)

                # match inter-residue bonds
                atoms_with_bonds = set()
                from_raw = {value: key for (key, value) in mapping.items()}
                for raw_index in raw_atoms_with_bonds:
                    if raw_index in from_raw:  # bonds can occur on atoms the template does not have
                        atom_index = from_raw[raw_index]
                        atoms_with_bonds.add(atom_index)
                # we treat blunt ends like bonds
                for res_id, atom_idx in blunt_ends:
                    if res_id == residue_key:
                        atoms_with_bonds.add(from_raw[atom_idx])
                expected = set(template.link_labels)
                bonded_atoms_found = atoms_with_bonds.intersection(expected)
                bonded_atoms_missing = expected.difference(atoms_with_bonds)
                bonded_atoms_excess = atoms_with_bonds.difference(expected)

                all_stats["heavy_missing"].append(match_stats["heavy"]["missing"])
                all_stats["heavy_excess"].append(match_stats["heavy"]["excess"])
                all_stats["H_excess"].append(match_stats["H"]["excess"])
                all_stats["H_missing"].append(match_stats["H"]["missing"])
                all_stats["bonded_atoms_missing"].append(bonded_atoms_missing)
                all_stats["bonded_atoms_excess"].append(bonded_atoms_excess)

            passed = []

            embedded_indices = [index for index, template in enumerate(candidate_templates) if len(template.link_labels) >= 2]
            # 1st round
            for i in embedded_indices:
                if (
                    all_stats["heavy_missing"][i]
                    or all_stats["heavy_excess"][i]
                    or all_stats["H_excess"][i]
                    or all_stats["bonded_atoms_missing"][i]
                    or len(all_stats["bonded_atoms_excess"][i])
                ):
                    continue
                passed.append(i)

            # 2nd round
            if len(passed) == 0: 
                for i in embedded_indices:
                    auto_blunt = set()
                    for j, padder_label in candidate_templates[i].link_labels.items():
                        if residue_chem_templates.padders[padder_label].auto_blunt:
                            auto_blunt.add(j)
                    if (
                        all_stats["heavy_missing"][i]
                        or all_stats["heavy_excess"][i]
                        or (not set(all_stats["H_excess"][i]) <= set(candidate_templates[i].link_labels) and not excess_H_ok)
                        or not all_stats["bonded_atoms_missing"][i] <= auto_blunt
                        or (len(all_stats["bonded_atoms_excess"][i]) and not forgive_extra_bonds)
                    ):
                        continue
                    passed.append(i)

            # 3rd round
            if len(passed) == 0 or any(all_stats["H_excess"][i] for i in passed): 
                for i in range(len(candidate_templates)):
                    if (
                        all_stats["heavy_missing"][i]
                        or all_stats["heavy_excess"][i]
                        or (all_stats["H_excess"][i] and not excess_H_ok)
                        or len(all_stats["bonded_atoms_missing"][i])
                        or (len(all_stats["bonded_atoms_excess"][i]) and not forgive_extra_bonds)
                    ):
                        continue
                    if i not in passed:
                        passed.append(i)

            if len(passed) == 0:
                template_key = None
                template = None
                mapping = None
                m = f"No template matched for {residue_key=}" + eol
                m += f"tried {len(candidate_templates)} templates for {residue_key=}"
                m += f"{excess_H_ok=}"
                m += eol
                for i in range(len(all_stats["H_excess"])):
                    heavy_miss = all_stats["heavy_missing"][i]
                    heavy_excess = all_stats["heavy_excess"][i]
                    H_excess = all_stats["H_excess"][i]
                    bond_miss = all_stats["bonded_atoms_missing"][i]
                    bond_excess = all_stats["bonded_atoms_excess"][i]
                    tkey = candidate_template_keys[i]
                    m += (
                        f"{tkey:10} {heavy_miss=} {heavy_excess=} {H_excess=} {bond_miss=} {bond_excess=}"
                        + eol
                    )
                logger.warning(m)
            elif len(passed) == 1 or not raw_mol_has_H:
                index = passed[0]
                template_key = candidate_template_keys[index]
                template = candidate_templates[index]
                mapping = mappings[index]
            else:
                min_missing_H = 999999
                for index in passed:
                    H_missed = all_stats["H_missing"][index]
                    if H_missed < min_missing_H:
                        best_idxs = []
                        min_missing_H = H_missed
                    if H_missed == min_missing_H:
                        best_idxs.append(index)

                if len(best_idxs) > 1:
                    number_excess_H = [len(all_stats["H_excess"][index]) for index in best_idxs]
                    min_excess_H = min(number_excess_H)
                    best_idxs = [index for index in best_idxs if len(all_stats["H_excess"][index]) == min_excess_H]
                    
                    if len(best_idxs) > 1: 
                        tied = " ".join(candidate_template_keys[i] for i in best_idxs)
                        m = f"for {residue_key=}, {len(passed)} have passed: "
                        tkeys = [candidate_template_keys[i] for i in passed]
                        m += f"{tkeys} and tied for fewest missing and excess H: {tied} "

                        raise RuntimeError(m)
                
                index = best_idxs[0]
                template_key = candidate_template_keys[index]
                template = residue_templates[template_key]
                mapping = mappings[index]
                H_miss = all_stats["H_missing"][index]
                log["chosen_by_fewest_missing_H"][residue_key] = template_key

            H_miss = all_stats["H_missing"][index]
            H_excess = all_stats["H_excess"][index]
            if H_miss or H_excess: 
                log["matched_with_H_anomaly"][residue_key] = [
                    template_key, 
                    {"H_miss": H_miss, "H_excess": len(H_excess)}
                ]
            bond_excess = all_stats["bonded_atoms_excess"][index]
            if bond_excess:
                log["matched_with_excess_bond"].append(residue_key)
                logger.warning(f"matched with excess inter-residue bond(s): {residue_key}")

            if template is None:
                rdkit_mol = None
                atom_names = None
                mapping = None
            else:
                rdkit_mol = cls._build_rdkit_mol(
                    raw_mol,
                    template,
                    mapping,
                    H_miss,
                )
                atom_names = template.atom_names
            monomers[residue_key] = Monomer(
                raw_mol,
                rdkit_mol,
                mapping,
                input_resname,
                template_key,
                atom_names,
            )
            monomers[residue_key].template = template
            if template_key is not None and template_key in template_charges:
                monomers[residue_key].template_charge = template_charges[template_key]
            else:
                monomers[residue_key].template_charge = None

        return monomers, log

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

        for residue_id, monomer in monomers.items():
            if monomer.rdkit_mol is None:
                continue

            padded_mol = monomer.rdkit_mol
            mapidx_pad = {atom.GetIdx(): atom.GetIdx() for atom in padded_mol.GetAtoms()}
            padded_links = set()

            for atom_index, link_label in monomer.template.link_labels.items():
                if (atom_index, link_label) in padded_links:
                    continue

                # Find all bonds involving this link atom
                found_bond = False
                for (r1_id, r2_id), bond_list in bonds.items():
                    for idx1, idx2 in bond_list:
                        if r1_id == residue_id and idx1 == atom_index:
                            adjacent_rid = r2_id
                            adjacent_atom_index = idx2
                            adjacent_mol = monomers[adjacent_rid].rdkit_mol
                            bond_use_count[(r1_id, r2_id)] += 1
                            found_bond = True
                            break
                        elif r2_id == residue_id and idx2 == atom_index:
                            adjacent_rid = r1_id
                            adjacent_atom_index = idx1
                            adjacent_mol = monomers[adjacent_rid].rdkit_mol
                            bond_use_count[(r1_id, r2_id)] += 1
                            found_bond = True
                            break
                    if found_bond:
                        break

                if not found_bond:
                    adjacent_mol = None
                    adjacent_atom_index = None

                # Always call the padder
                padded_mol, mapidx = padders[link_label](
                    padded_mol, adjacent_mol, atom_index, adjacent_atom_index
                )

                # Update mapidx_pad
                tmp = {}
                for i, j in enumerate(mapidx):
                    if j is None:
                        continue  # new atom
                    if j not in mapidx_pad:
                        continue  # previously added atom, not traceable
                    tmp[i] = mapidx_pad[j]
                mapidx_pad = tmp
                padded_links.add((atom_index, link_label))

            # Update hydrogen positions and add hydrogens
            inv_map = {v: k for k, v in mapidx_pad.items()}
            padded_H_idxs = []
            padded_H_idxs_in_rdkit_mol = []

            for atom_index in monomer.template.link_labels:
                heavy_atom = monomer.rdkit_mol.GetAtomWithIdx(atom_index)
                for neighbor in heavy_atom.GetNeighbors():
                    if neighbor.GetAtomicNum() != 1:
                        continue
                    if neighbor.GetIdx() in monomer.mapidx_to_raw:
                        continue  # already has a known position
                    padded_idx = inv_map.get(neighbor.GetIdx())
                    if padded_idx is not None:
                        padded_H_idxs.append(padded_idx)
                        padded_H_idxs_in_rdkit_mol.append(neighbor.GetIdx())

            update_H_positions(padded_mol, padded_H_idxs)
            padded_mols[residue_id] = (padded_mol, mapidx_pad)

            # update added H positions in Monomer.rdkit_mol, just in case anyone
            # considers those positions (as Polymer.to_pdb() used to).
            source = padded_mol.GetConformer()
            destination = monomer.rdkit_mol.GetConformer()
            for i, j in zip(padded_H_idxs, padded_H_idxs_in_rdkit_mol):
                destination.SetAtomPosition(j, source.GetAtomPosition(i))

        # Validate all bonds were used twice (A padded with B, and B with A)
        err_msg = ""
        for (r1, r2), bond_list in bonds.items():
            expected = 2 * len(bond_list)
            actual = bond_use_count[(r1, r2)]
            if actual != expected:
                err_msg += (
                    f"Expected {expected} paddings for ({r1}, {r2}) with bonds {bond_list}, "
                    f"but got {actual}\n"
                )
        if err_msg:
            raise RuntimeError(err_msg)

        return padded_mols

    

    # ----- parser wrappers: delegate to meeko/polymer/parsers.py -----

    @staticmethod
    def _add_if_new(to_dict, key, value, repeat_log):
        from . import parsers
        return parsers._add_if_new(to_dict, key, value, repeat_log)

    @staticmethod
    def _pdb_to_residue_mols(pdb_string, wanted_altloc=None, default_altloc=None):
        from . import parsers
        return parsers.pdb_to_residue_mols(pdb_string, wanted_altloc, default_altloc)

    @staticmethod
    def _pqr_to_residue_mols(pqr_string):
        from . import parsers
        return parsers.pqr_to_residue_mols(pqr_string)

    @staticmethod
    def _prody_to_residue_mols(prody_obj, wanted_altloc_dict=None, default_altloc=None):
        from . import parsers
        return parsers.prody_to_residue_mols(
            prody_obj, wanted_altloc_dict, default_altloc
        )



    def to_pdb(self, new_positions: Optional[dict]=None):
        """
        Parameters
        ----------
        new_positions: dict (str -> dict (int -> (float, float, float)))
                             |            |      |
                    residue_id            |      |
                                 atom_index      |
                                                 new_position
        Returns
        _______
        pdb_string: str
        """    

        if new_positions is None:
            new_positions = {}
        valid_monomers = self.get_valid_monomers()

        # check that residue IDs passed in new_positions are valid
        unknown_res_ids = set()
        for res_id in new_positions:
            if res_id not in valid_monomers:
                unknown_res_ids.add(res_id)
        if unknown_res_ids:
            msg = f"Residue IDs not in valid monomers: {unknown_res_ids}"
            raise ValueError(msg)

        pdbout = ""
        atom_count = 0
        pdb_line = "{:6s}{:5d} {:^4s} {:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}                      {:>2s} "
        pdb_line += eol
        for res_id, monomer in self.get_valid_monomers().items():
            rdkit_mol = monomer.rdkit_mol
            if res_id in new_positions:
                positions = get_updated_positions(
                    monomer,
                    new_positions[res_id],
                )
            else:
                rdkit_to_padded = {j: i for i, j in monomer.molsetup_mapidx.items()}
                positions = [monomer.molsetup.atoms[rdkit_to_padded[i]].coord for i in range(rdkit_mol.GetNumAtoms())]

            chain, resnum = res_id.split(":")
            if resnum[-1].isalpha():
                icode = resnum[-1]
                resnum = resnum[:-1]
            else:
                icode = ""
            resnum = int(resnum)

            for i, atom in enumerate(rdkit_mol.GetAtoms()):
                atom_count += 1
                props = atom.GetPropsAsDict()
                atom_name = monomer.atom_names[i]
                x, y, z = positions[i]
                element = mini_periodic_table[atom.GetAtomicNum()]
                pdbout += pdb_line.format(
                    "ATOM",
                    atom_count,
                    atom_name,
                    monomer.input_resname,
                    chain,
                    resnum,
                    icode,
                    x,
                    y,
                    z,
                    element,
                )
        return pdbout

    def export_static_atom_params(self):
        """

        Returns
        -------
        atom_params: dict
        coords: list
        """
        atom_params = {}
        counter_atoms = 0
        coords = []
        dedicated_attribute = (
            "charge",
            "atom_type",
        )  # molsetup has a dedicated attribute
        for res_id in self.get_valid_monomers():
            molsetup = self.monomers[res_id].molsetup
            wanted_atom_indices = []
            for atom in molsetup.atoms:
                if not atom.is_ignore and not self.monomers[res_id].is_flexres_atom[atom.index]:
                    wanted_atom_indices.append(atom.index)
                    coords.append(molsetup.get_coord(atom.index))
            for key, values in molsetup.atom_params.items():
                atom_params.setdefault(key, [None] * counter_atoms)  # add new "column"
                for i in wanted_atom_indices:
                    atom_params[key].append(values[i])
            # This was reworked to specifically address the new MoleculeSetup structure. Needs re-thinking
            charge_dict = {atom.index: atom.charge for atom in molsetup.atoms}
            atom_type_dict = {atom.index: atom.atom_type for atom in molsetup.atoms}
            for key in dedicated_attribute:
                atom_params.setdefault(key, [None] * counter_atoms)  # add new "column"
                if key == "charge":
                    values_dict = charge_dict
                else:
                    values_dict = atom_type_dict
                for i in wanted_atom_indices:
                    atom_params[key].append(values_dict[i])
            counter_atoms += len(wanted_atom_indices)
            added_keys = set(molsetup.atom_params).union(dedicated_attribute)
            for key in set(atom_params).difference(
                added_keys
            ):  # <key> missing in current molsetup
                atom_params[key].extend(
                    [None] * len(wanted_atom_indices)
                )  # fill in incomplete "row"
        if hasattr(self, "param_rename"):  # e.g. "gasteiger" -> "q"
            for key, new_key in self.param_rename.items():
                atom_params[new_key] = atom_params.pop(key)
        return atom_params, coords

    # region Filtering Residues
    def get_ignored_monomers(self):
        return {k: v for k, v in self.monomers.items() if v.rdkit_mol is None}

    def get_valid_monomers(self):
        return {k: v for k, v in self.monomers.items() if v.rdkit_mol is not None}

    # endregion

