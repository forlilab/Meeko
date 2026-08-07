
import pathlib
import pytest

from meeko import Polymer
from meeko import PDBQTWriterLegacy
from meeko import MoleculePreparation
from meeko import ResidueChemTemplates
from meeko.polymer import PolymerCreationError

from rdkit import Chem
import numpy as np

workdir = pathlib.Path(__file__)
datadir = workdir.parents[0] / "polymer_data"

pdb_file = datadir / "5fnt.pdb"
disulfide_bridge = datadir / "just_a_disulfide_bridge.pdb"
disulfide_adjacent = datadir / "disulfide_bridge_in_adjacent_residues.pdb"
loop_with_disulfide = datadir / "loop_with_disulfide.pdb"
chem_templates = ResidueChemTemplates.create_from_defaults()
mk_prep = MoleculePreparation(compute_charges=True)


def test_build_adjacency():
    with open(pdb_file) as f:
        pdb_string = f.read()

    polymer = Polymer.from_pdb_string(pdb_string)

    adj = polymer.adjacency(polymer.bonds)

    assert len(adj) == len(polymer.monomers)

    for (r1, r2), _ in polymer.bonds.items():
        assert r1 in adj 
        assert r2 in adj

def test_paddings():

    with open(pdb_file) as f:
        pdb_string = f.read()


    # mk_config = {}
    # mk_config["adj_padding"] = True
    # mk_config["compute_charges"] = True

    polymer = Polymer.from_pdb_string(pdb_string)

    padders = polymer.residue_chem_templates.padders
    monomers = polymer.monomers
    bonds = polymer.bonds

    adj = polymer.adjacency(polymer.bonds)

    padded_templ = polymer._build_padded_mols(monomers, bonds, padders)
    padded_adj = polymer.build_adj_padding(adj)

    assert len(padded_templ) == len(padded_adj)

    for id, _ in padded_templ.items():
        assert id in padded_adj

def test_flexres_pdbqt():
    with open(loop_with_disulfide) as f:
        pdb_string = f.read()
    set_templates = {
        ":6": "CYX",
        ":17": "CYX",
    }  # TODO remove this to test use of bonds to set templates
    polymer = Polymer.from_pdb_string(
        pdb_string,
        chem_templates,
        mk_prep,
        set_templates,
        blunt_ends=[(":5", 0), (":18", 2)],
        adj_padding=True,
    )
    res11 = polymer.monomers[":11"]
    assert sum(res11.is_flexres_atom) == 0
    polymer.flexibilize_sidechain(":11", mk_prep)
    assert sum(res11.is_flexres_atom) == 9
    rigid, flex_dict = PDBQTWriterLegacy.write_from_polymer(polymer)
    nr_rigid_atoms = len(rigid.splitlines())
    assert nr_rigid_atoms == 124
    nr_flex_atoms = 0
    for line in flex_dict[":11"].splitlines():
        nr_flex_atoms += int(line.startswith("ATOM"))
    assert nr_flex_atoms == 9


def test_disulfides():
    with open(disulfide_bridge, "r") as f:
        pdb_text = f.read()
    # auto disulfide detection is enabled by default
    polymer_disulfide = Polymer.from_pdb_string(
        pdb_text,
        chem_templates,
        mk_prep,
        blunt_ends=[("B:22", 0), ("B:22", 2), ("B:95", 0), ("B:95", 2)],
        adj_padding=True,
    )
    # the disulfide bond is detected, and it expects two paddings,
    # but forcing CYS not CYX disables the padding, so error expected
    with pytest.raises(RuntimeError):
        polymer_thiols = Polymer.from_pdb_string(
            pdb_text,
            chem_templates,
            mk_prep,
            set_template={"B:22": "CYS"},
            blunt_ends=[("B:22", 0), ("B:22", 2), ("B:95", 0), ("B:95", 2)],
        )

    # remove bond and expect CYS between residues
    # currently, all bonds between a pair of residues will be removed
    polymer_thiols = Polymer.from_pdb_string(
        pdb_text,
        chem_templates,
        mk_prep,
        bonds_to_delete=[("B:22", "B:95")],
        blunt_ends=[("B:22", 0), ("B:22", 2), ("B:95", 0), ("B:95", 2)],
        adj_padding=True,
    )

    # check residue names
    assert polymer_disulfide.monomers["B:22"].residue_template_key == "CYX"
    assert polymer_disulfide.monomers["B:95"].residue_template_key == "CYX"
    assert polymer_thiols.monomers["B:22"].residue_template_key == "CYS"
    assert polymer_thiols.monomers["B:95"].residue_template_key == "CYS"

def test_disulfide_adjacent():
    """ disulfide bridge in adjacent residues broke a version of the code
        that assumed only one bond between each pair of residues
    """
    with open(disulfide_adjacent, "r") as f:
        pdb_text = f.read()
    polymer = Polymer.from_pdb_string(
        pdb_text,
        chem_templates,
        mk_prep,
        adj_padding=True,
    )

