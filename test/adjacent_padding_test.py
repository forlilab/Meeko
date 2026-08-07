
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