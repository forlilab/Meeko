#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko Espaloma typer
#

import numpy as np
import logging
from math import pi

from rdkit import Chem

from .molsetup import RDKitMoleculeSetup


logger = logging.getLogger(__name__)

# The functions below build the charge-only graph inputs directly from the RDKit molecule,
# mirroring espaloma/graphs/utils/read_homogeneous_graph.py (fp_rdkit, from_openff_toolkit_mol),
# without the OpenFF back and forth conversion and without creating torch tensors for each atom.

HYBRIDIZATION_RDKIT = {
    Chem.rdchem.HybridizationType.SP: [1, 0, 0, 0, 0],
    Chem.rdchem.HybridizationType.SP2: [0, 1, 0, 0, 0],
    Chem.rdchem.HybridizationType.SP3: [0, 0, 1, 0, 0],
    Chem.rdchem.HybridizationType.SP3D: [0, 0, 0, 1, 0],
    Chem.rdchem.HybridizationType.SP3D2: [0, 0, 0, 0, 1],
    Chem.rdchem.HybridizationType.S: [0, 0, 0, 0, 0],
}

# d and f block elements in periodic table, whose radicals OpenFF's Molecule.from_rdkit accepts
_D_AND_F_BLOCK_ELEMENTS = {*range(21, 31), *range(39, 49), *range(57, 81), *range(89, 113)}


def fp_rdkit(atom):
    """ espaloma's fp_rdkit, as a list """
    return [
        atom.GetTotalDegree(),
        # Note: Discard resonance-variant features
        # Issue related to https://github.com/choderalab/espaloma_charge/issues/18
        # atom.GetTotalValence(),
        # atom.GetExplicitValence(),
        # atom.GetFormalCharge(),
        atom.GetIsAromatic() * 1.0,
        atom.GetMass(),
        atom.IsInRingSize(3) * 1.0,
        atom.IsInRingSize(4) * 1.0,
        atom.IsInRingSize(5) * 1.0,
        atom.IsInRingSize(6) * 1.0,
        atom.IsInRingSize(7) * 1.0,
        atom.IsInRingSize(8) * 1.0,
    ] + HYBRIDIZATION_RDKIT[atom.GetHybridization()]


def rdmol_with_openff_perception(mol):
    """ The RDKit molecule that espaloma uses after OpenFF Molecule.from_rdkit and
    Molecule.to_rdkit contains information that is changed / lost from the interconversion:
    no isotopes, MDL aromaticity and hybridization perceived under
    that model, SSSR ring info. Radicals are rejected as in Molecule.from_rdkit.
    """
    from openff.toolkit.utils.exceptions import RadicalsNotSupportedError

    rdmol = Chem.Mol(mol)
    for atom in rdmol.GetAtoms():
        atom.SetIsotope(0)
    Chem.Kekulize(rdmol, clearAromaticFlags=True)
    Chem.SetAromaticity(rdmol, Chem.AromaticityModel.AROMATICITY_MDL)
    Chem.SanitizeMol(rdmol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_ADJUSTHS ^ Chem.SANITIZE_SETAROMATICITY)
    Chem.SetAromaticity(rdmol, Chem.AromaticityModel.AROMATICITY_MDL)
    Chem.GetSSSR(rdmol)
    for atom in rdmol.GetAtoms():
        if atom.GetAtomicNum() not in _D_AND_F_BLOCK_ELEMENTS and atom.GetNumRadicalElectrons() != 0:
            raise RadicalsNotSupportedError(
                "The OpenFF Toolkit does not currently support parsing molecules with S- and P-block radicals. "
                f"Found {atom.GetNumRadicalElectrons()} radical electrons on molecule {Chem.MolToSmiles(rdmol)}."
            )
    return rdmol


def from_rdkit_mol(mol):
    """ Atom features "h0" and total charge "sum_q" as in espaloma's from_openff_toolkit_mol """
    import torch

    rdmol = rdmol_with_openff_perception(mol)
    n_atoms = rdmol.GetNumAtoms()
    total_charge = Chem.GetFormalCharge(rdmol)
    sum_q = torch.Tensor([[total_charge]])

    h_v = np.zeros((n_atoms, 100), dtype=np.float32)
    h_v[np.arange(n_atoms), [atom.GetAtomicNum() for atom in rdmol.GetAtoms()]] = 1.0

    h_v_fp = np.array([fp_rdkit(atom) for atom in rdmol.GetAtoms()], dtype=np.float32)

    h_v = np.concatenate([h_v, h_v_fp], axis=-1)  # (n_atoms, 114)

    return torch.from_numpy(h_v), sum_q

# module layout of the espaloma model that the reduced charge-only path relies on
_EXPECTED_MODEL_LAYOUT = [
    "Sequential",
    "JanossyPooling",
    "JanossyPoolingWithSmirnoffImproper",
    "ExpCoefficients",
    "ChargeEquilibrium",
    "LinearMixtureToOriginal",
]

class EspalomaTyper:

    def __init__(self,
                 version='latest'
    ):
        try:
            import espaloma as esp
        except ImportError:
            raise ImportError("Espaloma is required")
        
        try:
            import torch
        except ImportError:
            raise ImportError("Pytorch is required")
        
        try:
            from openff.toolkit.topology import Molecule
        except ImportError:
            raise ImportError("OpenFF is required")

        # Fetch and load latest pretrained model from GitHub
        self.espaloma_model = esp.get_model(version)

        # run the model on the GPU when available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.espaloma_model.to(self.device)

        # For charges only, a reduced graph is built and only the model parts
        # that produce atom charges (n1 "q") are run.
        layout = [type(module).__name__ for module in self.espaloma_model]
        self.charges_only_supported = (
            layout == _EXPECTED_MODEL_LAYOUT
            and set(self.espaloma_model[1].out_features.get(1, {})) == {"e", "s"}
        )
        if not self.charges_only_supported:
            logger.warning(
                f"Unrecognized espaloma model layout {layout} (version {version}), "
                "falling back to building the full espaloma graph and running the whole model "
                "for charges. Espaloma charge assignment can be considerably slower."
            )

        # store methods in instance, otherwise they are out of scope
        self.openffmol_from_rdkit = Molecule.from_rdkit
        self.EspalomaGraph = esp.Graph
        self.Pytorch = torch

    def get_espaloma_graph(self, molsetup, charges_only=False):
        """ Apply espaloma model to a graph representation of the molecule.

        If charges_only, only atom charges (n1 "q") are computed, using a reduced
        graph and only the model parts that produce them.
        """

        if not isinstance(molsetup, RDKitMoleculeSetup):
            raise NotImplementedError("need rdkit molecule for espaloma typing")

        rdmol = molsetup.mol
        if not (charges_only and self.charges_only_supported):
            openffmol = self.openffmol_from_rdkit(rdmol, hydrogens_are_explicit=True, allow_undefined_stereo=True)
            molgraph = self.EspalomaGraph(openffmol)
            heterograph = molgraph.heterograph.to(self.device)
            self.espaloma_model(heterograph)
            molgraph.heterograph = heterograph.to("cpu")
            return molgraph

        heterograph = self._get_charges_only_heterograph(rdmol).to(self.device)
        representation, janossy, charge_equilibrium = self.espaloma_model[0], self.espaloma_model[1], self.espaloma_model[4]
        with self.Pytorch.no_grad():
            representation(heterograph)
            # atom level readout, as in espaloma's JanossyPooling
            x = janossy.sequential_1(g=None, x=heterograph.nodes["n1"].data["h"])
            heterograph.nodes["n1"].data["e"] = janossy.f_out_1_to_e(x)
            heterograph.nodes["n1"].data["s"] = janossy.f_out_1_to_s(x)
            charge_equilibrium(heterograph)

        return self.EspalomaGraph(heterograph=heterograph.to("cpu"))

    def _get_charges_only_heterograph(self, rdmol):
        """ Build only the parts of espaloma's heterograph needed for atom charges.

        Mirrors espaloma's featurization and bond enumeration, so the atom features
        and edge order are the same as in the full heterograph.
        """
        import dgl
        from espaloma.graphs.utils.read_heterogeneous_graph import duplicate_index_ordering

        torch = self.Pytorch
        h0, sum_q = from_rdkit_mol(rdmol)
        n_atoms = rdmol.GetNumAtoms()

        bonds = np.array([(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in rdmol.GetBonds()])
        if len(bonds) == 0:
            bonds = np.empty((0, 2), dtype=np.int64)
        bonds = duplicate_index_ordering(bonds)

        atoms = torch.arange(n_atoms)
        zeros = torch.zeros(n_atoms, dtype=torch.int64)
        data = {
            ("n1", "n1_neighbors_n1", "n1"): (torch.from_numpy(bonds[:, 0]), torch.from_numpy(bonds[:, 1])),
            ("n1", "n1_in_g", "g"): (atoms, zeros),
            ("g", "g_has_n1", "n1"): (zeros, atoms),
        }

        heterograph = dgl.heterograph(data, num_nodes_dict={"n1": n_atoms, "g": 1})
        heterograph.nodes["n1"].data["h0"] = h0
        heterograph.nodes["g"].data["sum_q"] = sum_q
        return heterograph
    
    def set_espaloma_charges(self, molsetup, molgraph):
        """Grab charges from graph node and set them to the molsetup """

        charges = [float(q) for q in molgraph.nodes["n1"].data["q"]]
        total_charge = 0.0
        for i in range(len(charges)):
            # print("%12.4f %12.4f" % (molsetup.charge[i], charges[i]))
            molsetup.atoms[i].charge = charges[i]
            total_charge += charges[i]
        for j in range(i+1, len(molsetup.atoms)):
            if molsetup.get_charge(j) != 0.:
                raise RuntimeError("expected zero charge beyond real atoms, at this point")
        
            
    def set_espaloma_dihedrals(self, molsetup: RDKitMoleculeSetup, molgraph):
        """Grab dihedrals from graph node and set them to the molsetup """

        # TODO replace torch with np, not tensors but arrays instead so no importing Pytorch

        if (
            "periodicity" not in molgraph.nodes["n4"].data
            or "phase" not in molgraph.nodes["n4"].data
        ):

            molgraph.nodes["n4"].data["periodicity"] = self.Pytorch.arange(
                1, 7
            )[None, :].repeat(molgraph.heterograph.number_of_nodes("n4"), 1)

            molgraph.nodes["n4"].data["phases"] = self.Pytorch.zeros(
                molgraph.heterograph.number_of_nodes("n4"), 6
            )

        n_torsions = range(molgraph.heterograph.number_of_nodes("n4"))
        torsions = []
        partaking_atoms = {}

        for idx in n_torsions:
            idx0 = molgraph.nodes["n4"].data["idxs"][idx, 0].item()
            idx1 = molgraph.nodes["n4"].data["idxs"][idx, 1].item()
            idx2 = molgraph.nodes["n4"].data["idxs"][idx, 2].item()
            idx3 = molgraph.nodes["n4"].data["idxs"][idx, 3].item()

            # assuming both (a,b,c,d) and (d,c,b,a) are listed for every torsion, only pick one of the orderings
            if idx0 < idx3:
                periodicities = molgraph.nodes["n4"].data[
                    "periodicity"
                ][idx]
                phases = molgraph.nodes["n4"].data["phases"][idx]
                ks = molgraph.nodes["n4"].data["k"][idx]

                this_torsion = []
                for sub_idx in range(ks.flatten().shape[0]):
                    k = ks[sub_idx].item()
                    if k != 0.0:
                        _periodicity = periodicities[sub_idx].item()
                        _phase = phases[sub_idx].item()

                        if k < 0:
                            k = -k
                            _phase = pi - _phase
                            
                        k = k * 627.509  # hartree to kcal/mol

                        partaking_atoms[(idx0, idx1, idx2, idx3)] = len(torsions)
                        this_torsion.append({"k": k, "phase": _phase, "periodicity": _periodicity, "idivf": 1.0}) #idivf is not used here, just for compatibility

                torsions.append(this_torsion)

        molsetup.dihedral_interactions = torsions
        molsetup.dihedral_partaking_atoms = partaking_atoms
