#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko Espaloma typer
#

# import numpy as np
import logging
from math import pi


from .molsetup import RDKitMoleculeSetup


logger = logging.getLogger(__name__)

# loaded espaloma models, keyed by version, so they are loaded once per process
_espaloma_models = {}
# whether the model layout supports the reduced charge-only path, keyed by version
_charges_only_supported = {}

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
        # check if model is loaded before loading
        if version not in _espaloma_models:
            model = esp.get_model(version)
            _espaloma_models[version] = model
            # For charges only, a reduced graph is built and only the model parts
            # that produce atom charges (n1 "q") are run.
            layout = [type(module).__name__ for module in model]
            _charges_only_supported[version] = (
                layout == _EXPECTED_MODEL_LAYOUT
                and set(model[1].out_features.get(1, {})) == {"e", "s"}
            )
            if not _charges_only_supported[version]:
                logger.warning(
                    f"Unrecognized espaloma model layout {layout} (version {version}), "
                    "falling back to building the full espaloma graph and running the whole model "
                    "for charges. Espaloma charge assignment can be considerably slower."
                )
        self.espaloma_model = _espaloma_models[version]
        self.charges_only_supported = _charges_only_supported[version]

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
        openffmol = self.openffmol_from_rdkit(rdmol, hydrogens_are_explicit=True, allow_undefined_stereo=True)
        if not (charges_only and self.charges_only_supported):
            molgraph = self.EspalomaGraph(openffmol)
            self.espaloma_model(molgraph.heterograph)
            return molgraph

        heterograph = self._get_charges_only_heterograph(openffmol)
        representation, janossy, charge_equilibrium = self.espaloma_model[0], self.espaloma_model[1], self.espaloma_model[4]
        with self.Pytorch.no_grad():
            representation(heterograph)
            # atom level readout, as in espaloma's JanossyPooling
            x = janossy.sequential_1(g=None, x=heterograph.nodes["n1"].data["h"])
            heterograph.nodes["n1"].data["e"] = janossy.f_out_1_to_e(x)
            heterograph.nodes["n1"].data["s"] = janossy.f_out_1_to_s(x)
            charge_equilibrium(heterograph)

        return self.EspalomaGraph(heterograph=heterograph)

    def _get_charges_only_heterograph(self, openffmol):
        """ Build only the parts of espaloma's heterograph needed for atom charges.

        Uses espaloma's own featurization and bond enumeration, so the atom features
        and edge order are the same as in the full heterograph.
        """
        import numpy as np
        import dgl
        from espaloma.graphs.utils import offmol_indices, read_homogeneous_graph
        from espaloma.graphs.utils.read_heterogeneous_graph import duplicate_index_ordering

        torch = self.Pytorch
        homograph = read_homogeneous_graph.from_openff_toolkit_mol(openffmol)
        n_atoms = homograph.number_of_nodes()

        bonds = offmol_indices.bond_indices(openffmol)
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
        heterograph.nodes["n1"].data["h0"] = homograph.ndata["h0"]
        heterograph.nodes["g"].data["sum_q"] = homograph.ndata["sum_q"][0].reshape(1, 1)
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
