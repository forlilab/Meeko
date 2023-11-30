#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko Espaloma typer
#

# import numpy as np
from math import pi


from .molsetup import RDKitMoleculeSetup

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

        # store methods in instance, otherwise they are out of scope
        self.openffmol_from_rdkit = Molecule.from_rdkit 
        self.EspalomaGraph = esp.Graph
        self.Pytorch = torch

    def get_espaloma_graph(self, molsetup):
        """ Apply espaloma model to a graph representation of the molecule. """

        if not isinstance(molsetup, RDKitMoleculeSetup):
            raise NotImplementedError("need rdkit molecule for espaloma typing")
        
        rdmol = molsetup.mol
        openffmol = self.openffmol_from_rdkit(rdmol, hydrogens_are_explicit=True, allow_undefined_stereo=True)
        molgraph = self.EspalomaGraph(openffmol)
        self.espaloma_model(molgraph.heterograph)
        
        return molgraph
    
    def set_espaloma_charges(self, molsetup, molgraph):
        """Grab charges from graph node and set them to the molsetup """

        charges = [float(q) for q in molgraph.nodes["n1"].data["q"]]
        total_charge = 0.0
        for i in range(len(charges)):
            #print("%12.4f %12.4f" % (molsetup.charge[i], charges[i]))
            molsetup.charge[i] = charges[i] 
            total_charge += charges[i]
        for j in range(i+1, len(molsetup.charge)):
            if molsetup.charge[j] != 0.:
                raise RuntimeError("expected zero charge beyond real atoms, at this point")
            
    def set_espaloma_dihedrals(self, molsetup, molgraph):
        """Grab dihedrals from graph node and set them to the molsetup """

        ENERGY_UNIT_FACTOR = 0.00038087988 #Avoid relying on OpenFF for unit conversion 
        
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
                            
                        k = k/0.00038087988

                        partaking_atoms[(idx0, idx1, idx2, idx3)] = len(torsions)
                        this_torsion.append({"k": k, "phase": _phase, "periodicity": _periodicity, "idivf": 1.0}) #idivf is not used here, just for compatibility

                torsions.append(this_torsion)

        molsetup.dihedral_interactions = torsions
        molsetup.dihedral_partaking_atoms = partaking_atoms
