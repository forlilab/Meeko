
import pytest

from meeko.espaloma_onnx import charges_from_onnx
from espaloma_charge import charge

from rdkit import Chem
import numpy as np

SMILES =  ["CCO", "CC(=O)[O-]", "CC(=O)Oc1ccccc1C(=O)O", "C[NH3+]", "c1ccsc1"]


def test_espaloma_onnx():

    for smi in SMILES:

        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        onnx_charges = charges_from_onnx(m)
        esp_charges = charge(m)

        # pprint(onnx_charges)
        # pprint(esp_charges)
        d = np.abs(onnx_charges - esp_charges).max()
        assert d == pytest.approx(0, abs=1e-5)
        