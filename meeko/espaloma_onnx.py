"""Partial charges from an RDKit molecule using the exported espaloma_charge ONNX model.

The featurization mirrors espaloma_charge.utils.from_rdkit_mol (116 features per atom,
both bond directions as edges, total charge = sum of formal charges). If the espaloma_charge
featurization changes, re-export the model and update `_featurize` to match.
"""

import functools

import numpy as np
from rdkit import Chem
from importlib.resources import files

SUPPORTED_ELEMENTS = frozenset(["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"])

_N_FEATURES = 116

# one-hot position (offset from 111) of each hybridization; None means all zeros
_HYBRIDIZATION = {
    Chem.rdchem.HybridizationType.SP: 0,
    Chem.rdchem.HybridizationType.SP2: 1,
    Chem.rdchem.HybridizationType.SP3: 2,
    Chem.rdchem.HybridizationType.SP3D: 3,
    Chem.rdchem.HybridizationType.SP3D2: 4,
    Chem.rdchem.HybridizationType.S: None,
    Chem.rdchem.HybridizationType.UNSPECIFIED: None,
}


@functools.lru_cache(maxsize=None)
def _get_session(onnx_path, providers):
    """Create the onnxruntime session once."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # tiny model: threading overhead costs more than it saves
    return ort.InferenceSession(onnx_path, sess_options=opts, providers=list(providers))


def _featurize(mol, total_charge=None):
    """RDKit mol -> (h0 (N,116) float32, edge_index (2,E) int64, total charge float)."""
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        raise ValueError("molecule has no atoms")

    h0 = np.zeros((n_atoms, _N_FEATURES), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        if symbol not in SUPPORTED_ELEMENTS:
            raise ValueError(
                f"Element {symbol} is not supported by the espaloma charge model "
                f"(supported: {SUPPORTED_ELEMENTS})."
            )
        if atom.GetTotalNumHs(includeNeighbors=False) > 0:
            raise ValueError(
                "molecule has implicit hydrogens; the model expects explicit hydrogens "
                "(use Chem.AddHs before calling)."
            )
        hybridization = atom.GetHybridization()
        if hybridization not in _HYBRIDIZATION:
            raise ValueError(f"Unsupported hybridization {hybridization} on atom {i}")

        h0[i, atom.GetAtomicNum()] = 1.0
        h0[i, 100:111] = [
            atom.GetTotalDegree(),
            atom.GetTotalValence(),
            atom.GetExplicitValence(),
            atom.GetIsAromatic() * 1.0,
            atom.GetMass(),
            *(atom.IsInRingSize(k) * 1.0 for k in range(3, 9)),
        ]
        hyb_idx = _HYBRIDIZATION[hybridization]
        if hyb_idx is not None:
            h0[i, 111 + hyb_idx] = 1.0

    bonds = np.array(
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()],
        dtype=np.int64,
    ).reshape(-1, 2).T
    edge_index = np.ascontiguousarray(np.concatenate([bonds, bonds[::-1]], axis=1))

    if total_charge is None:
        total_charge = float(sum(a.GetFormalCharge() for a in mol.GetAtoms()))
    return h0, edge_index, float(total_charge)


def charges_from_onnx(mol, onnx_path = None, total_charge=None, providers=("CPUExecutionProvider",)):
    """Partial charges for one RDKit molecule.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Sanitized molecule with explicit hydrogens.
    onnx_path : str
        Path to the exported espaloma_charge.onnx file.
    total_charge : float, optional
        Net charge to enforce. Default: sum of the atoms' formal charges, which is what
        espaloma_charge.charge() always uses (it ignores its own total_charge argument).
    providers : tuple of str
        onnxruntime execution providers, e.g. ("CUDAExecutionProvider", "CPUExecutionProvider").

    Returns
    -------
    np.ndarray, shape (n_atoms,), float32, in the atom order of `mol`.
    """

    if onnx_path is None:
        onnx_path = files("meeko.data").joinpath("espaloma_charges_0.0.8.onnx")
    
    h0, edge_index, q_tot = _featurize(mol, total_charge)
    session = _get_session(str(onnx_path), tuple(providers))
    q = session.run(
        ["charges"],
        {
            "h0": h0,
            "edge_index": edge_index,
            "batch": np.zeros(h0.shape[0], dtype=np.int64),
            "total_charge": np.array([[q_tot]], dtype=np.float32),
        },
    )[0]
    return q


def charges_from_onnx_multiple(mols, onnx_path = None, providers=("CPUExecutionProvider",)):
    """Partial charges for a list of molecules in a single model call.

    Returns a list of arrays, one per molecule. Total charge is the formal-charge sum.
    """

    if onnx_path is None:
            onnx_path = files("meeko.data").joinpath("espaloma_charges_0.0.8.onnx")
            
    mols = list(mols)
    if not mols:
        return []
    h0s, eis, batch, totals, offset = [], [], [], [], 0
    for g, mol in enumerate(mols):
        h0, edge_index, q_tot = _featurize(mol)
        h0s.append(h0)
        eis.append(edge_index + offset)
        batch.append(np.full(h0.shape[0], g, dtype=np.int64))
        totals.append(q_tot)
        offset += h0.shape[0]

    session = _get_session(str(onnx_path), tuple(providers))
    q = session.run(
        ["charges"],
        {
            "h0": np.concatenate(h0s),
            "edge_index": np.concatenate(eis, axis=1),
            "batch": np.concatenate(batch),
            "total_charge": np.array(totals, dtype=np.float32).reshape(-1, 1),
        },
    )[0]
    return np.split(q, np.cumsum([m.GetNumAtoms() for m in mols])[:-1])