from rdkit import Chem
from rdkit.Chem import rdMolInterchange


SERIALIZATION_SEPARATOR_CHAR = ","


def rdkit_mol_from_json(json_str: str):
    """
    Takes in a JSON string and attempts to use RDKit's JSON to Mols utility to extract just one RDKitMol from the
    json string. If none or more than one Mols are returned, raises an error.

    Parameters
    ----------
    json_str: str
        A JSON string representing an RDKit Mol.

    Returns
    -------
    rdkit_mol: rdkit.Chem.rdchem.Mol
        An RDKit Mol object corresponding to the input JSON string

    Raises
    ------
    ValueError
        If no RDKitMol objects are returned, or if more than one is returned, throws a ValueError.
    """
    if json_str is None:
        return None
    rdkit_mols = rdMolInterchange.JSONToMols(json_str)
    if len(rdkit_mols) != 1:
        raise ValueError(
            f"Expected 1 rdkit mol from json string but got {len(rdkit_mols)}"
        )
    Chem.SanitizeMol(rdkit_mols[0])  # needed to compute gasteiger charges
    return rdkit_mols[0]


def tuple_to_string(input_tuple: tuple):
    """
    Converts a tuple to a JSON serializable string.

    Parameters
    ----------
    input_tuple: tuple
        A tuple to convert to a JSON serializable string.

    Returns
    -------
    A string representation of the tuple using the specified serialization separator character.
    """
    return SERIALIZATION_SEPARATOR_CHAR.join([str(i) for i in input_tuple])


def string_to_tuple(input_string: str, element_type: type = str):
    """
    Takes a JSON string and converts it back to a tuple. If element type is specified, converts all elements of the
    tuple to that type.

    Parameters
    ----------
    input_string: str
        String deserialized from JSON.
    element_type: type
        Data type for all of the elements of the tuple.

    Returns
    -------
    A deserialized tuple with the specified element type.
    """
    if element_type is not str:
        return tuple(
            [element_type(i) for i in input_string.split(SERIALIZATION_SEPARATOR_CHAR)]
        )
    else:
        return tuple(input_string)
