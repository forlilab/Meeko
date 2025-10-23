from rdkit import Chem
from rdkit.Chem import rdMolInterchange

import json
import logging
from typing import Optional, Any
from abc import ABC, abstractmethod


SERIALIZATION_SEPARATOR_CHAR = ","

def serialize_optional(serializer, value):
    return serializer(value) if value is not None else None


def convert_to_int_keyed_dict(data: Optional[dict[str, Any]]) -> Optional[dict[int, Any]]:
    if data is None:
        return None
    return {int(k): v for k, v in data.items()}


def convert_to_tuple_keyed_dict(data: Optional[dict[str, Any]], element_type: type = str) -> Optional[dict[tuple[int], Any]]:
    if data is None:
        return None
    return {string_to_tuple(k, element_type = element_type): v for k, v in data.items()}


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

class BaseJSONParsable(ABC):

    # Define in Individual Subclasses 
    expected_json_keys: Optional[frozenset[str]] = None

    @abstractmethod
    def json_encoder(cls, obj):
        pass
    
    @abstractmethod
    def _decode_object(cls, obj):
        pass

    # Inheritable JSON Interchange Functions
    @classmethod
    def json_decoder(cls, obj: dict[str, Any], check_keys: bool = True):
        # Avoid using json_decoder as object_hook for nested objects
        if not isinstance(obj, dict):
            return obj
        if check_keys and not cls.expected_json_keys.issubset(obj.keys()):
            return obj

        # Delegate specific decoding logic to a subclass-defined method
        return cls._decode_object(obj)
    
    @classmethod
    def access_with_deprecated_key(cls, obj: dict[str, Any], old_key: str, new_key: str): 
        outcome = obj.get(new_key)
        if not outcome: 
            logging.warning(
                    f"Key '{old_key}' is deprecated for class {cls.__name__} and has been replaced with '{new_key}'. "
                    f"Support for '{old_key}' is provided for backward compatibility. "
                )
            outcome = obj.get(old_key)
        return outcome

    @classmethod
    def from_json(cls, json_string: str):
        try: 
            # Log mismatched keys if non-critical 
            obj_dict = json.loads(json_string)
            actual_keys = obj_dict.keys()
            if actual_keys != cls.expected_json_keys:
                missing_keys = cls.expected_json_keys - actual_keys
                extra_keys = actual_keys - cls.expected_json_keys
                logging.warning(
                    f"Key mismatch for {cls.__name__}. "
                    f"Missing keys: {missing_keys}, Extra keys: {extra_keys}."
                )

            try: 
                obj = json.loads(json_string, object_hook=cls.json_decoder)
            
            # Error occurred within cls.json_decoder
            except ValueError as decoder_error: 
                raise RuntimeError(
                    f"An error occurred when creating {cls.__name__} from JSON."
                    f"Error: {decoder_error}"
                )

            # Validate the decoded object type
            if not isinstance(obj, cls):
                raise ValueError(
                    f"Decoded object is not an instance of {cls.__name__}. "
                    f"Got: {type(obj)}"
                )
            return obj

        # Error occurred within json.loads
        except Exception as parser_error: 
            raise RuntimeError(
                    f"Unable to load JSON string for {cls.__name__}. "
                    f"Error: {parser_error}"
            )

    @classmethod
    def from_dict(cls, obj: dict) -> "BaseJSONParsable":
        return cls.json_decoder(obj, check_keys=False)
    
    def to_json(self):
        return json.dumps(self, default=self.__class__.json_encoder)
