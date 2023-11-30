from meeko import (
    LinkedRDKitChorizo, 
    ChorizoResidue, 
    ResidueAdditionalConnection, 
    PDBQTWriterLegacy, 
    MoleculePreparation
    )

import pathlib
import pytest

# Example Files (should be moved to tests directory eventually)
ahhy_example = pathlib.Path("example/chorizo/AHHY.pdb")
just_one_ALA_missing = pathlib.Path("example/chorizo/just-one-ALA-missing-CB.pdb")
just_one_ALA = pathlib.Path("example/chorizo/just-one-ALA.pdb")
just_three_residues = pathlib.Path("example/chorizo/just-three-residues.pdb")


# TODO: add checks for untested chorizo fields (e.g. input options not indicated here)
# TODO: clean up tests by pulling repeated test logic into helper functions

def test_AHHY_all_static_residues():
    f = open(ahhy_example, 'r')
    pdb_string = f.read()
    chorizo = LinkedRDKitChorizo(pdb_string)
    # Asserts that the residues have been imported in a way that makes sense, and that all the 
    # private functions we expect to have run have run as expected.
    assert len(chorizo.residues) == 4
    assert len(chorizo.getIgnoredResidues()) == 0

    expected_suggested_mutations = {'A:HIS:2': 'A:HID:2', 'A:HIS:3': 'A:HIE:3'}
    assert chorizo.suggested_mutations == expected_suggested_mutations

    expected_residue_data = {
        'A:ALA:1': ChorizoResidue('A:ALA:1', 'ATOM      1  N   ALA A   1       6.061   2.529  -3.691  1.00  0.00           N  \nATOM      2  CA  ALA A   1       5.518   2.870  -2.403  1.00  0.00           C  \nATOM      3  C   ALA A   1       4.995   1.645  -1.690  1.00  0.00           C  \nATOM      4  O   ALA A   1       5.294   0.515  -2.156  1.00  0.00           O  \nATOM      5  CB  ALA A   1       4.421   3.891  -2.559  1.00  0.00           C  \n', None, 'A:HIS:2'),
        'A:HIS:2': ChorizoResidue('A:HIS:2', 'ATOM      6  N   HIS A   2       4.201   1.774  -0.543  1.00  0.00           N  \nATOM      7  CA  HIS A   2       3.690   0.569   0.155  1.00  0.00           C  \nATOM      8  C   HIS A   2       2.368   0.239  -0.349  1.00  0.00           C  \nATOM      9  O   HIS A   2       1.827   0.959  -1.278  1.00  0.00           O  \nATOM     10  CB  HIS A   2       3.958   0.658   1.602  1.00  0.00           C  \nATOM     11  CG  HIS A   2       3.518  -0.435   2.481  1.00  0.00           C  \nATOM     12  ND1 HIS A   2       4.232  -1.588   2.706  1.00  0.00           N  \nATOM     13  CD2 HIS A   2       2.407  -0.562   3.290  1.00  0.00           C  \nATOM     14  CE1 HIS A   2       3.592  -2.347   3.583  1.00  0.00           C  \nATOM     15  NE2 HIS A   2       2.438  -1.715   3.961  1.00  0.00           N  \nATOM     16  H13 HIS A   2       5.120  -1.737   2.040  1.00  0.00           H  \n', 'A:ALA:1', 'A:HIS:3'),
        'A:HIS:3': ChorizoResidue('A:HIS:3', 'ATOM     17  N   HIS A   3       1.527  -0.823   0.040  1.00  0.00           N  \nATOM     18  CA  HIS A   3       0.243  -1.075  -0.553  1.00  0.00           C  \nATOM     19  C   HIS A   3      -0.832  -0.126  -0.071  1.00  0.00           C  \nATOM     20  O   HIS A   3      -0.560   1.096   0.147  1.00  0.00           O  \nATOM     21  CB  HIS A   3      -0.214  -2.454  -0.694  1.00  0.00           C  \nATOM     22  CG  HIS A   3       0.654  -3.363  -1.491  1.00  0.00           C  \nATOM     23  ND1 HIS A   3       1.819  -2.985  -2.046  1.00  0.00           N  \nATOM     24  CD2 HIS A   3       0.457  -4.684  -1.802  1.00  0.00           C  \nATOM     25  CE1 HIS A   3       2.360  -4.051  -2.700  1.00  0.00           C  \nATOM     26  NE2 HIS A   3       1.515  -5.068  -2.538  1.00  0.00           N  \nATOM     27  H20 HIS A   3       1.651  -6.016  -2.918  1.00  0.00           H  \n', 'A:HIS:2', 'A:TYR:4'),
        'A:TYR:4': ChorizoResidue('A:TYR:4', 'ATOM     28  N   TYR A   4      -2.156  -0.543   0.154  1.00  0.00           N  \nATOM     29  CA  TYR A   4      -3.237   0.354   0.596  1.00  0.00           C  \nATOM     30  C   TYR A   4      -3.373   0.217   2.071  1.00  0.00           C  \nATOM     31  O   TYR A   4      -2.656  -0.595   2.677  1.00  0.00           O  \nATOM     32  CB  TYR A   4      -4.460  -0.123  -0.108  1.00  0.00           C  \nATOM     33  CG  TYR A   4      -5.699   0.602   0.156  1.00  0.00           C  \nATOM     34  CD1 TYR A   4      -6.089   1.698  -0.613  1.00  0.00           C  \nATOM     35  CD2 TYR A   4      -6.492   0.168   1.200  1.00  0.00           C  \nATOM     36  CE1 TYR A   4      -7.276   2.306  -0.282  1.00  0.00           C  \nATOM     37  CE2 TYR A   4      -7.679   0.783   1.528  1.00  0.00           C  \nATOM     38  CZ  TYR A   4      -8.060   1.866   0.764  1.00  0.00           C  \nATOM     39  OH  TYR A   4      -9.262   2.477   1.103  1.00  0.00           O  \nATOM     40  OXT TYR A   4      -4.293   0.998   2.728  1.00  0.00           O  \nATOM     41  H29 TYR A   4      -9.644   3.279   0.612  1.00  0.00           H  \n', 'A:HIS:3', None),
        }
            
    for residue_id in chorizo.residues:
        residue_object = chorizo.residues[residue_id]
        expected_object = expected_residue_data[residue_id]
        assert residue_object.residue_id == expected_object.residue_id
        assert residue_object.pdb_text == expected_object.pdb_text
        assert residue_object.previous_id == expected_object.previous_id
        assert residue_object.next_id == expected_object.next_id
        assert residue_object.rdkit_mol != None

        pdbqt_strings = PDBQTWriterLegacy.write_string_from_linked_rdkit_chorizo(chorizo)
        rigid_part, movable_part = pdbqt_strings
        rigid_part = "".join(rigid_part.splitlines()) # remove newline chars because Windows/Unix differ

        assert len(rigid_part) == 3476
        assert len(movable_part) == 0

def test_AHHY_flexible_residues():
    f = open(ahhy_example, 'r')
    pdb_string = f.read()
    chorizo = LinkedRDKitChorizo(pdb_string)
    assert len(chorizo.residues) == 4
    assert len(chorizo.getIgnoredResidues()) == 0

    mk_prep = MoleculePreparation()
    residue_id = "A:HIS:2"
    
    molsetup, mapidx, ignored_in_molsetup = chorizo.res_to_molsetup(residue_id, mk_prep)
    
    expected_mapidx = {2: 0, 3: 1, 4: 2, 5: 4, 6: 3, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 15, 13: 11, 14: 12, 15: 14, 16: 16, 17: 13, 19: 5}
    expected_ignored = []
    assert molsetup
    assert mapidx == expected_mapidx
    assert ignored_in_molsetup == expected_ignored

def test_just_three_padded_mol():
    f = open(just_three_residues, 'r')
    pdb_string = f.read()
    termini = {":MET:15": "N"}
    chorizo = LinkedRDKitChorizo(pdb_string, termini=termini)
    assert len(chorizo.residues) == 3
    assert len(chorizo.getIgnoredResidues()) == 0

    expected_suggested_mutations = {}
    assert chorizo.suggested_mutations == expected_suggested_mutations
    
    expected_residue_data = {
        ':MET:15': ChorizoResidue(':MET:15', 'ATOM    220  N   MET    15      14.163  15.881  16.252  1.00  0.00           N  \nATOM    221  H   MET    15      14.428  15.654  17.200  1.00  0.00           H  \nATOM    222  CA  MET    15      13.351  17.068  16.029  1.00  0.00           C  \nATOM    223  HA  MET    15      12.366  16.779  15.662  1.00  0.00           H  \nATOM    224  CB  MET    15      13.177  17.850  17.335  1.00  0.00           C  \nATOM    225  HB2 MET    15      14.168  18.063  17.735  1.00  0.00           H  \nATOM    226  HB3 MET    15      12.671  18.787  17.102  1.00  0.00           H  \nATOM    227  CG  MET    15      12.367  17.079  18.369  1.00  0.00           C  \nATOM    228  HG2 MET    15      11.362  16.943  17.969  1.00  0.00           H  \nATOM    229  HG3 MET    15      12.839  16.106  18.503  1.00  0.00           H  \nATOM    230  SD  MET    15      12.262  17.912  19.967  1.00  0.00           S  \nATOM    231  CE  MET    15      11.025  19.105  19.625  1.00  0.00           C  \nATOM    232  HE1 MET    15      10.104  18.601  19.332  1.00  0.00           H  \nATOM    233  HE2 MET    15      10.842  19.705  20.516  1.00  0.00           H  \nATOM    234  HE3 MET    15      11.358  19.752  18.814  1.00  0.00           H  \nATOM    235  C   MET    15      13.900  17.930  14.890  1.00  0.00           C  \nATOM    236  O   MET    15      13.137  18.390  14.061  1.00  0.00           O  \n', None, ':SER:16'),
        ':SER:16': ChorizoResidue(':SER:16', 'ATOM    237  N   SER    16      15.208  18.136  14.839  1.00  0.00           N  \nATOM    238  H   SER    16      15.818  17.724  15.530  1.00  0.00           H  \nATOM    239  CA  SER    16      15.812  18.990  13.787  1.00  0.00           C  \nATOM    240  HA  SER    16      15.275  19.938  13.751  1.00  0.00           H  \nATOM    241  CB  SER    16      17.291  19.276  14.049  1.00  0.00           C  \nATOM    242  HB2 SER    16      17.812  18.327  14.172  1.00  0.00           H  \nATOM    243  HB3 SER    16      17.701  19.806  13.189  1.00  0.00           H  \nATOM    244  OG  SER    16      17.464  20.070  15.224  1.00  0.00           O  \nATOM    245  HG  SER    16      18.399  20.234  15.364  1.00  0.00           H  \nATOM    246  C   SER    16      15.682  18.311  12.430  1.00  0.00           C  \nATOM    247  O   SER    16      15.436  18.983  11.441  1.00  0.00           O  \n',':MET:15', ':LEU:17'),
        ':LEU:17': ChorizoResidue(':LEU:17', 'ATOM    248  N   LEU    17      15.835  16.986  12.389  1.00  0.00           N  \nATOM    249  H   LEU    17      16.162  16.471  13.194  1.00  0.00           H  \nATOM    250  CA  LEU    17      15.588  16.249  11.147  1.00  0.00           C  \nATOM    251  HA  LEU    17      16.256  16.610  10.366  1.00  0.00           H  \nATOM    252  CB  LEU    17      15.838  14.741  11.351  1.00  0.00           C  \nATOM    253  HB2 LEU    17      16.705  14.733  12.011  1.00  0.00           H  \nATOM    254  HB3 LEU    17      14.953  14.441  11.912  1.00  0.00           H  \nATOM    255  CG  LEU    17      16.069  13.752  10.185  1.00  0.00           C  \nATOM    256  HG  LEU    17      15.182  13.764   9.551  1.00  0.00           H  \nATOM    257  CD1 LEU    17      17.278  14.110   9.362  1.00  0.00           C  \nATOM    258 HD11 LEU    17      18.165  14.098   9.995  1.00  0.00           H  \nATOM    259 HD12 LEU    17      17.397  13.386   8.556  1.00  0.00           H  \nATOM    260 HD13 LEU    17      17.148  15.106   8.938  1.00  0.00           H  \nATOM    261  CD2 LEU    17      16.290  12.344  10.717  1.00  0.00           C  \nATOM    262 HD21 LEU    17      15.413  12.027  11.282  1.00  0.00           H  \nATOM    263 HD22 LEU    17      16.451  11.661   9.883  1.00  0.00           H  \nATOM    264 HD23 LEU    17      17.164  12.334  11.368  1.00  0.00           H  \nATOM    265  C   LEU    17      14.162  16.490  10.648  1.00  0.00           C  \nATOM    266  O   LEU    17      13.948  16.756   9.478  1.00  0.00           O  \n',':SER:16', None),
        }
    for residue_id in chorizo.residues:
        residue_object = chorizo.residues[residue_id]
        expected_object = expected_residue_data[residue_id]
        assert residue_object.residue_id == expected_object.residue_id
        assert residue_object.pdb_text == expected_object.pdb_text
        assert residue_object.previous_id == expected_object.previous_id
        assert residue_object.next_id == expected_object.next_id
        assert residue_object.rdkit_mol != None

    pdbqt_strings = PDBQTWriterLegacy.write_string_from_linked_rdkit_chorizo(chorizo)
    rigid_part, movable_part = pdbqt_strings
    rigid_part = "".join(rigid_part.splitlines()) # remove newline chars because Windows/Unix differ
    assert len(rigid_part) == 2212
    assert len(movable_part) == 0

    expected_termini = {':MET:15': 'N'}
    assert chorizo.termini == expected_termini

    met15_padded, is_actual_res, atom_index_map = chorizo.get_padded_mol(":MET:15")
    met15_resmol = chorizo.residues[":MET:15"].rdkit_mol

    expected_is_actual_res = [False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False]
    expected_atom_index_map = {2: 0, 3: 1, 4: 2, 5: 4, 6: 3, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 5, 19: 6, 20: 7}
    assert met15_resmol
    assert met15_padded
    assert is_actual_res == expected_is_actual_res
    assert atom_index_map == expected_atom_index_map

def test_AHHY_mutate_residues():
    # We want both histidines to be "HIP" and to delete the tyrosine
    mutations = {
        "A:HIS:2": "A:HIP:2",
        "A:HIS:3": "A:HIP:3",
        }
    delete_residues = ("A:TYR:4",)
    f = open(ahhy_example, 'r')
    pdb_string = f.read()
    chorizo = LinkedRDKitChorizo(
        pdb_string,
        deleted_residues=delete_residues,
        mutate_res_dict=mutations)
    assert len(chorizo.residues) == 4
    assert len(chorizo.getIgnoredResidues()) == 0

    expected_suggested_mutations = {}
    assert chorizo.suggested_mutations == expected_suggested_mutations

    expected_residue_data = {
        'A:ALA:1': ChorizoResidue('A:ALA:1', 'ATOM      1  N   ALA A   1       6.061   2.529  -3.691  1.00  0.00           N  \nATOM      2  CA  ALA A   1       5.518   2.870  -2.403  1.00  0.00           C  \nATOM      3  C   ALA A   1       4.995   1.645  -1.690  1.00  0.00           C  \nATOM      4  O   ALA A   1       5.294   0.515  -2.156  1.00  0.00           O  \nATOM      5  CB  ALA A   1       4.421   3.891  -2.559  1.00  0.00           C  \n', None, 'A:HIP:2'),
        'A:HIP:2': ChorizoResidue('A:HIP:2', 'ATOM      6  N   HIS A   2       4.201   1.774  -0.543  1.00  0.00           N  \nATOM      7  CA  HIS A   2       3.690   0.569   0.155  1.00  0.00           C  \nATOM      8  C   HIS A   2       2.368   0.239  -0.349  1.00  0.00           C  \nATOM      9  O   HIS A   2       1.827   0.959  -1.278  1.00  0.00           O  \nATOM     10  CB  HIS A   2       3.958   0.658   1.602  1.00  0.00           C  \nATOM     11  CG  HIS A   2       3.518  -0.435   2.481  1.00  0.00           C  \nATOM     12  ND1 HIS A   2       4.232  -1.588   2.706  1.00  0.00           N  \nATOM     13  CD2 HIS A   2       2.407  -0.562   3.290  1.00  0.00           C  \nATOM     14  CE1 HIS A   2       3.592  -2.347   3.583  1.00  0.00           C  \nATOM     15  NE2 HIS A   2       2.438  -1.715   3.961  1.00  0.00           N  \nATOM     16  H13 HIS A   2       5.120  -1.737   2.040  1.00  0.00           H  \n', 'A:ALA:1', 'A:HIP:3'),
        'A:HIP:3': ChorizoResidue('A:HIP:3', 'ATOM     17  N   HIS A   3       1.527  -0.823   0.040  1.00  0.00           N  \nATOM     18  CA  HIS A   3       0.243  -1.075  -0.553  1.00  0.00           C  \nATOM     19  C   HIS A   3      -0.832  -0.126  -0.071  1.00  0.00           C  \nATOM     20  O   HIS A   3      -0.560   1.096   0.147  1.00  0.00           O  \nATOM     21  CB  HIS A   3      -0.214  -2.454  -0.694  1.00  0.00           C  \nATOM     22  CG  HIS A   3       0.654  -3.363  -1.491  1.00  0.00           C  \nATOM     23  ND1 HIS A   3       1.819  -2.985  -2.046  1.00  0.00           N  \nATOM     24  CD2 HIS A   3       0.457  -4.684  -1.802  1.00  0.00           C  \nATOM     25  CE1 HIS A   3       2.360  -4.051  -2.700  1.00  0.00           C  \nATOM     26  NE2 HIS A   3       1.515  -5.068  -2.538  1.00  0.00           N  \nATOM     27  H20 HIS A   3       1.651  -6.016  -2.918  1.00  0.00           H  \n', 'A:HIP:2', 'A:TYR:4'),
        'A:TYR:4': ChorizoResidue('A:TYR:4', 'ATOM     28  N   TYR A   4      -2.156  -0.543   0.154  1.00  0.00           N  \nATOM     29  CA  TYR A   4      -3.237   0.354   0.596  1.00  0.00           C  \nATOM     30  C   TYR A   4      -3.373   0.217   2.071  1.00  0.00           C  \nATOM     31  O   TYR A   4      -2.656  -0.595   2.677  1.00  0.00           O  \nATOM     32  CB  TYR A   4      -4.460  -0.123  -0.108  1.00  0.00           C  \nATOM     33  CG  TYR A   4      -5.699   0.602   0.156  1.00  0.00           C  \nATOM     34  CD1 TYR A   4      -6.089   1.698  -0.613  1.00  0.00           C  \nATOM     35  CD2 TYR A   4      -6.492   0.168   1.200  1.00  0.00           C  \nATOM     36  CE1 TYR A   4      -7.276   2.306  -0.282  1.00  0.00           C  \nATOM     37  CE2 TYR A   4      -7.679   0.783   1.528  1.00  0.00           C  \nATOM     38  CZ  TYR A   4      -8.060   1.866   0.764  1.00  0.00           C  \nATOM     39  OH  TYR A   4      -9.262   2.477   1.103  1.00  0.00           O  \nATOM     40  OXT TYR A   4      -4.293   0.998   2.728  1.00  0.00           O  \nATOM     41  H29 TYR A   4      -9.644   3.279   0.612  1.00  0.00           H  \n', 'A:HIP:3', None),
        }
    
    for residue_id in chorizo.residues:
        residue_object = chorizo.residues[residue_id]
        expected_object = expected_residue_data[residue_id]
        assert residue_object.residue_id == expected_object.residue_id
        assert residue_object.pdb_text == expected_object.pdb_text
        assert residue_object.previous_id == expected_object.previous_id
        assert residue_object.next_id == expected_object.next_id
        if not residue_object.user_deleted:
            assert residue_object.rdkit_mol != None


    # We want to check that the mutation has not changed the order of the residues
    assert list(chorizo.residues.keys()) == list(expected_residue_data.keys())

    pdbqt_strings = PDBQTWriterLegacy.write_string_from_linked_rdkit_chorizo(chorizo)
    rigid_part, movable_part = pdbqt_strings
    rigid_part = "".join(rigid_part.splitlines()) # remove newline chars because Windows/Unix differ

    assert len(rigid_part) == 2528
    assert len(movable_part) == 0

    assert chorizo.residues['A:TYR:4'].user_deleted

def test_residue_missing_atoms():
    # checks that we get a Runtime Error when we don't provide the argument
    f = open(just_one_ALA_missing, 'r')
    pdb_string = f.read()
    with pytest.raises(RuntimeError):
        chorizo = LinkedRDKitChorizo(pdb_string)

    chorizo = LinkedRDKitChorizo(pdb_string, allow_bad_res=True)
    assert len(chorizo.residues) == 1
    assert len(chorizo.getIgnoredResidues()) == 1

    expected_removed_residues = ['A:ALA:1']
    assert list(chorizo.getIgnoredResidues().keys()) == expected_removed_residues
    expected_suggested_mutations = {}
    assert chorizo.suggested_mutations == expected_suggested_mutations

    #TODO: This was based off of what we were getting in beoids, we may want something different here actually
    expected_residue_data = {
        'A:ALA:1': ChorizoResidue('A:ALA:1', 'ATOM      1  N   ALA A   1       6.061   2.529  -3.691  1.00  0.00           N  \nATOM      2  CA  ALA A   1       5.518   2.870  -2.403  1.00  0.00           C  \nATOM      3  C   ALA A   1       4.995   1.645  -1.690  1.00  0.00           C  \nATOM      4  O   ALA A   1       5.294   0.515  -2.156  1.00  0.00           O  \n', None, None)
    }
