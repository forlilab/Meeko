from meeko import LinkedRDKitChorizo, ChorizoResidue, ResidueAdditionalConnection, PDBQTWriterLegacy, MoleculePreparation

# Example Files (should be moved to tests eventually)
ahhy_example = "example\chorizo\AHHY.pdb"
just_one_ALA_missing = "example\chorizo\just-one-ALA-missing-CB.pdb"
just_one_ALA = "example\chorizo\just-one-ALA.pdb"
just_three_residues = "example\chorizo\just-three-residues.pdb"

def test_AHHY_all_static_residues():
    chorizo = LinkedRDKitChorizo(ahhy_example)
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
        assert len(rigid_part) == 3564
        assert len(movable_part) == 0

def test_AHHY_flexible_residues():
    chorizo = LinkedRDKitChorizo(ahhy_example)
    mk_prep = MoleculePreparation()

    residue_id = "A:HIS:2"
    chorizo.res_to_molsetup(residue_id, mk_prep)

    pdbqt_strings = PDBQTWriterLegacy.write_string_from_linked_rdkit_chorizo(chorizo)
    rigid_part, movable_part = pdbqt_strings
    assert 1 == 1

def test_just_three_padded_mol():
    termini = {":MET:15": "N"}
    chorizo = LinkedRDKitChorizo(just_three_residues, termini=termini)

    assert len(chorizo.residues) == 3
    assert len(chorizo.getIgnoredResidues()) == 0

def test_AHHY_mutate_residues():
    # This is a placeholder for future test cases
    assert 1 == 1

def test_residue_missing_atoms():
    # This is a placeholder for future test cases
    assert 1 == 1

#class LinkedRDKitChorizoCreationIntegrationTest()