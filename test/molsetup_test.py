from meeko import MoleculeSetup

import pathlib
import pytest

import meeko


# region Fixtures
# We may want to find a different place for these as tests get more complex structure and revisions.
@pytest.fixture
def empty_molecule_setup():
    molsetup = MoleculeSetup()
    yield molsetup


# endregion


class TestMoleculeSetupInit:
    def test_add_one_atom(self):
        pass

    def test_add_one_pseudo_atom(self):
        pass

    def test_add_multiple_atoms(self):
        pass

    def test_add_bonds(self):
        pass
