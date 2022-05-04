from meeko import MoleculePreparation
from rdkit import Chem
import pathlib

num_cycle_breaks = {
    "adamantane": None,
    "adamantane-stretched": None,
    "cubane": None,
    "oxabicyclo-2-2-2-octane": None,
    "tricycle-6-5-5": None,
    "tricycle-6-5-6": None,
    "tricycle-6-5-6_B": None,
    "tricycle-6-7-7": None,
}
workdir = pathlib.Path(__file__)
filenames = {name: str(workdir.parents[0] / "small_cycle_data" / ("%s.sdf" % name)) for name in num_cycle_breaks}
mk_prep = MoleculePreparation() #keep_equivalent_rings=True)

def run(molname):
    filename = filenames[molname]
    mol = Chem.MolFromMolFile(filename, removeHs=False)
    mk_prep.prepare(mol)
    setup = mk_prep.setup
    print(molname)
    for ring in setup.rings:
        print("len=%d" % len(ring), ring)
    print()

def test_all():
    for molname in num_cycle_breaks:
        run(molname)
