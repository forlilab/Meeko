from meeko import MoleculePreparation
from rdkit import Chem
import pathlib

min_rings_dict = {
    "adamantane": 4,
    "adamantane-stretched": 4,
    "cubane": 6,
    "oxabicyclo-2-2-2-octane": 3,
    "tricycle-6-5-5": 3,
    "tricycle-6-5-6": 3,
    "tricycle-6-5-6_B": 3,
    "tricycle-6-7-7": 2,
    "tricycle-5-6-7": 2,
    "break-equiv": 4,
    "break-equiv-small": 4,
    "break-equiv-b": 4,
    "break-equiv-a": 4,
    "break-equiv-2": 4,
    "adamantanish-5-5-6": 2,
}
workdir = pathlib.Path(__file__)
filenames = {name: str(workdir.parents[0] / "small_cycle_data" / ("%s.sdf" % name)) for name in min_rings_dict}
mk_prep = MoleculePreparation() #keep_equivalent_rings=True)

def run(molname, min_rings):
    filename = filenames[molname]
    mol = Chem.MolFromMolFile(filename, removeHs=False)
    setups = mk_prep.prepare(mol)
    assert(len(setups) == 1)
    setup = setups[0]
    print("\n%s" % molname)
    found_rings = 0
    for ring in setup.rings:
        print("len=%d, atoms:" % (len(ring)), ring)
        found_rings += 1
    print("Got %d, needed %d" % (found_rings, min_rings))
    assert(found_rings >= min_rings)
    

def test_all():
    for molname in min_rings_dict:
        min_rings = min_rings_dict[molname]
        run(molname, min_rings)
