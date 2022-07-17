import xml.etree.ElementTree as ET

from rdkit import Chem


def validate_key(key, keyword, terms):
    """validate Proper torsion from OFFXML"""

    if not key.startswith(keyword):
        return False
    length = len(keyword)
    if not key[length:].isdigit():  # e.g. key="phase1", keyword="phase", length=5
        return False
    series_index = int(key[length:])
    terms.setdefault(series_index, set())
    terms[series_index].add(keyword)
    return True


def smirks_to_smarts(smirks):
    """returns:
    - SMARTS - same as smirks, but without labels, i.e. [C:1] -> [C]
    - labels - mapping of labels to atom indices, i.e. {1: 0}

    not needed for rdkit, but for openbabel which lacks support for labels
    """

    n_open = 0
    n_close = 0
    depth_recursion = 0  # recursive smarts, e.g. [C$([N])]
    bracket_depth = 0
    last_recursion_bracket_depth = [None]  # bracket depth when '$(' was found
    labels = (
        {}
    )  # , keys: integer label in smirks, values: index of atom in smarts molecule
    chars_to_delete = []
    for i, char in enumerate(smirks):
        if char == "[" and depth_recursion == 0:
            n_open += 1
        elif char == "]" and depth_recursion == 0:
            n_close += 1
            string = ""  # to store chars between : and ]
            candidate_chars_to_delete = []
            for j in range(i):
                k = i - j - 1  # chars up to 'i', in reverse
                c = smirks[k]
                candidate_chars_to_delete.append(k)  # includes ":"
                if c == ":":
                    break
                string = c + string
            if string.isdigit():
                label_id = int(string)
                if label_id in labels:
                    msg = "\ncan't convert smirks with repeated label: %d" % label_id
                    msg += "\nsmirks: %s" % smirks
                    raise ValueError(msg)
                labels[label_id] = n_open - 1
                chars_to_delete.extend(candidate_chars_to_delete)
        elif char == "$" and smirks[i + 1] == "(":
            last_recursion_bracket_depth.append(bracket_depth + 1)
            depth_recursion += 1
        elif char == "(":
            bracket_depth += 1
        elif (
            char == ")"
            and depth_recursion > 0
            and last_recursion_bracket_depth[-1] == bracket_depth
        ):
            depth_recursion -= 1
            last_recursion_bracket_depth.pop(-1)
            bracket_depth -= 1
        elif char == ")":
            bracket_depth -= 1

    if n_open != n_close:
        raise RuntimeError("ooops different number of [ and ]")
    mol = Chem.MolFromSmarts(smirks)
    n_atoms = mol.GetNumAtoms()
    if n_open != n_atoms:
        print(n_open, n_atoms)
        raise RuntimeError("need all atoms to be encapsulated by [ ]")

    assert len(set(chars_to_delete)) == len(chars_to_delete)

    smarts = list(smirks)
    for i in sorted(chars_to_delete)[::-1]:
        smarts.pop(i)
    smarts = "".join(smarts)
    return smarts, labels


def make_dihedral_entry(attrib_dict_from_xml):
    """convert 'Proper' dict from OFFXML to autodockdev dict"""

    smirks = attrib_dict_from_xml["smirks"]
    smarts, labels = smirks_to_smarts(smirks)
    assert len(labels) == 4
    assert 1 in labels and 2 in labels and 3 in labels and 4 in labels
    dihedral_entry = {
        "smarts": smarts,
        "IDX": [labels[index] + 1 for index in (1, 2, 3, 4)],
    }
    terms = {}  # keywords found for each term of the fourier series
    expected_keywords = set(["k", "phase", "periodicity", "idivf"])
    for key, value in attrib_dict_from_xml.items():
        if key == "smirks":
            continue
        elif key == "id":
            dihedral_entry["id"] = value
        else:
            for keyword in expected_keywords:
                is_valid_key = validate_key(key, keyword, terms)
                if is_valid_key:
                    break
            if not is_valid_key:
                msg = "\nGot unexpected key: %s" % key
                msg += "\nOffending 'Proper' entry: %s" % attrib_dict_from_xml
                raise ValueError(msg)
            elif keyword == "k":
                dihedral_entry[key] = float(
                    value.replace("* mole**-1 * kilocalorie", "")
                )
            elif keyword == "phase":
                dihedral_entry[key] = float(value.replace("* degree", ""))
            elif keyword == "periodicity":
                dihedral_entry[key] = int(value)
            elif keyword == "idivf":
                dihedral_entry[key] = float(value)
    # check that all terms in the fourier series have all expected keywords
    assert len(terms) >= 1
    for series_index in terms:
        if terms[series_index] != expected_keywords:
            msg = "\nmismatch between input keywords: %s\n" % terms[series_index]
            msg += "           and expected keywords: %s\n" % expected_keywords
            msg += "               with series_index: %d" % series_index
            raise ValueError(msg)
    return dihedral_entry

def make_vdw_entry(attrib_dict_from_xml):
    """convert 'Atom' dict from OFFXML to autodockdev dict"""

    smirks = attrib_dict_from_xml["smirks"]
    smarts, labels = smirks_to_smarts(smirks)
    assert len(labels) == 1
    assert 1 in labels
    vdw_entry = {
        "smarts": smarts,
        "IDX": [labels[1] + 1],
        "id": attrib_dict_from_xml["id"],
    }
    assert ("epsilon" in attrib_dict_from_xml) and (
        ("rmin_half" in attrib_dict_from_xml) ^ ("sigma" in attrib_dict_from_xml)
    )
    for key, value in attrib_dict_from_xml.items():
        if key in ["id", "smirks"]:
            continue
        elif key == "epsilon":
            vdw_entry["epsilon"] = float(value.replace("* mole**-1 * kilocalorie", ""))
        elif key == "rmin_half":
            vdw_entry["rmin_half"] = float(value.replace("* angstrom", ""))
        elif key == "sigma":
            vdw_entry["rmin_half"] = (
                float(value.replace("* angstrom", "")) * (4 ** (1.0 / 12)) * 0.5
            )
        else:
            msg = "\nGot unexpected key: %s" % key
            msg += "\nOffending 'Atom' entry: %s" % attrib_dict_from_xml
            raise ValueError(msg)
    return vdw_entry

def assign_atypes(vdw_list, use_openff_id=True):
    mini_periodic_table = {
         1:'H',   2:'He',
         3:'Li',  4:'Be',  5:'B',   6:'C',   7:'N',   8:'O',   9:'F',  10:'Ne',
        11:'Na', 12:'Mg', 13:'Al', 14:'Si', 15:'P',  16:'S',  17:'Cl', 18:'Ar',
        19:'K',  20:'Ca', 21:'Sc', 22:'Ti', 23:'V',  24:'Cr', 25:'Mn', 26:'Fe', 27:'Co', 28:'Ni', 29:'Cu', 30:'Zn',
        31:'Ga', 32:'Ge', 33:'As', 34:'Se', 35:'Br', 36:'Kr',
        37:'Rb', 38:'Sr', 39:'Y',  40:'Zr', 41:'Nb', 42:'Mo', 43:'Tc', 44:'Ru', 45:'Rh', 46:'Pd', 47:'Ag', 48:'Cd',
        49:'In', 50:'Sn', 51:'Sb', 52:'Te', 53:'I',  54:'Xe',
        55:'Cs', 56:'Ba',
        57:'La', 58:'Ce', 59:'Pr', 60:'Nd', 61:'Pm', 62:'Sm', 63:'Eu', 64:'Gd', 65:'Tb', 66:'Dy', 67:'Ho', 68:'Er', 69:'Tm', 70:'Yb',
        71:'Lu', 72:'Hf', 73:'Ta', 74:'W',  75:'Re', 76:'Os', 77:'Ir', 78:'Pt', 79:'Au', 80:'Hg',
        81:'Tl', 82:'Pb', 83:'Bi', 84:'Po', 85:'At', 86:'Rn',
        87:'Fr', 88:'Ra'
        }

    number_by_element = {}
    used_numbers = {}
    atomic_numbers = []

    for v in vdw_list:
        mol = Chem.MolFromSmarts(v["smarts"])
        atom = mol.GetAtomWithIdx(v["IDX"][0] - 1) # consider only the first if multiple IDX
        element = mini_periodic_table[atom.GetAtomicNum()]
        atomic_numbers.append(atom.GetAtomicNum())
        used_numbers.setdefault(element, set())

        if use_openff_id:
            # use id ("n1", "n2", "n3") -> [H1, H2, C3]
            off_id = v["id"]
            if off_id[0] == 'n' and off_id[1:].isdigit():
                n = int(off_id[1:])
            else:
                n = len(vdw_list) + 1
            while n in used_numbers[element]:
                n += 1
            used_numbers[element].add(n)
        else:
            # each element starts from 1 -> [H1, H2, C1]
            number_by_element.setdefault(element, 0)
            number_by_element[element] += 1
            n = number_by_element[element]

        atype = "%s%d" % (element, n)
        v["atype"] = atype
    return atomic_numbers # needed to get atomic mass aftwerwards

def make_vdw_by_atype(vdw_list, atomic_numbers):
    bytype = {}
    index = 0
    for v in vdw_list:
        atype = v["atype"]
        param = {
            "rmin_half": v["rmin_half"],
            "epsilon": v["epsilon"],
            "atomic_number": atomic_numbers[index],
        }
        if atype in bytype:
            assert bytype[atype] == param
        else:
            bytype[atype] = param
        index += 1
    return bytype

def parse_offxml(offxml_filename):
    """
        Convert OpenFF XML entries to autodockdev dictionaries
    """

    root = ET.parse(offxml_filename).getroot()

    torsions = root.findall("ProperTorsions")
    vdw = root.findall("vdW")
    assert len(torsions) == 1
    assert len(vdw) == 1
    torsions = torsions[0]
    vdw = vdw[0]

    vdw_list = []
    for child in vdw:
        if child.tag != "Atom":
            print(" SKIPPING: %s" % child.tag)
            continue
        v = make_vdw_entry(child.attrib)
        vdw_list.append(v)
    atomic_numbers = assign_atypes(vdw_list)
    vdw_by_atype = make_vdw_by_atype(vdw_list, atomic_numbers)

    dihedral_list = []
    for child in torsions:
        if child.tag != "Proper":
            print(" SKIPPING: %s" % child.tag)
            continue
        d = make_dihedral_entry(child.attrib)
        dihedral_list.append(d)

    return vdw_list, dihedral_list, vdw_by_atype
