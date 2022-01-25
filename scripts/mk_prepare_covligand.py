
import sys
import os
import argparse

from rdkit import Chem
import prody
from meeko import MoleculePreparation, CovalentBuilder



def parse_ligand(fname):
    # parse the ligand
    lig_name, lig_ext = os.path.splitext(fname)
    lig_ext = lig_ext[1:].lower()
    if not lig_ext in lig_supported:
        print("Error> [%s] format is not supported for ligand." % lig_ext)
        sys.exit(1)

    if not lig_ext == 'sdf':
        lig_mol = lig_supported[lig_ext](fname)
        if lig_ext == 'mol2':
            print("Warning: Mol2 format is not supported as output format, so SDF will be used.")
            lig_ext = 'sdf'
    else:
        # bad hack to read the first SDF file (no multi supported!)
        with Chem.SDMolSupplier(fname) as suppl:
            lig_mol = [x for x in suppl if x is not None][0]
    if not lig_mol is None:
        return lig_mol, lig_name, lig_ext
    print("Error: the ligand file [%s] cannot be read." % fname)
    sys.exit(1)

def parse_receptor(fname):
    """ Read the receptor molecule if supported """
    # parse receptor
    rec_n, rec_e = os.path.splitext(fname)
    rec_e = rec_e[1:].lower()
    if not rec_e in rec_supported:
        print("Error> [%s] format is not supported for receptor.")
        sys.exit(1)
    rec_mol = rec_supported[rec_e](fname)
    return rec_mol


def write_cov_flex_res_pdbqt(fname, string, residue, at_names, legacy=False):
    """ write the input string into the requested filename, and by default,
        adapt the string to be compatible with the AutoDock4 format.
    """
    res, chain, num = residue
    with open (fname, "w") as fp:
        if legacy:
            # add begin_reVs
            fp.write("BEGIN_RES %s %s %s\n" % (res, chain, num))
            inside = 0
            for line in string.split("\n"):
                # skip empty line at the end (PDBQT writer bug?)
                if line == "":
                    continue
                # skip torsdof
                if line.startswith("TORSDOF"):
                    continue
                # detect when inside ROOT block
                if line.startswith("ATOM"):
                    inside+=1
                    if inside == 1: # :and line.startswith("ATOM"):
                        line = line[:13] + at_names[0] + line[15:]
                    elif inside == 2:
                        # print(Q)
                        line = line[:13] + at_names[1] + line[15:]
                    fp.write("%s\n" % line)
                    continue
                fp.write("%s\n" % line)
            # add end_res
            fp.write("END_RES %s %s %s\n" % (res, chain, num))
        else:
            fp.write(string)
    print("CovalentBuilder.__main__ > saved : %s" %  fname)


def write_molecule(lig_name, lig_ext, cov_lig):
    """ function to write molecules in RDKit supported formats (SDF and mol) """
    out_name = "%s_%s.%s" % (lig_name, cov_lig.label, lig_ext)
    if lig_ext == 'mol':
        Chem.MolToMolFile(cov_lig, out_name)
    elif lig_ext == 'sdf':
        writer = Chem.SDWriter(out_name)
        writer.write(cov_lig.mol)
        writer.close()


description = "Meeko PrepareCovalent: prepare ligands for covalent (tethered) docking"
# TODO expand/fix usage.
usage = "Ligands need to be modified to contain the state following the chemical reaction, and with the target residue side chain alread attached, up to the C-alpha carbon (CA).   By default, the final PDBQT file is written. Glycine is not supported."


#### Defaults are here

# default SMARTS indices for CA and CB
# default_lig_idx = (0,1)
default_lig_idx = "0,1"
default_lig_rec_res_atom_names ="CA,CB" # ('CA','CB')
default_pdbqt_legacy=False
default_write_pdbqt = False

lig_supported = {'mol': Chem.MolFromMolFile,
        'mol2': Chem.MolFromMol2File,
        'sdf': Chem.SDMolSupplier,
        # 'pdb': Chem.MolFromPDBFile
        }

rec_supported = {
        # 'mol2': Chem.MolFromMol2File,
        # 'sdf': Chem.MolFromMolFile,
        'pdb': prody.parsePDB,
        'mmcif':prody.parseMMCIF,
        }

options = {
    '--ligand' : {
        'help': 'ligand fi le in one of the supported formats (%s)' % ", ".join(lig_supported.keys()),
        'action':'store',
        'type':str,
        'required':True,

        },
    '--receptor': {
        'help': 'receptor file in one of the supported formats (%s)' % ", ".join(rec_supported.keys()),
        'action':'store',
        'type':str,
        'required':True,
        },
    '--lig_smarts' : {
        'help': ('SMARTS pattern used for the recognition of the ligand attachment points. The pattern must to '
            'include the two atoms that define the bond used for the alignment (usually CA and CB atoms to '
            'be overlapped with the target sidechain). By default, the first two atoms in the pattern are used, '
            'unless the option \"--ligand_smarts_indices\" is used'),
        'action':'store',
        'type':str,
        'required':True,
        },
    '--lig_smarts_indices' : {
        'help': ('indices of the SMARTS pattern features that identify the atoms used for the alignment '
            '(usually the portions of CA and CB from the target side chain), comma-separated, e.g.: 1,2 or 9,6. By default, '
            'the first two features of the SMARTS pattern are used, unless this option is used. Indices are '
            '0-based [ default: 1,2 ]'),
        'action':'store',
        'metavar': "IDX1,IDX2",
        # 'default': (1,2),
        'default': default_lig_idx,
        'type':str,
        },

    '--rec_residue' : {
        'help': ('residue(s) on the target to be used for generating the alignment(s). Residues can be specified '
            'using the string "chain:res:num", or "chain:res:num:atom_name1,atom_name2", if the default CA and CB atoms '
            'are not being used. At least the "res" term must be included, while if any other temr is '
            'omitted, than all matches found in the target will be used. For example, ":LYS:" will generate '
            ' individual alignments for each lysine found on the target. "A:HIS::ND1,CE1" would match all '
            'histidines in chain A, and use the atoms ND1 and CE1 for the alignment. ** NOTE **  the target '
            'structure must contain the atoms to be used for the alignment (either the default CA and '
            'CB atoms or the user-defined ones).'),
        'metavar': "[CHAIN]:RES:[NUM][:AT_NAME1,AT_NAME2]",
        'action':'store',
        'type':str,
        'required':True,
        },

    '--write_pdbqt' : {
            'help': ('By default, only the aligned structure is saved in the original format; with this option, '
                'the output format is written in PDBQT format'),
            'action':'store_true',
            'default': default_write_pdbqt,
            },

    '--output_name' : {
            'help': ('By default, the output file is generated from the input file and the residues '
                'used for the alignment. If this option is used, the string defined will be used as a '
                'basename for the output file.'),
            'action':'store',
            'default': None,
            'type': str,
            },

    '--write_pdbqt_legacy' : {
        'help': ('By enabling the legacy mode, a fully compliant AutoDock4 PDBQT covalent '
            'flexible residue is written (i.e.: with BEGIN_RES/END_RES '
            'tags, and first two atoms renamed CA and CB). **Note** this '
            'option is not available if atoms other than CA and CB '
            'are specified with the option \"--rec_residue\"'),
        'action':'store_true',
        'default': default_pdbqt_legacy,
        },
}

#######################################################################
# build the parser
parser = argparse.ArgumentParser(description = description, usage=usage)
for opt, info in options.items():
    parser.add_argument(opt, **info)
args = parser.parse_args()
# sys.exit(0)

#######################################################################
# validate ligand indices
try:
    lig_smarts_indices = [int(i) for i in args.lig_smarts_indices.split(",")]
    # test if indices are larger than size of the requested pattern
    smarts_size = Chem.MolFromSmarts(args.lig_smarts).GetNumAtoms()
    if (lig_smarts_indices[0] >= smarts_size) or (lig_smarts_indices[1] >= smarts_size):
        print("Error> At least one of the values specified in \"--lig_smarts_indices\" is larger than the size of the SMARTS pattern (%d)" % (smarts_size))
        sys.exit(1)
except ValueError:
    # print("GOTH ERE")
    print("Error> The atom indices specification not valid. Specify two atoms with a 0-based index. Examples: 0,2 or 9,6")
    sys.exit(1)

#######################################################################
# validate residue specification
try:
    chain, res, num, *atom_names = args.rec_residue.split(":")
    print("PARSED c:%s r:%s n:%s AAA:%s" % (chain, res, num, atom_names))
    try:
        if not num == "":
            num = int(num)
    except ValueError:
        print("Error> The residue specification is not valid. Use CHAIN:RES:NUM or CHAIN:RES:NUM:ATOM1,ATOM2")
        sys.exit(1)
    # parse atom names
    #  TODO clean all specifications
    if atom_names == []:
        # print("USING DEFAULT aTOM NAMES")
        atom_names= default_lig_rec_res_atom_names
    else:
        atom_names=atom_names[0]
    # trigger error if comma not found
    a1, a2 = atom_names.split(",")
    # check that the legacy_format flag is not used
    if args.write_pdbqt_legacy and (not a1 == 'CA' or not a2 == 'CB'):
        print("Error> PDBQT legacy mode supports only using CA,CB atoms for the alignment")
        sys.exit(1)
    residue =  (chain, res, num, a1, a2)
except ValueError:
    print("Error> The residue specification is not valid. Use CHAIN:RES:NUM or CHAIN:RES:NUM:ATOM1,ATOM2")
    sys.exit(1)


#######################################################################
# parse pdbqt options
# by default, legacy format is disabled
write_pdbqt = args.write_pdbqt
legacy_pdbqt = args.write_pdbqt_legacy
if legacy_pdbqt:
    write_pdbqt = True

#######################################################################
# start parsing the structures
rec_mol = parse_receptor(args.receptor)
lig_mol, lig_name, lig_ext = parse_ligand(args.ligand)
lig_smarts = args.lig_smarts

#######################################################################
# output name
if not args.output_name is None:
    lig_name = args.output_name

#######################################################################
# initialize molecule processor
preparator = MoleculePreparation()

#######################################################################
# initialize the covalent builder
try:
    covbuild = CovalentBuilder(rec_mol)
    covbuild.find_residues(residue)
except:
    print("[ Failure ]")
    sys.exit(1)
# start covalent building
for cov_lig in covbuild.process(lig_mol, lig_smarts, lig_smarts_indices, first_only=True):
    # manage the output
    if write_pdbqt:
        # prepare the flexible molecule
        preparator.prepare(cov_lig.mol, root_atom_index=cov_lig.indices[1])
        #
        pdbqt_string = preparator.write_pdbqt_string()
        pdbqt_fname = "%s_%s.pdbqt" % (lig_name, cov_lig.label)
        write_cov_flex_res_pdbqt(pdbqt_fname, pdbqt_string, cov_lig.res_id,
                cov_lig.at_names, legacy=legacy_pdbqt)
    else:
        # write the aligned ligand in the original format.
        write_molecule(lig_name, lig_ext, cov_lig)
