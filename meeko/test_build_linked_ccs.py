from meeko.chemtempgen import *
basename = 'MLZ'
cc_list = build_linked_CCs(basename)
for cc in cc_list:
    fetch_template_dict = json.loads(export_chem_templates_to_json([cc]))['residue_templates'][cc.resname]