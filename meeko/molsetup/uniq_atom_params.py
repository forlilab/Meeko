"""UniqAtomParams: row/column table of atom parameters."""


class UniqAtomParams:
    """A row/column store of atom parameter sets.

    Attributes
    ----------
    params: list[list]
        rows (one row per unique parameter set)
    param_names: list[str]
        column names
    """

    def __init__(self):
        self.params = []
        self.param_names = []

    @classmethod
    def from_dict(cls, dictionary):
        uap = cls()
        uap.params = [row.copy() for row in dictionary["params"]]
        uap.param_names = dictionary["param_names"].copy()
        return uap

    def get_indices_from_atom_params(self, atom_params):
        nr_items = set([len(values) for key, values in atom_params.items()])
        if len(nr_items) != 1:
            raise RuntimeError(
                f"all lists in atom_params must have same length, got {nr_items}"
            )
        if set(atom_params) != set(self.param_names):
            msg = "parameter names in atom_params differ from internal ones\n"
            msg += f"  - in atom_params: {set(atom_params)}"
            msg += f"  - internal: {set(self.param_names)}"
            raise RuntimeError(msg)
        nr_items = nr_items.pop()
        param_idxs = []
        for i in range(nr_items):
            row = [atom_params[key][i] for key in self.param_names]
            param_index = None
            for j, existing_row in enumerate(self.params):
                if row == existing_row:
                    param_index = j
                    break
            param_idxs.append(param_index)
        return param_idxs

    def add_parameter(self, new_param_dict):
        new_param_dict = {k: v for k, v in new_param_dict.items() if v is not None}
        incoming_keys = set(new_param_dict.keys())
        existing_keys = set(self.param_names)
        new_keys = incoming_keys.difference(existing_keys)
        for new_key in new_keys:
            self.param_names.append(new_key)
            for row in self.params:
                row.append(None)

        new_row = []
        for key in self.param_names:
            new_row.append(new_param_dict.get(key, None))

        if len(new_keys) == 0:
            for index, row in enumerate(self.params):
                if row == new_row:
                    return index

        new_row_index = len(self.params)
        self.params.append(new_row)
        return new_row_index

    def add_molsetup(
        self,
        molsetup,
        atom_params=None,
        add_atomic_nr=False,
        add_atom_type=False,
        remove_params=(),
    ):
        if "charge" in molsetup.atom_params or "atom_type" in molsetup.atom_params:
            msg = '"charge" and "atom_type" found in molsetup.atom_params'
            msg += " but are hard-coded to store molsetup.charge and"
            msg += " molsetup.atom_type in the internal data structure"
            raise RuntimeError(msg)
        if atom_params is None:
            atom_params = molsetup.atom_params
        param_idxs = []
        for atom in molsetup.atoms:
            if atom.is_ignore:
                param_idx = None
            else:
                p = {
                    k: v[atom.index]
                    for (k, v) in molsetup.atom_params.items()
                    if k not in remove_params
                }
                if add_atomic_nr:
                    if "atomic_nr" in p:
                        raise RuntimeError(
                            "trying to add atomic_nr but it's already in atom_params"
                        )
                    p["atomic_nr"] = atom.atomic_num
                if add_atom_type:
                    if "atom_type" in p:
                        raise RuntimeError(
                            "trying to add atom_type but it's already in atom_params"
                        )
                    p["atom_type"] = atom.atom_type
                param_idx = self.add_parameter(p)
            param_idxs.append(param_idx)
        return param_idxs
