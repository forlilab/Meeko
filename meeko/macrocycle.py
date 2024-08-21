#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko macrocycle builder
#

from collections import defaultdict
from operator import itemgetter

from .molsetup import Bond

# region
DEFAULT_MIN_RING_SIZE = 7
DEFAULT_MAX_RING_SIZE = 33
DEFAULT_DOUBLE_BOND_PENALTY = 50
DEFAULT_MAX_BREAKS = 4
# endregion


class FlexMacrocycle:
    """
    Attributes
    ----------
    _min_ring_size: int
    _max_ring_size: int
    _double_bond_penalty: float
    max_breaks: int
    setup:
    breakable_rings:
    """

    def __init__(
        self,
        min_ring_size: int = DEFAULT_MIN_RING_SIZE,
        max_ring_size: int = DEFAULT_MAX_RING_SIZE,
        double_bond_penalty: float = DEFAULT_DOUBLE_BOND_PENALTY,
        max_breaks: int = DEFAULT_MAX_BREAKS,
        allow_break_atype_A: bool = False,
    ):
        """
        Initialize macrocycle typer.

        Parameters
        ----------
        min_ring_size: int
            Minimum size of the ring, default is 7.
        max_ring_size: int
            Maximum size of the ring, default is 33.
        double_bond_penalty: float
        max_breaks: int
        allow_break_type_A: bool
            Allow breaking bonds involving atoms typed A, default is False.
        """
        self._min_ring_size = min_ring_size
        self._max_ring_size = max_ring_size
        # accept also double bonds (if nothing better is found)
        self._double_bond_penalty = double_bond_penalty
        self.max_breaks = max_breaks
        self.allow_break_atype_A = allow_break_atype_A

        self.setup = None
        self.breakable_rings = None

    def collect_rings(self, setup):
        """
        Gets non-aromatic rings of desired size and lists bonds that are part of unbreakable rings. Bonds belonging to
        rigid cycles can't be deleted or made rotatable even if they are part of a breakable ring.

        Parameters
        ----------
        setup: RDKitMoleculeSetup

        Returns
        -------
        breakable_rings: list
            A list of breakable ring ids
        bonds_in_rigid_cycles: set
             A set of the bonds in rigid cycles
        """
        breakable_rings = []
        rigid_rings = []
        for ring_id in list(
            setup.rings.keys()
        ):  # ring_id are the atom indices in each ring
            size = len(ring_id)
            if setup.rings[ring_id].is_aromatic:
                rigid_rings.append(ring_id)
            elif size < self._min_ring_size:
                rigid_rings.append(ring_id)
                # do not add rings > _max_ring_size to rigid_rings
                # because bonds in rigid rings will not be breakable
                # and these bonds may also belong to breakable rings
            elif size <= self._max_ring_size:
                breakable_rings.append(ring_id)

        bonds_in_rigid_cycles = set()
        for ring_atom_indices in rigid_rings:
            for bond in setup.get_bonds_in_ring(ring_atom_indices):
                bonds_in_rigid_cycles.add(bond)

        return breakable_rings, bonds_in_rigid_cycles

    def _score_bond(self, bond: tuple[int, int]) -> int:
        """
        Calculates a score for the likeliness that a bond will be broken.

        Parameters
        ----------
        bond: tuple[int, int]
            Input bond to score

        Returns
        -------
        score: int
            A score for the bond.
        """
        bond = Bond.get_bond_id(bond[0], bond[1])
        if not self.setup.bond_info[bond].rotatable:
            return -1
        atom_idx1, atom_idx2 = bond
        for i in (atom_idx1, atom_idx2):
            atype = self.setup.get_atom_type(i)
            is_allowed_A = self.allow_break_atype_A and atype == "A"
            if atype != "C" and not is_allowed_A:
                return -1
        # historically we returned a score <= 100, that was lower for triple
        # bonds, chiral atoms, conjugated bonds, and double bonds. This score
        # gets combined with the graph depth score in flexibility.py that
        # is lower when more consecutive torsions exist in a single "branch"
        # of the torsion tree. Any positive number can be returned here.
        return 100

    def get_breakable_bonds(self, bonds_in_rigid_rings):
        """
        Find breaking points for rings following the guidelines defined in [1].
        The optimal bond has the following properties:
        - does not involve a chiral atom
        - is not double/triple (?)
        - is between two carbons
        (preferably? we can now generate pseudoAtoms on the fly!)
        - is a bond present only in one ring

         [1] Forli, Botta, J. Chem. Inf. Model., 2007, 47 (4)
          DOI: 10.1021/ci700036j

        Parameters
        ----------
        bonds_in_rigid_rings: set[tuple]
            A set of bonds in rigid rings.

        Returns
        -------
        breakable: dict
            A dictionary of mapping breakable bonds to bond scores
        """
        breakable = {}
        for ring_atom_indices in self.breakable_rings:
            for bond in self.setup.get_bonds_in_ring(ring_atom_indices):
                score = self._score_bond(bond)
                if score > 0 and bond not in bonds_in_rigid_rings:
                    breakable[bond] = score
        return breakable

    def search_macrocycle(self, setup, delete_these_bonds=[]):
        """
        Search for macrocycles in the molecule

        Parameters
        ----------
        setup: RDKitMoleculeSetup
        delete_these_bonds: list

        Returns
        -------
        break_combo_data:
        bonds_in_rigid_rings:
        """
        self.setup = setup

        self.breakable_rings, bonds_in_rigid_rings = self.collect_rings(setup)
        if len(delete_these_bonds) == 0:
            breakable_bonds = self.get_breakable_bonds(bonds_in_rigid_rings)
        else:
            breakable_bonds = {}
            for bond in delete_these_bonds:
                bond = Bond.get_bond_id(bond[0], bond[1])
                breakable_bonds[bond] = self._score_bond(bond)
        break_combo_data = self.combinatorial_break_search(breakable_bonds)
        return break_combo_data, bonds_in_rigid_rings

    def combinatorial_break_search(self, breakable_bonds):
        """
        Enumerate all combinations of broken bonds. Once a bond is broken, it will break one or more rings. Subsequent
        bonds will be pulled from intact (unbroken) rings. The number of broken bonds may be variable.
        Returns only combinations of broken bonds that break the maximum number of broken bonds.

        Parameters
        ----------
        breakable_bonds: dict
            A dictionary mapping breakable bonds to the bond score.

        Returns
        -------
        break_combo_data: dict
            A dictionary containing information about possible bond break combinations
        """

        max_breaks = self.max_breaks
        break_combos = self._recursive_break(
            self.breakable_rings, max_breaks, breakable_bonds, set(), []
        )
        break_combos = list(break_combos)  # convert from set
        max_broken_bonds = 0
        output_break_combos = []  # found new max, discard prior data
        output_bond_scores = []
        output_unbroken_rings = []
        for broken_bonds in break_combos:
            n_broken_bonds = len(broken_bonds)
            bond_score = sum([breakable_bonds[bond] for bond in broken_bonds])
            if n_broken_bonds > max_broken_bonds:
                max_broken_bonds = n_broken_bonds
                output_break_combos = []  # found new max, discard prior data
                output_bond_scores = []
                output_unbroken_rings = []
            if n_broken_bonds == max_broken_bonds:
                output_break_combos.append(broken_bonds)
                output_bond_scores.append(bond_score)
                u = self.get_unbroken_rings(self.breakable_rings, broken_bonds)
                output_unbroken_rings.append(u)

        break_combo_data = {
            "bond_break_combos": output_break_combos,
            "bond_break_scores": output_bond_scores,
            "unbroken_rings": output_unbroken_rings,
        }
        return break_combo_data

    def _recursive_break(
        self, rings, max_breaks, breakable_bonds, output=set(), broken_bonds=[]
    ):
        """

        Parameters
        ----------
        rings: list
            List of rings to check
        max_breaks: int
            Maxumimum number of breaks allowed
        breakable_bonds:
        output: set
        broken_bonds:

        Returns
        -------
        output: set

        """
        if max_breaks == 0:
            return output
        unbroken_rings = self.get_unbroken_rings(rings, broken_bonds)
        atoms_in_broken_bonds = [
            atom_idx for bond in broken_bonds for atom_idx in bond
        ]
        for bond in breakable_bonds:
            if bond[0] in atoms_in_broken_bonds or bond[1] in atoms_in_broken_bonds:
                continue  # each atom can be in only one broken bond
            is_bond_in_ring = False
            for ring in unbroken_rings:
                if bond in self.setup.get_bonds_in_ring(ring):
                    is_bond_in_ring = True
                    break
            if is_bond_in_ring:
                current_broken_bonds = [(a, b) for (a, b) in broken_bonds + [bond]]
                num_unbroken_rings = len(
                    self.get_unbroken_rings(rings, current_broken_bonds)
                )
                data_row = tuple(sorted([(a, b) for (a, b) in current_broken_bonds]))
                output.add(data_row)
                if num_unbroken_rings > 0:
                    output = self._recursive_break(
                        rings,
                        max_breaks - 1,
                        breakable_bonds,
                        output,
                        current_broken_bonds,
                    )
        return output

    def get_unbroken_rings(self, rings, broken_bonds):
        """

        Parameters
        ----------
        rings:
        broken_bonds:

        Returns
        -------
        unbroken: list
            List of unbroken rings.
        """
        unbroken = []
        for ring in rings:
            is_unbroken = True
            for bond in broken_bonds:
                if bond in self.setup.get_bonds_in_ring(
                    ring
                ):  # consider precalculating bonds
                    is_unbroken = False
                    break  # pun intended
            if is_unbroken:
                unbroken.append(ring)
        return unbroken

    # region Deprecated

    def get_broken_rings(self, rings, broken_bonds):
        broken_rings = []
        for ring in rings:
            is_broken = False
            for bond in broken_bonds:
                if bond in self.setup.get_bonds_in_ring(
                    ring
                ):  # consider precalculating bonds
                    is_broken = True
                    break  # pun intended
            if is_broken:
                broken_rings.append(ring)
        return broken_rings

    def show_macrocycle_scores(self, setup):
        print("Warning: not showing macrocycle scores, check implementation.")
        return
        if setup is not None:
            print("\n==============[ MACROCYCLE SCORES ]================")
            bond_by_ring = defaultdict(list)

            for bond_id, data in list(setup.ring_bond_breakable.items()):
                ring_id = data["ring_id"]
                bond_by_ring[ring_id].append(bond_id)

            for ring_id, bonds in list(bond_by_ring.items()):
                data = []
                print(
                    "-----------[ ring id: %s | size: %2d ]-----------"
                    % (",".join([str(x) for x in ring_id]), len(ring_id))
                )

                for b in bonds:
                    score = setup.ring_bond_breakable[b]["score"]
                    data.append((b, score))

                data = sorted(data, key=itemgetter(1), reverse=True)

                for b_count, b in enumerate(data):
                    begin = b[0][0]
                    end = b[0][1]
                    # bond = self.mol.setup.get_bond(b[0][0], b[0][1])
                    # begin = bond.GetBeginAtomIdx()
                    # end = bond.GetEndAtomIdx()
                    info = (
                        b_count,
                        begin,
                        end,
                        b[1],
                        "#" * int(b[1] / 5),
                        "-" * int(20 - b[1] / 5),
                    )
                    print("[ %2d] Bond [%3d --%3d] s:%3d [%s%s]" % info)

    # endregion
