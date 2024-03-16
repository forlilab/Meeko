# Molecule Setup Notes

This is a document meant as a temporary stop-gap to help us get technical tasks organized.
This one pertains specifically to molecule setups, and contains notes, bugs, and feature 
requests from the original molsetup.py file as well as details for plans to further update
and refactor molecule setups, and information about the status of what has been done so far.

### Implementation Notes From The Original File

* Modify so there are no more dictionaries and only lists/arrays.
  * methods like `add_x`, `delete_x`, `get_x` wil deal with indexing
* Provide infrastructure to calculate "differential"
modifications, like tautomers: the class calculating tautomers will copy and modify the setup 
to swap protons and change the atom types (and possibly rotations?), then
there will e a function to extract only differences (i.e., atom types and coordinates,
in the case of tautomers) and store them to make it a switch-state
* Only use 1-dimensional arrays (C)
* Update `add_atom` to not specify neighbors, which should be defined only when a bond is created?
* Change all attributes_to_copy to have underscore?
* There was a note in a commented out function called `get_atom_ring_count` that said that it should be
replaced by `get_atom_rings`. `Get_atom_rings` exists now in some form right below this comment.
* `self.get_atom_rings` should replace `get_atom_ring_count`
* Evaluate if `_get_attrib` is useful
* In `del_bond`, check if we want to delete nodes that have no connections (we might want to
keep them)
* In `copy_attributes_from`:  enable some kind of plugin system here too, to allow other 
\objects to add attributes? ALthough this would make the setup more fragile -> better to have
attributes explicitly added here and that's it
* After the `setattr` line of `copy_attributes_from` there is a todo marking a potential bug. The molecule
is shared by the different setups. If one of them alters the molecule, properties will not be the same.
* In the RDKit implementation of init_atom, there is a todo to check the consiostency for chiral model between
OB and RDKit

### Refactoring Molecule Setups

#### Proposed Changes
* Eliminate dictionaries and add support to checks that make sure the molecule setup is a valid molecule setup.
  1. Convert all of the dictionaries that used to be ordered dicts mapping from array index to some property to lists.
  2. Ensure that there are the checks mentioned in the original comment in add, delete, and get functions.
  3. Determine if there are additional molsetup-wide checks that we want to do to make sure (for instance) that all
     populated lists are the same length.
* Create enums for `atom_type`, `atom_params`, `flexibility_model`, where we want only certain defined values to appear
  (confirm with other people in the lab on what these should be and what variables it would be a good idea to do this 
   for)
  * Current observed `atom_type` values in Meeko are as follows: atom_type, rmin_half, epsilon, pull_charge_fraction,
    gasteiger, atype
* Clean up the inheritance and interface design
  * More strictly define what is a function that we need the inheriting classes to define and what is a function that is
    fully implemented in the parent MoleculeSetup class.
  * Make sure call signatures for functions are consistent between parent and subclasses.
  * Refine what we're using formal vs informal interfaces for.
* Variables that are specific to specific subclasses should not be mentioned or initialized in the parent MoleculeSetup 
  class.
  * Currently the only identified ones are mol and rmsd_symmetry_indices, check if this is incorrect.
* Clean up and fully comment `from_prmtop_inpcrd`. Check existing comments for accuracy as they are full of assumptions 
  Parnika made.
* Organize Molecule Setup class so all the getters and setters are together, and that functionality is generally grouped
  * Reconsider the use of getters and setters, convert the appropriate attributes to properties and limit access
    as necessary
* Fully comment and document.
  * `add_pseudo_atom` needs info
  * `write_xyz_string` needs info
* Rethink the organization of the information storage (potentially pull out certain pieces of data to be in their own
  data structures/classes if it makes logical sense to group them and if they're being set at a particular time).
* Remove coords from molsetups (they will be stored elsewhere, likely in a different data structure)
* Define what variables get initialized when in Molecule Setups (so what is a basic MoleculeSetup object vs what is 
  a populated MoleculeSetup), and modify the variable initialization to reflect this.
* Add unit tests for all of the MoleculeSetup functions.
* Fill out unit tests for all the MoleculeSetup functions, confirm with other members of the lab that these are the
  correct test cases to have and that we're covering edge and corner cases to the best of our ability.
* Refactor all functions to be stricter about types and to be more robust for checks.
  * Make sure all variable names make sense and that there is less room for ambiguity/interpretation
  * Get rid of magic numbers (typically int or float literals that are in places in code)
  * Make sure all loops are clear and well-documented

#### Implemented Changes
* Convert OrderedDict collection to dicts since newer versions of python dictionaries maintain insertion order.
* Refine existing tasks for Molecule Setups

### Open Questions
* Are we deprecating/deleting `del_atom` or is it functionality we still want to add. It currently exists as
the shell of a function that could be.
* Re: Stefano's earlier comments on `get_atom_rings`, is any of that still relevant or are we leaving the current
implementation as-is
* I have decided to leave all of the inits as literals rather than `dict()` and `list()` declarations because it is
a bit more efficient, but the other choice might be a little bit clearer to a reader unfamiliar with Python. I do not
think it would be unreasonable to expect a reader to be familiar with Python, but I thought it would be good to formally
point this out.
* Is there a particular reason why there is exactly one dict that is using defaultdict? How heavily are we relying on
  that?
* Both atom and pseudoatom add methods have this overwrite feature, and we are manually tracking pseudatoms 