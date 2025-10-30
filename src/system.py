from vampyr import vampyr3d as vp

# This file contains the classes Molecule and Atom used to define the molecular system
# as well as a helper function to intitialize the MRA.

class Molecule:
    """The Molecule class represents a molecular system.
    
    Attributes:
        atoms: A list of Atom objects representing the atoms in the molecule.
        path: The file path for the initial guess of the molecular orbitals.
    """
    def __init__(self, atoms = [], path = "./initial_guess/"):
        """Initializes the Molecule with a list of atoms and a file path."""
        self.atoms = atoms
        self.path = path

    def getAtoms(self):
        """Returns the list of atoms in the molecule."""
        return self.atoms
    
    def getPath(self):
        """Returns the file path for the initial guess of the molecular orbitals."""
        return self.path

class Atom:
    """The Atom class represents an atom in a molecule.

    Attributes:
        coords: A tuple representing the (x, y, z) coordinates of the atom.
        charge: The atomic charge of the atom.
    """
    def __init__(self, coords = (0.0, 0.0, 0.0), charge = 1):
        """Initializes the Atom with coordinates and charge."""
        self.coords = coords
        self.charge = charge

    def getCoords(self):
        """Returns the coordinates of the atom."""
        return self.coords
    
    def getCharge(self):
        """Returns the atomic charge of the atom."""
        return self.charge
    
def initMRA(nBoxes = 2, scaling = 1.0, scale = -4, order = 8):
    """Initializes the Multi-Resolution Analysis (MRA) for the SCF calculations.

    Arguments:
        nBoxes: Number of boxes in each dimension.
        scaling: Scaling factor for the MRA.
        scale: Scale level for the MRA.
        order: Order of the wavelet basis functions.

    Returns:
        An initialized Multi-Resolution Analysis (MRA) object.
    """
    world = vp.BoundingBox(corner = [-1]*3, nboxes = [nBoxes]*3, scaling = [scaling]*3, scale = scale)
    return vp.MultiResolutionAnalysis(order = order, box = world)