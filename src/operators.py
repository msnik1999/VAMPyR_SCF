from vampyr import vampyr3d as vp
import numpy as np

# This file contains various operators used in SCF calculations.

class HelmholtzOperator:
    """The Helmholtz operator.

    Implements the Helmholtz operator for a vector of orbitals for use in the SCF procedure.
    Outputs a vector of orbitals after applying the Helmholtz operator to each input orbital.

    Attributes:
        mra: The multi-resolution analysis (MRA) object.
        alpha: The exponent parameter for the Helmholtz operator, usually the diagonal elements of the Fock matrix.
        precision: The desired precision for the operator.
    """
    def __init__(self, mra, alpha, precision):
        """Initializes the Helmholtz operators for each orbital."""
        self.mra = mra
        self.alpha = alpha
        self.precision = precision
        self.operators = []

        # create the exponents for the Helmholtz operators
        mu = [np.sqrt(-2.0 * l) if l < 0 else 1.0 for l in self.alpha]
        # initialize a Helmholtz operator for each exponent
        for m in mu:
            self.operators.append(vp.HelmholtzOperator(mra = self.mra, exp = m, prec = self.precision))

    def __call__(self, Phi):
        """Applies the Helmholtz operator to each orbital in the input vector.
        Arguments:
            Phi: A vector of orbitals to which the Helmholtz operator will be applied.
        Returns:
            A vector of orbitals after applying the Helmholtz operator."""
        # return np.array([self.operators[i](Phi[i]).crop(self.precision / 10) for i in range(len(Phi))])
        return np.array([self.operators[i](Phi[i]) for i in range(len(Phi))])

class CoulombOperator:
    """The Coulomb operator.
    
    Implements the Coulomb operator for a vector of orbitals for use in the SCF procedure.
    Outputs a vector of orbitals after applying the Coulomb operator to each input orbital.
    
    Attributes:
        mra: The multi-resolution analysis (MRA) object.
        Phi: The vector of orbitals.
        precision: The desired precision for the operator.
    """
    def __init__(self, mra, Phi, precision):
        """Initializes the Coulomb operator based on the input orbitals."""
        self.mra = mra
        self.Phi = Phi
        self.precision = precision
        # initialize the Poisson operator
        poisson = vp.PoissonOperator(mra, precision)

        # compute the total electron density rho
        rho = self.Phi[0]**2
        for i in range(1, len(self.Phi)):
            rho += self.Phi[i]**2
        # apply the Poisson operator to rho to get the Coulomb potential
        self.potential = 4.0 * np.pi * poisson(rho).crop(self.precision / 10)

    def __call__(self):
        """Applies the Coulomb operator to each orbital in the input vector.
        Returns:
            A vector of orbitals after applying the Coulomb operator.
        """
        return np.array([(self.potential * phi).crop(self.precision / 10) for phi in self.Phi])
    
class CoulombInteractionOperator:
    """The Coulomb interaction operator between two sets of orbitals.
    
    Implements the Coulomb operator for a vector of orbitals for use in the SCF procedure.
    Outputs a vector of orbitals after applying the Coulomb operator to each input orbital.
    
    Attributes:
        mra: The multi-resolution analysis (MRA) object.
        Phi: The vector of orbitals.
        precision: The desired precision for the operator.
    """
    def __init__(self, mra, Phi, precision):
        """Initializes the Coulomb operator based on the input orbitals."""
        self.mra = mra
        self.Phi = Phi
        self.precision = precision
        # initialize the Poisson operator
        poisson = vp.PoissonOperator(mra, precision)

        # compute the total electron density rho
        rho = self.Phi[0]**2
        for i in range(1, len(self.Phi)):
            rho += self.Phi[i]**2
        # apply the Poisson operator to rho to get the Coulomb potential
        self.potential = 4.0 * np.pi * poisson(rho).crop(self.precision / 10)

    def __call__(self, Phi):
        """Applies the Coulomb operator to each orbital in the input vector.
        Returns:
            A vector of orbitals after applying the Coulomb operator.
        """
        return np.array([(self.potential * phi).crop(self.precision / 10) for phi in Phi])

class ExchangeOperator:
    """The Exchange operator.
    
    Implements the Exchange operator for a vector of orbitals for use in the SCF procedure.
    Outputs a vector of orbitals after applying the Exchange operator to each input orbital.
    
    Attributes:
        mra: The multi-resolution analysis (MRA) object.
        Phi: The vector of orbitals.
        precision: The desired precision for the operator.
    """
    def __init__(self, mra, Phi, precision):
        """Initializes the Exchange operator based on the input orbitals."""
        self.mra = mra
        self.Phi = Phi
        self.precision = precision
        # Initialize the Poisson operator
        self.poisson = vp.PoissonOperator(mra, precision)

    def __call__(self):
        """Applies the Exchange operator to each orbital in the input vector.
        Returns:
            A vector of orbitals after applying the Exchange operator.
        """
        out = []
        # for the exchange, two loops over orbitals are needed
        for phi in self.Phi:
            tmp = (self.Phi[0] * self.poisson(phi * self.Phi[0])).crop(self.precision / 10)
            for i in range (1, len(self.Phi)):
                tmp += (self.Phi[i] * self.poisson(phi * self.Phi[i])).crop(self.precision / 10)
            out.append(tmp)
        return 4.0 * np.pi * np.array([phi.crop(self.precision / 10) for phi in out])
    
class ExchangeInteractionOperator:
    """The Exchange interaction operator between two sets of orbitals.
    
    Implements the Exchange operator for a vector of orbitals for use in the SCF procedure.
    Outputs a vector of orbitals after applying the Exchange operator to each input orbital.
    
    Attributes:
        mra: The multi-resolution analysis (MRA) object.
        Psi: The vector of orbitals.
        precision: The desired precision for the operator.
    """
    def __init__(self, mra, Psi, precision):
        """Initializes the Exchange operator based on the input orbitals."""
        self.mra = mra
        self.Psi = Psi
        self.precision = precision
        # Initialize the Poisson operator
        self.poisson = vp.PoissonOperator(mra, precision)

    def __call__(self, Phi):
        """Applies the Exchange operator to each orbital in the input vector.
        Returns:
            A vector of orbitals after applying the Exchange operator.
        """
        out = []
        # for the exchange, two loops over orbitals are needed
        for psi in self.Psi:
            tmp = (Phi[0] * self.poisson(psi * Phi[0])).crop(self.precision / 10)
            for i in range (1, len(Phi)):
                tmp += (Phi[i] * self.poisson(psi * Phi[i])).crop(self.precision / 10)
            out.append(tmp)
        return 4.0 * np.pi * np.array([phi.crop(self.precision / 10) for phi in out])

class KineticOperator:
    """The Kinetic operator.

    Implements the Kinetic operator for a vector of orbitals for use in the SCF procedure.
    Outputs a vector of orbitals after applying the Kinetic operator to each input orbital.

    Attributes:
        mra: The multi-resolution analysis (MRA) object.
        precision: The desired precision for the operator.
    """
    def __init__(self, mra, precision):
        """Initializes the Kinetic operator."""
        self.mra = mra
        self.precision = precision
        # Initialize the derivative operator
        self.derivative = vp.ABGVDerivative(mra, 0.5, 0.5)

    def laplacian(self, phi):
        """Computes the Laplacian of a given orbital phi.
        Arguments:
            phi: The orbital for which the Laplacian will be computed.
        Returns:
            The Laplacian of the orbital phi.
        """
        return self.derivative(self.derivative(phi, 0), 0) + self.derivative(self.derivative(phi, 1), 1) + self.derivative(self.derivative(phi, 2), 2)

    def __call__(self, Phi):
        """Applies the Kinetic operator to each orbital in the input vector.
        Arguments:
            Phi: A vector of orbitals to which the Kinetic operator will be applied.
        Returns:
            A vector of orbitals after applying the Kinetic operator.
        """
        return np.array([(-0.5 * self.laplacian(phi)).crop(self.precision / 10) for phi in Phi])

class NuclearOperator:
    """The Nuclear operator.

    Implements the Nuclear operator for a vector of orbitals for use in the SCF procedure.
    Outputs a vector of orbitals after applying the Nuclear operator to each input orbital.

    Attributes:
        molecule: The molecule object containing atomic information.
        mra: The multi-resolution analysis (MRA) object.
        precision: The desired precision for the operator.
    """
    def __init__(self, molecule, mra, precision):
        """Initializes the Nuclear operator based on the input molecule."""
        self.molecule = molecule
        self.mra = mra
        self.precision = precision

        # Initialize the ScalingProjector
        P = vp.ScalingProjector(mra, precision)
        # Initialize the Nuclear function
        nucFunc = NucFunc(molecule)
        self.potential = P(nucFunc)

    def __call__(self, Phi):
        """Applies the Nuclear operator to each orbital in the input vector.
        Arguments:
            Phi: A vector of orbitals to which the Nuclear operator will be applied.
        Returns:
            A vector of orbitals after applying the Nuclear operator.
        """
        return np.array([(self.potential * phi).crop(self.precision / 10) for phi in Phi])

class NucFunc:
    """Nuclear potential function.

    Implements the nuclear potential function for a given molecule.

    Attributes:
        molecule: The molecule object containing atomic information.
        coords: The coordinates of the atoms in the molecule.
        charges: The charges of the atoms in the molecule.
        nAtoms: The number of atoms in the molecule.
    """
    def __init__(self, molecule):
        """Initializes the Nuclear potential function based on the input molecule."""
        self.molecule = molecule
        self.coords = np.array([atom.getCoords() for atom in self.molecule.getAtoms()])
        self.charges = np.array([atom.getCharge() for atom in self.molecule.getAtoms()])
        self.nAtoms = len(self.molecule.getAtoms())
    
    def __call__(self, r):
        """Computes the nuclear potential at a given point r. 

        V_nuc(r) = - sum_A Z_A / |R_A - r|

        Arguments:
            r: The point at which the nuclear potential will be computed.
        Returns:
            The nuclear potential at point r.
        """
        r = np.asarray(r, dtype=float)
        out = -sum(self.charges / np.linalg.norm(self.coords - r, axis=1))
        return float(out)