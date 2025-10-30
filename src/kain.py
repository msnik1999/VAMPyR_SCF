from vampyr import vampyr3d as vp
import numpy as np

import operators
import system

# This file contains all the functions needed for the SCF KAIN procedure. The implementation is based on 
# src/scfsolv.py from https://github.com/Dheasra/response.

class SCF:
    """The SCF class implements the KAIN SCF procedure.

    Attributes:
        molecule: The molecular system for which the SCF procedure is performed.
        mra: The multi-resolution analysis (MRA) object.
        precision: The desired precision for the SCF calculations.
    """

    mra = vp.MultiResolutionAnalysis            # mra object
    molecule = system.Molecule                  # molecular system
    precision = float                           # desired precision
    Phi_np1 = np.array                          # new orbitals
    Phi = list                                  # history of orbitals
    f_history = list                            # history of orbital updates
    F = np.array                                # Fock matrix   
    V = np.array                                # total potential
    J_n = np.array                              # Coulomb potential 
    K_n = np.array                              # Exchange potential
    Kin = operators.KineticOperator             # Kinetic operator
    V_nuc = operators.NuclearOperator           # Nuclear operator
    kain_history = int                          # number of previous iterations to use in KAIN
    kain_start = int                            # iteration number to start applying KAIN
    nOrbs = int                                 # number of orbitals
    energies = list                             # energy contributions      
    updates = list                              # orbital updates
    E_nuc = float                               # nuclear repulsion energy


    def __init__(self, molecule, mra, precision):
        """Initializes the SCF procedure with a molecular system, MRA, and precision."""
        self.molecule = molecule
        self.mra = mra
        self.precision = precision

        self.V_nuc = operators.NuclearOperator(self.molecule, self.mra, self.precision)
        self.Kin = operators.KineticOperator(self.mra, self.precision)

        self.energies = []
        self.updates = []

        self.E_nuc = self.calculateNuclearEnergy()


    def initialGuess(self):
        """Generates the initial guess for the KAIN SCF procedure by
        loading orbitals generated with MRChem from disk.

        Performs a first SCF step to populate the history needed for KAIN.
        """
        # initialize variables
        init_g_dir = self.molecule.getPath()
        atoms = self.molecule.getAtoms()
        self.nOrbs = int((np.array([atom.getCharge() for atom in atoms]).sum()) / 2.0)
        Phi_n = []
        # load initial guess from disk
        for i in range(self.nOrbs):
            phi = vp.FunctionTree(self.mra)
            phi.loadTree(f"{init_g_dir}phi_p_scf_idx_{i}_re")
            Phi_n.append(phi)
        # populate the KAIN history structures
        self.Phi_np1 = np.array(Phi_n)
        self.Phi = [[phi] for phi in Phi_n]
        self.f_history = [[] for i in range(self.nOrbs)]

        # make a first step to obtain a minimal history
        # calculate initial Fock matrix
        self.calculateFock()
        # calculate new orbitals
        Lambda = np.diag(self.F)
        G = operators.HelmholtzOperator(self.mra, Lambda, self.precision)
        self.Phi_np1 = -2 * G(self.V + (np.diag(Lambda) - self.F) @ self.Phi_np1)
        # orthogonalize new orbitals and normalize
        for i in range(self.nOrbs):
            self.f_history[i].append(self.Phi_np1[i] - self.Phi[i][-1])
            self.Phi_np1[i].normalize()
        self.loewdinOrthogonalization()
        # add new orbitals to history and crop history if necessary
        for i in range(self.nOrbs):
            self.Phi[i].append(self.Phi_np1[i])
            if self.kain_history == 0:
                del self.f_history[i][0]
                del self.Phi[i][0]
        return
    
    def runSCF(self, threshold, maxIter = 100, kain_start = 0, kain_history = 5):
        """Runs the KAIN SCF procedure.

        Arguments:
            threshold: The convergence threshold for the SCF procedure.
            maxIter: The maximum number of SCF iterations to perform.
            kain_start: The iteration number to start applying KAIN.
            kain_history: The number of previous iterations to use in KAIN.

        Returns:
            The converged orbitals, energies, and updates.
        """
        self.kain_start = kain_start
        self.kain_history = kain_history

        # calculate first initial guess and create a first history
        self.initialGuess()
        # calculate next Fock matrix
        self.calculateFock()
        
        iteration = 0
        while True:
            print(f"=============Iteration: {iteration}")
            # Here, we calculate new orbitals, apply KAIN and calculate the new Fock matrix
            self.expandSolution(iteration)
            energy = self.energies[-1]
            print(iteration, " |  E_tot:", energy["$E_{tot}$"], " |  E_HF:", energy["$E_{HF}$"], " |  dPhi:", max(self.updates[-1]))
            print("E_orb:", energy["$E_{orb}$"], " |  E_en:", energy["$E_{en}$"], " |  E_el:", energy["$E_{el}$"], " |  E_kin:", energy["$E_{kin}$"])
            iteration += 1

            if max(self.updates[-1]) < threshold or iteration > maxIter:
                break
        return self.Phi, self.energies, np.array(self.updates)

    def expandSolution(self, iteration):
        """Expands the current solution using KAIN to obtain new orbitals.
        
        Arguments:
            iteration: The current SCF iteration number.
        """
        # obatain new orbitals from current guess
        Lambda = np.diag(self.F)
        G = operators.HelmholtzOperator(self.mra, Lambda, self.precision)
        self.Phi_np1 = -2 * G(self.V + (np.diag(Lambda) - self.F) @ self.Phi_np1)
        # orthogonalize new orbitals and obtain new orbital update
        self.loewdinOrthogonalization() # maybe not?
        
        update = []
        for i in range(self.nOrbs):
            # add current orbital and new orbital update to history
            self.f_history[i].append(self.Phi_np1[i] - self.Phi[i][-1])
            # solve Ax = b
            x = self.setupLinearSystem(self.Phi[i], self.f_history[i])
            # calculate correction to orbital update
            delta = self.f_history[i][-1]
            # dPhi[i] = sum_j (Phi[i][j] + f_history[i][j] - Phi[i][-1] - f_history[i][-1]) * x[j]
            tmp = self.Phi[i][-1] + self.f_history[i][-1] # for efficiency
            for j in range(len(x)):
                linSys = self.Phi[i][j] + self.f_history[i][j] - tmp
                delta += x[j] * linSys
            # calculate convergence measure
            update.append(delta.norm())
            # update current orbitals with new correction
            self.Phi_np1[i] = self.Phi[i][-1] + delta
            self.Phi_np1[i].normalize()

        # orthogonalize new orbitals
        self.loewdinOrthogonalization()
        # add new orbitals to history and crop history if necessary
        for i in range(self.nOrbs):
            self.Phi[i].append(self.Phi_np1[i])
            if iteration < self.kain_start or len(self.Phi[i]) > self.kain_history:
                del self.Phi[i][0]
                del self.f_history[i][0]
        
        # calculate new Fock matrix and energies
        self.calculateFock()
        energy = self.calculateEnergy() 

        self.energies.append(energy)
        self.updates.append(update)
        return 

    def setupLinearSystem(self, phi, dphi_history):
        """Sets up and solves the linear system Ax = b needed for KAIN.

        Arguments:
            phi: The history of a given orbital.
            dphi_history: The history of the orbital updates for the given orbital.

        Returns:
            The solution vector x of the linear system Ax = b."""
        # lenHistory only includes previous steps, not current
        lenHistory = len(phi) - 1
        # initialize A and b
        A = np.zeros((lenHistory, lenHistory))
        b = np.zeros((lenHistory))
        for i in range(lenHistory):
            dPhi = phi[i] - phi[-1]
            # b_i = < phi_i - phi_n | f_n >
            b[i] = vp.dot(dPhi, dphi_history[-1])
            for j in range(lenHistory):
                # A_ij = < phi_i - phi_n | f_j - f_n >
                A[i, j] -= vp.dot(dPhi, dphi_history[j] - dphi_history[-1])
        # solve Ax = b
        return np.linalg.solve(A, b)

    def calculateFock(self):
        """Calculates the Fock matrix, as well as the underlying potentials based on the current orbitals."""
        # initialize Coulomb and exchange operators
        J = operators.CoulombOperator(self.mra, self.Phi_np1, self.precision)
        K = operators.ExchangeOperator(self.mra, self.Phi_np1, self.precision)
        # calculate V_ne based on current orbitals
        V_n = self.V_nuc(self.Phi_np1)
        # calculate J and K based on current orbitals
        self.J_n = J()
        self.K_n = K()
        # calculate total potential V
        self.V = V_n + 2 * self.J_n - self.K_n
        # caclulate the Fock matrix
        self.F = self.calc_overlap(self.Phi_np1, self.V + self.Kin(self.Phi_np1), diag = False)
        return
    
    def loewdinOrthogonalization(self):
        """Applies Loewdin orthogonalization to the current set of orbitals."""
        # calculate overlap matrix S
        S = self.calc_overlap(self.Phi_np1, self.Phi_np1)
        # solve for S^(-1/2)
        sigma, U = np.linalg.eigh(S)
        Sm5 = U @ np.diag(sigma**(-0.5)) @ U.transpose()
        # apply S^(-1/2) to orbitals
        self.Phi_np1 = Sm5 @ self.Phi_np1
        # normalize orbitals
        for phi in self.Phi_np1:
            phi.normalize()
        return 

    def calc_overlap(self, Psi, Phi, diag = False):
        """Calculates the overlap matrix S between two sets of orbitals.

        S[i,j] = <Psi[i] | Phi[j]>\n
        Accounts for symmetry if diag = True.

        Arguments:
            Psi: First set of orbitals.
            Phi: Second set of orbitals.
            diag: If True, symmetrize the overlap matrix.

        Returns:
            The overlap matrix S.
        """
        # initialize overlap matrix
        S = np.zeros((len(Psi), len(Phi)))
        for i in range(len(Psi)):
            # only compute upper triangle if diag = True by starting j from i
            # 0 + diag * i evaluates to i if diag = True, else 0
            for j in range(0 + diag * i, len(Phi)):
                S[i, j] = vp.dot(Psi[i], Phi[j])
        # symmetrize overlap matrix if diag = True
        if diag:
            S += S.T
            # the diagonal elements were added twice, so divide them by 2
            S[np.diag_indices_from(S)] /= 2.0
        return S

    def calculateEnergy(self):
        """Calculates the different energy contributions based on the current orbitals and Fock matrix.

        Returns:
            A dictionary containing the different energy contributions."""
        # calculate different Fock contributions as matrices
        V_mat = self.calc_overlap(self.Phi_np1, self.V)
        J_mat = self.calc_overlap(self.Phi_np1, self.J_n)
        K_mat = self.calc_overlap(self.Phi_np1, self.K_n)

        # calculate energy contributions
        e = 2.0 * self.F.trace()
        E_en = 2.0 * V_mat.trace()
        E_el = 2.0 * J_mat.trace() - K_mat.trace()
        E_tot = e - E_el
        E_kin = E_tot - E_en - E_el
        E_HF = E_tot + self.E_nuc

        return {
            "$E_{orb}$": e,
            "$E_{en}$": E_en,
            "$E_{el}$": E_el,
            "$E_{kin}$": E_kin,
            "$E_{tot}$": E_tot,
            "$E_{HF}$": E_HF
        }

    def calculateNuclearEnergy(self):
        """Calculates the nuclear repulsion energy of the molecular system.
        Returns:
            The nuclear repulsion energy as a float.
        """
        # get atomic coordinates and charges 
        coords = np.array([atom.getCoords() for atom in self.molecule.getAtoms()])
        charges = np.array([atom.getCharge() for atom in self.molecule.getAtoms()])
        nAtoms = len(self.molecule.getAtoms())
        # calculate nuclear repulsion energy by using double sum over all atom pairs
        energy = np.sum([
            charges[i] * charges[j] / np.linalg.norm(coords[i] - coords[j])
            for i in range(nAtoms) for j in range(i + 1, nAtoms)
        ])
        return float(energy)