from vampyr import vampyr3d as vp
import numpy as np
import os

import operators
import system
import kain

# This file contains the Frozen-Density Embedding (FDE) implementation

class FDE:

    sysA = system.Molecule
    sysB = system.Molecule
    mra = vp.MultiResolutionAnalysis
    precision = float
    nOrbs = int
    kain_start = int
    kain_history = int
    Phi_np1 = np.array
    PhiB = np.array
    Phi = list
    f_history = list
    F = np.array
    V = np.array
    V_act = np.array
    V_emb = np.array
    J_emb = operators.CoulombInteractionOperator
    K_emb = operators.ExchangeInteractionOperator
    J_n_emb = np.array
    K_n_emb = np.array
    J_n = np.array
    K_n = np.array
    Kin = operators.KineticOperator
    V_nuc_env = operators.NuclearOperator
    V_nuc_act = operators.NuclearOperator
    E_nuc = float
    E_nuc_act = float
    E_nuc_emb = float
    E_HF_env = float
    energies = list
    updates = list


    def __init__(self, sysA: system.Molecule, sysB: system.Molecule, mra, precision):
        self.sysA = sysA
        self.sysB = sysB
        self.mra = mra
        self.precision = precision
        self.energies = []
        self.updates = []

    def runFDE(self, threshold, maxIter = 100, kain_start = 0, kain_history = 5):
        # Placeholder for FDE SCF procedure
        self.kain_start = kain_start
        self.kain_history = kain_history

        print("Setting up isolated SCF calculations for subsystems A and B")
        scfA = kain.SCF(self.sysA, self.mra, self.precision)
        scfB = kain.SCF(self.sysB, self.mra, self.precision)

        print("Running isolated SCF for subsystem A")
        PhiA_history, energiesA, updatesA = scfA.runSCF(threshold, maxIter, self.kain_start, self.kain_history)
        print("Running isolated SCF for subsystem B")
        PhiB_history, energiesB, updatesB = scfB.runSCF(threshold, maxIter, self.kain_start, self.kain_history)
        print("Isolated SCF complete.\n")
        self.E_HF_env = energiesB[-1]['$E_{HF}$']

        PhiA = []
        PhiB = []
        for phi in PhiA_history:
            PhiA.append(phi[-1])
        for phi in PhiB_history:
            PhiB.append(phi[-1])
        self.nOrbs = len(PhiA)
        self.PhiB = np.array(PhiB)

        self.V_nuc_act = scfA.V_nuc
        V_nuc_act_tmp = self.V_nuc_act(self.PhiB)
        self.E_nuc_act = self.calc_overlap(self.PhiB, V_nuc_act_tmp).trace()
        self.E_nuc = self.calculateNuclearEnergy()
        self.E_nuc_emb = self.calculateNuclearEnergy(emb = True)

        self.V_nuc_env = scfB.V_nuc
        self.Kin = operators.KineticOperator(self.mra, self.precision)

        print("Running Embedded SCF")
        self.runEmbeddedSCF(threshold, maxIter, PhiA, PhiB)

        return self.Phi, self.energies, np.array(self.updates)

    def initialGuessEmbedding(self, PhiA, PhiB):
        # Placeholder for initial guess of embedding
        self.Phi_np1 = np.array(PhiA)
        self.Phi = [[phi] for phi in PhiA]
        self.f_history = [[] for i in range(self.nOrbs)]
        self.PhiB = PhiB

        # make a first SCF step to obtain a minimal history
        # calculate initial Fock matrix
        self.J = operators.CoulombInteractionOperator(self.mra, self.PhiB, self.precision)
        self.K = operators.ExchangeInteractionOperator(self.mra, self.PhiB, self.precision)
        self.calculateEmbeddingFock()
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
    
    def runEmbeddedSCF(self, threshold, maxIter, PhiA, PhiB):
        # Placeholder for embedded SCF procedure
        self.initialGuessEmbedding(PhiA, PhiB)

        self.calculateEmbeddingFock()

        iteration = 0
        while True:
            print(f"=============Iteration: {iteration}")
            # Here, we calculate new orbitals, apply KAIN and calculate the new Fock matrix
            self.expandSolution(iteration)
            energy = self.energies[-1]
            print("Energy Contributions:")
            print("---------------------")
            print(f"E_orb: {energy['$E_{orb}$']}", f" | E_en_act: {energy['$E_{en_act}$']}", f" | E_el_act: {energy['$E_{el_act}$']}")
            print(f"E_tot_act: {energy['$E_{tot_act}$']}", f" | E_HF_act: {energy['$E_{HF_act}$']}")
            print(f"E_en_emb: {energy['$E_{en_emb}$']}", f" | E_el_emb: {energy['$E_{el_emb}$']}")
            print(f"E_tot_emb: {energy['$E_{tot_emb}$']}", f" | E_HF_emb: {energy['$E_{HF_emb}$']}")
            print(f"E_kin: {energy['$E_{kin}$']}")
            print(f"Max Orbital Update: {max(self.updates[-1])}")
            print("---------------------\n")
            iteration += 1

            if max(self.updates[-1]) < threshold or iteration > maxIter:
                break
        return
    
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
        self.calculateEmbeddingFock()
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

    def calculateEmbeddingFock(self):
        # active system contributions only
        J_act = operators.CoulombOperator(self.mra, self.Phi_np1, self.precision)
        K_act = operators.ExchangeOperator(self.mra, self.Phi_np1, self.precision)
        self.J_n = J_act()
        self.K_n = K_act()
        # V_act = V_nuc_act + 2 * J_act - K_act 
        self.V_act = self.V_nuc_act(self.Phi_np1) + 2 * self.J_n - self.K_n

        # embedding contributions
        self.J_n_emb = self.J(self.Phi_np1)
        self.K_n_emb = self.K(self.Phi_np1)
        V_nuc_env_act = self.V_nuc_env(self.Phi_np1)
        # V_emb = V_nuc_env_act + 2 * J_act_env - K_act_env
        self.V_emb = V_nuc_env_act + 2 * self.J_n_emb
        # self.V_emb = V_nuc_env_act + 2 * self.J_n_emb - self.K_n_emb

        self.V = self.V_act + self.V_emb
        # calculate the embedded Fock matrix
        self.F = self.calc_overlap(self.Phi_np1, self.V + self.Kin(self.Phi_np1))
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
        V_act_mat = self.calc_overlap(self.Phi_np1, self.V_act)
        J_mat = self.calc_overlap(self.Phi_np1, self.J_n)
        K_mat = self.calc_overlap(self.Phi_np1, self.K_n)
        
        V_emb_mat = self.calc_overlap(self.Phi_np1, self.V_emb)
        J_mat_emb = self.calc_overlap(self.Phi_np1, self.J_n_emb)
        K_mat_emb = self.calc_overlap(self.Phi_np1, self.K_n_emb)

        # calculate energy contributions
        e = 2.0 * self.F.trace()
        E_en_act = 2.0 * V_act_mat.trace()
        E_el_act = 2.0 * J_mat.trace() - K_mat.trace()
        E_tot_act = e - E_el_act
        E_HF_act = E_tot_act + self.E_nuc

        E_en_emb = 2.0 * V_emb_mat.trace()
        E_el_emb = 2.0 * J_mat_emb.trace()
        # E_el_emb = 2.0 * J_mat_emb.trace() - K_mat_emb.trace()
        E_tot_emb = e - E_el_emb
        E_esi = 2 * self.E_nuc_act + self.E_nuc_emb + 4 * J_mat_emb.trace() + 2 * self.calc_overlap(self.Phi_np1, self.V_nuc_env(self.Phi_np1)).trace() 
        E_HF_emb = E_HF_act + E_esi + self.E_HF_env
        E_kin = E_tot_act - E_en_act - E_el_act + (E_tot_emb - E_en_emb - E_el_emb)

        print("E_esi: ", E_esi, " | E_nuc_act: ", 2 * self.E_nuc_act, " | E_nuc_emb: ", self.E_nuc_emb, " | E_e_env_e: ", 4 * J_mat_emb.trace(), " | E_nuc_env_act: ", 2 * self.calc_overlap(self.Phi_np1, self.V_nuc_env(self.Phi_np1)).trace())

        return {
            "$E_{orb}$": e,
            "$E_{en_act}$": E_en_act,
            "$E_{el_act}$": E_el_act,
            "$E_{tot_act}$": E_tot_act,
            "$E_{HF_act}$": E_HF_act,
            "$E_{en_emb}$": E_en_emb,
            "$E_{el_emb}$": E_el_emb,
            "$E_{tot_emb}$": E_tot_emb,
            "$E_{HF_emb}$": E_HF_emb,
            "$E_{kin}$": E_kin
        }
    
    def calculateNuclearEnergy(self, emb = False):
        """Calculates the nuclear repulsion energy of the molecular system.
        Returns:
            The nuclear repulsion energy as a float.
        """
        # get atomic coordinates and charges 
        coordsA = np.array([atom.getCoords() for atom in self.sysA.getAtoms()])
        chargesA = np.array([atom.getCharge() for atom in self.sysA.getAtoms()])
        nAtomsA = len(self.sysA.getAtoms())
        if emb:
            coordsB = np.array([atom.getCoords() for atom in self.sysB.getAtoms()])
            chargesB = np.array([atom.getCharge() for atom in self.sysB.getAtoms()])
            nAtomsB = len(self.sysB.getAtoms())
            energy = np.sum([
                chargesA[i] * chargesB[j] / np.linalg.norm(coordsA[i] - coordsB[j])
                for i in range(nAtomsA) for j in range(nAtomsB)
            ])
        else:
            # calculate nuclear repulsion energy by using double sum over all atom pairs
            energy = np.sum([
                chargesA[i] * chargesA[j] / np.linalg.norm(coordsA[i] - coordsA[j])
                for i in range(nAtomsA) for j in range(i + 1, nAtomsA)
            ])
        return float(energy)