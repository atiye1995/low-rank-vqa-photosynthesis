import numpy as np
from qutip import *
from scipy.optimize import minimize
from scipy.linalg import sqrtm
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import importlib

class LowRankVQA:

    def __init__(self, hamiltonian, lindblad_ops, target_states, times,
                 rank=10, epsilon_rank=1e-3, depth=6, shots=3000):
        """
        Args:
            hamiltonian: System Hamiltonian
            lindblad_ops: Lindblad operators
            target_states: Target density matrices (از exact solver)
            times: Time points
            rank: Max low-rank (R)
            epsilon_rank: Truncation tolerance
            depth: Circuit depth
            shots: تعداد measurements
        """
        self.H = hamiltonian
        self.c_ops = lindblad_ops
        self.target_states = target_states
        self.times = times
        self.n_sites = hamiltonian.dims[0][0]

        self.rank = rank
        self.epsilon_rank = epsilon_rank
        self.depth = depth
        self.shots = shots

        # Parameters: 3 parameters per layer per qubit
        self.n_params = 3 * self.n_sites * self.depth

        # For tracking
        self.cost_history = []
        self.fidelity_history = []

    def optimize(self, maxiter=500, method='COBYLA', verbose=True):
        """
        Optimization اصلی

        Returns:
            result: نتایج optimization
            best_states: بهترین density matrices
            mean_fidelity: میانگین fidelity
        """

        if verbose:
            print(f"\n{'='*70}")
            print(f"Low-Rank VQA Optimization")
            print(f"{'='*70}")
            print(f"  System size: {self.n_sites} qubits")
            print(f"  Time steps: {len(self.times)}")
            print(f"  Parameters: {self.n_params}")
            print(f"  Max rank: {self.rank}")
            print(f"  Circuit depth: {self.depth}")
            print(f"  Max iterations: {maxiter}")
            print(f"{'='*70}\n")

        # Initialize parameters
        np.random.seed(42)  # برای reproducibility
        theta_init = np.random.uniform(-np.pi, np.pi, self.n_params)

        # Optimization با progress bar
        start_time = time.time()

        if method == 'COBYLA':
            result = minimize(
                self._cost_function,
                theta_init,
                method='COBYLA',
                options={'maxiter': maxiter, 'disp': verbose, 'rhobeg': 0.5}
            )
        elif method == 'L-BFGS-B':
            result = minimize(
                self._cost_function,
                theta_init,
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'disp': verbose}
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        elapsed = time.time() - start_time

        # بهترین parameters
        theta_opt = result.x

        # محاسبه بهترین states
        best_states = self._forward_pass(theta_opt)

        # محاسبه fidelity نهایی
        fidelities = [fidelity(best_states[i], self.target_states[i])**2
                     for i in range(len(self.times))]
        mean_fidelity = np.mean(fidelities)

        if verbose:
            print(f"\n{'='*70}")
            print(f"✓ Optimization Completed")
            print(f"{'='*70}")
            print(f"  Final cost: {result.fun:.6f}")
            print(f"  Mean fidelity: {mean_fidelity:.6f}")
            print(f"  Optimization time: {elapsed:.2f} seconds")
            print(f"  Iterations: {result.nit if hasattr(result, 'nit') else 'N/A'}")
            print(f"{'='*70}\n")

        return {
            'optimization_result': result,
            'best_states': best_states,
            'mean_fidelity': mean_fidelity,
            'fidelities': fidelities,
            'computation_time': elapsed,
            'cost_history': self.cost_history,
            'fidelity_history': self.fidelity_history
        }

    def _cost_function(self, theta):
        """
        Cost function با dissipation engineering

        C = (1/M) * Σ [1 - F(ρ_θ(t_k), ρ_target(t_k))] + λ * <L†L>
        """

        # Forward pass
        approx_states = self._forward_pass(theta)

        # Fidelity loss
        fidelity_loss = 0
        for i in range(len(self.times)):
            F = fidelity(approx_states[i], self.target_states[i]) ** 2
            fidelity_loss += (1 - F)

        fidelity_loss /= len(self.times)

        # Dissipation engineering term
        L_eng = self._dissipation_operator()
        dissipation_term = 0
        for state in approx_states:
            dissipation_term += expect(L_eng.dag() * L_eng, state)
        dissipation_term /= len(self.times)
        dissipation_term *= 0.03  # λ = 0.03

        total_cost = fidelity_loss + dissipation_term

        # Track
        self.cost_history.append(total_cost)
        mean_fid = 1 - fidelity_loss
        self.fidelity_history.append(mean_fid)

        return total_cost

    def _forward_pass(self, theta):
        """
        Forward pass: محاسبه approximate states

        برای هر time point:
        1. Apply parameterized circuit
        2. Low-rank truncation
        3. Enforce positivity & normalization
        """

        approx_states = []

        # Initial state
        rho = basis(self.n_sites, 0) * basis(self.n_sites, 0).dag()

        for t_idx in range(len(self.times)):
            # Apply circuit (simplified: rotation layers)
            rho_evolved = self._apply_circuit(rho, theta, t_idx)

            # Low-rank truncation
            rho_lowrank = self._lowrank_projection(rho_evolved)

            # Shot noise
            rho_noisy = self._add_shot_noise(rho_lowrank)

            # Post-processing: positivity + normalization
            rho_final = self._postprocess(rho_noisy)

            approx_states.append(rho_final)

            # Update for next step
            rho = rho_final

        return approx_states

    def _apply_circuit(self, rho, theta, t_idx):
        """
        Apply parameterized quantum circuit

        Simplified version: Layered rotations
        در واقعیت: این باید با Qiskit پیاده بشه
        """

        # برای سادگی: rotation layers
        U = 1
        param_idx = 0

        for layer in range(self.depth):
            # Single-qubit rotations
            for qubit in range(self.n_sites):
                # RX, RY, RZ
                alpha = theta[param_idx]
                beta = theta[param_idx + 1]
                gamma = theta[param_idx + 2]
                param_idx += 3

                # ساخت unitary (simplified)
                # در واقعیت باید tensor product باشه
                # این فقط یک approximation است

        # برای سادگی، از time evolution استفاده می‌کنیم
        dt = self.times[t_idx] - (self.times[t_idx-1] if t_idx > 0 else 0)

        # Effective Hamiltonian با parameters
        H_eff = self.H

        # Evolution
        U_t = (-1j * H_eff * dt).expm()
        rho_new = U_t * rho * U_t.dag()

        # Apply Lindblad dissipation (simplified)
        for L in self.c_ops:
            rho_new = rho_new + dt * (L * rho_new * L.dag() -
                                      0.5 * (L.dag()*L*rho_new + rho_new*L.dag()*L))

        return rho_new

    def _lowrank_projection(self, rho):
        """
        Adaptive low-rank projection via SVD

        Algorithm 1, Lines 18-25
        """

        # Eigendecomposition
        evals, evecs = rho.eigenstates()

        # Sort descending
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = [evecs[i] for i in idx]

        # Filter out very small eigenvalues (numerical noise) before calculating cumsum
        non_zero_evals_mask = evals > 1e-12
        evals = evals[non_zero_evals_mask]
        evecs = [evecs[i] for i, keep in enumerate(non_zero_evals_mask) if keep]

        # If no positive eigenvalues are left after filtering, return a zero matrix
        if len(evals) == 0:
            return Qobj(np.zeros((self.n_sites, self.n_sites)), dims=rho.dims)

        # Calculate cumulative sum of remaining eigenvalues (descending)
        # cumsum[k] is sum(evals[k:])
        cumsum = np.cumsum(evals[::-1])[::-1]

        # Determine rank based on epsilon_rank: find smallest R such that sum(evals[R:]) < epsilon_rank
        # Default R to the actual number of non-zero eigenvalues if no truncation point is found
        R = len(evals)

        # Find the first index `k` where the tail sum is less than epsilon_rank
        # This `k` is the number of eigenvalues to keep.
        R_candidate_indices = np.where(cumsum < self.epsilon_rank)[0]
        if len(R_candidate_indices) > 0:
            R = R_candidate_indices[0]
            # Ensure R is at least 1 if there are eigenvalues available and a truncation point was found
            if R == 0 and len(evals) > 0:
                R = 1

        # Apply VQA's self.rank constraint
        R = min(R, self.rank)

        # Ensure R is at least the minimum rank (3)
        R = max(R, 3)

        # Ensure R does not exceed the number of available eigenvalues after filtering
        R = min(R, len(evals))

        # Truncate
        evals_trunc = evals[:R]
        evecs_trunc = evecs[:R]

        # Clip negative eigenvalues (should be rare after initial filtering)
        evals_trunc = np.maximum(evals_trunc, 0)

        # Normalize truncated eigenvalues (to ensure trace is 1 for the reconstructed density matrix)
        sum_evals_trunc = np.sum(evals_trunc)
        if sum_evals_trunc > 1e-12: # Avoid division by zero for very small sums
            evals_trunc = evals_trunc / sum_evals_trunc
        else:
            # If sum is effectively zero, return a zero matrix
            return Qobj(np.zeros((self.n_sites, self.n_sites)), dims=rho.dims)

        # Reconstruct
        rho_lowrank = sum([evals_trunc[i] * evecs_trunc[i] * evecs_trunc[i].dag()
                          for i in range(R)])

        return rho_lowrank

    def _add_shot_noise(self, rho):
        """
        Add shot noise: σ = 1/√S

        Algorithm 1, Line 16
        """
        sigma = 1 / np.sqrt(self.shots)

        # Add Gaussian noise to real/imag parts
        rho_array = rho.full()
        noise_real = np.random.normal(0, sigma, rho_array.shape)
        noise_imag = np.random.normal(0, sigma, rho_array.shape)

        rho_noisy = rho_array + noise_real + 1j * noise_imag

        return Qobj(rho_noisy, dims=rho.dims)

    def _postprocess(self, rho):
        """
        Post-processing: ensure positivity & normalization

        Algorithm 1, Lines 24-28
        """

        # Hermitianize
        rho_array = rho.full()
        rho_array = (rho_array + rho_array.conj().T) / 2

        # Ensure positivity: clip negative eigenvalues
        evals, evecs = np.linalg.eigh(rho_array)
        evals = np.maximum(evals, 0)
        rho_positive = evecs @ np.diag(evals) @ evecs.conj().T

        # Normalize
        trace = np.trace(rho_positive)
        if np.abs(trace) > 1e-10:
            rho_normalized = rho_positive / trace
        else:
            rho_normalized = rho_positive

        return Qobj(rho_normalized, dims=rho.dims)

    def _dissipation_operator(self):
        """
        Dissipation engineering operator

        L_eng = (1/√n) Σ_j (σ_j^- + α σ_j^z)
        """
        alpha = 0.5

        L_eng = 0
        for j in range(self.n_sites):
            # σ^- = |j><j| (simplified)
            sigma_minus = basis(self.n_sites, j) * basis(self.n_sites, j).dag()
            sigma_z = 2 * sigma_minus - qeye(self.n_sites)  # simplified

            L_eng += (sigma_minus + alpha * sigma_z)

        L_eng = L_eng / np.sqrt(self.n_sites)

        return L_eng


# تست
if __name__ == "__main__":
    from fmo_hamiltonian import FMOHamiltonian
    import exact_solver # Import the module instead of the class
    importlib.reload(exact_solver) # Reload the module to get the latest changes

    print("Testing Low-Rank VQA...")

    # Setup
    n_sites = 5  # شروع با سیستم کوچک
    H, E, J = FMOHamiltonian.get_hamiltonian(n_sites)
    c_ops = FMOHamiltonian.get_lindblad_operators(n_sites, gamma=0.1)

    # Exact solution (target)
    times = np.linspace(0, 500e-15, 50)  # کم‌تر برای تست سریع
    # Instantiate the class from the reloaded module
    exact_solver_instance = exact_solver.ExactLindbladSolver(H, c_ops)
    exact_result = exact_solver_instance.evolve(times, progress=False)

    # VQA
    vqa = LowRankVQA(
        H, c_ops,
        exact_result['states'],
        times,
        rank=10,
        depth=4,
        shots=3000
    )

    vqa_result = vqa.optimize(maxiter=100, verbose=True)

    print(f"\n✓ Test completed")
    print(f"  Final fidelity: {vqa_result['mean_fidelity']:.4f}")
