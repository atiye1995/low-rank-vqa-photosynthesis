"""
noise_models.py
===============
مدل‌های نویز واقعی برای NISQ devices
IBM Heron specifications
"""

import numpy as np
from qutip import *
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
from qiskit_aer.noise import amplitude_damping_error, phase_damping_error
import warnings
warnings.filterwarnings('ignore')

class NISQNoiseModel:
    """
    NISQ Noise Models

    Implements realistic noise based on IBM Heron specifications:
    - Single-qubit gate error: 5×10^-5
    - Two-qubit gate error: 1×10^-4
    - T1 = 150 μs (amplitude damping)
    - T2 = 144 μs (dephasing)
    - Readout error: 0.8%
    """

    # IBM Heron parameters
    HERON_PARAMS = {
        'single_qubit_error': 5e-5,
        'two_qubit_error': 1e-4,
        'T1': 150e-6,  # seconds
        'T2': 144e-6,  # seconds
        'readout_error': 0.008,
        'gate_time_1q': 40e-9,  # 40 ns
        'gate_time_2q': 120e-9  # 120 ns
    }

    def __init__(self, backend='ibm_heron'):
        """
        Args:
            backend: 'ibm_heron', 'ideal', or custom dict
        """
        if backend == 'ibm_heron':
            self.params = self.HERON_PARAMS.copy()
        elif backend == 'ideal':
            self.params = {k: 0 for k in self.HERON_PARAMS.keys()}
        elif isinstance(backend, dict):
            self.params = backend
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.backend_name = backend if isinstance(backend, str) else 'custom'

    def apply_noise_to_state(self, rho, gate_type='single', n_gates=1):
        """
        Apply noise channels to density matrix

        Args:
            rho: Input density matrix (Qobj)
            gate_type: 'single' or 'two' qubit gate
            n_gates: تعداد gates اعمال شده

        Returns:
            rho_noisy: Density matrix با نویز
        """

        if self.backend_name == 'ideal':
            return rho

        rho_noisy = rho

        for _ in range(n_gates):
            # 1. Depolarizing noise
            rho_noisy = self._apply_depolarizing(rho_noisy, gate_type)

            # 2. Amplitude damping (T1)
            rho_noisy = self._apply_amplitude_damping(rho_noisy, gate_type)

            # 3. Phase damping (T2)
            rho_noisy = self._apply_phase_damping(rho_noisy, gate_type)

            # 4. Readout error
            rho_noisy = self._apply_readout_error(rho_noisy)

        return rho_noisy

    def _apply_depolarizing(self, rho, gate_type):
        """
        Depolarizing channel: ε(ρ) = (1-p)ρ + p*I/d
        """
        if gate_type == 'single':
            p = self.params['single_qubit_error']
        else:
            p = self.params['two_qubit_error']

        if p == 0:
            return rho

        n = rho.dims[0][0]
        I = qeye(n)

        rho_depolarized = (1 - p) * rho + p * I / n

        return rho_depolarized

    def _apply_amplitude_damping(self, rho, gate_type):
        """
        Amplitude damping channel (T1 relaxation)

        K0 = [[1, 0], [0, √(1-γ)]]
        K1 = [[0, √γ], [0, 0]]
        """
        if gate_type == 'single':
            t_gate = self.params['gate_time_1q']
        else:
            t_gate = self.params['gate_time_2q']

        T1 = self.params['T1']
        gamma = 1 - np.exp(-t_gate / T1)

        if gamma == 0:
            return rho

        # Apply to each qubit (simplified)
        n = rho.dims[0][0]
        rho_array = rho.full()

        # Diagonal terms affected
        for i in range(n):
            if i > 0:  # excited states decay
                rho_array[i, i] *= (1 - gamma)
                rho_array[0, 0] += gamma * rho_array[i, i]

        # Off-diagonal terms
        rho_array *= np.sqrt(1 - gamma)

        return Qobj(rho_array, dims=rho.dims)

    def _apply_phase_damping(self, rho, gate_type):
        """
        Phase damping channel (T2 dephasing)

        K0 = [[1, 0], [0, √(1-λ)]]
        K1 = [[0, 0], [0, √λ]]
        """
        if gate_type == 'single':
            t_gate = self.params['gate_time_1q']
        else:
            t_gate = self.params['gate_time_2q']

        T2 = self.params['T2']
        lambda_param = 1 - np.exp(-t_gate / T2)

        if lambda_param == 0:
            return rho

        # Off-diagonal decay
        rho_array = rho.full()

        # Coherences decay
        decay_factor = np.exp(-t_gate / T2)
        for i in range(rho_array.shape[0]):
            for j in range(rho_array.shape[1]):
                if i != j:
                    rho_array[i, j] *= decay_factor

        return Qobj(rho_array, dims=rho.dims)

    def _apply_readout_error(self, rho):
        """
        Measurement/readout error

        Bit-flip with probability p_readout
        """
        p = self.params['readout_error']

        if p == 0:
            return rho

        # Simplified: add small random perturbation
        rho_array = rho.full()
        noise = p * np.random.randn(*rho_array.shape)
        rho_array += noise + 1j * noise

        # Re-hermitianize
        rho_array = (rho_array + rho_array.conj().T) / 2

        return Qobj(rho_array, dims=rho.dims)

    def get_effective_fidelity_reduction(self, circuit_depth):
        """
        محاسبه کاهش fidelity به ازای depth مشخص

        Returns:
            fidelity_factor: ضریب کاهش (0 to 1)
        """
        # تقریبی: fidelity ∝ (1 - error)^depth
        single_gate_fidelity = 1 - self.params['single_qubit_error']
        two_gate_fidelity = 1 - self.params['two_qubit_error']

        # فرض: در هر layer، n single-qubit + n/2 two-qubit gates
        effective_fidelity = (
            single_gate_fidelity ** circuit_depth *
            two_gate_fidelity ** (circuit_depth / 2)
        )

        return effective_fidelity

    def __repr__(self):
        return f"NISQNoiseModel(backend='{self.backend_name}')"


class NoiseSimulator:
    """
    Noise-aware VQA Simulator
    """

    def __init__(self, vqa_instance, noise_model):
        """
        Args:
            vqa_instance: LowRankVQA instance
            noise_model: NISQNoiseModel instance
        """
        self.vqa = vqa_instance
        self.noise = noise_model

    def run_noisy_optimization(self, maxiter=500, verbose=True):
        """
        Run VQA با noise model
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Noisy VQA Simulation")
            print(f"{'='*70}")
            print(f"  Noise model: {self.noise}")
            print(f"  Circuit depth: {self.vqa.depth}")
            print(f"  Expected fidelity reduction: "
                  f"{self.noise.get_effective_fidelity_reduction(self.vqa.depth):.4f}")
            print(f"{'='*70}\n")

        # Override forward pass با noisy version
        original_forward = self.vqa._forward_pass
        self.vqa._forward_pass = self._noisy_forward_pass

        # Run optimization
        result = self.vqa.optimize(maxiter=maxiter, method='COBYLA', verbose=verbose)

        # Restore
        self.vqa._forward_pass = original_forward

        return result

    def _noisy_forward_pass(self, theta):
        """
        Forward pass با noise
        """
        approx_states = []
        rho = basis(self.vqa.n_sites, 0) * basis(self.vqa.n_sites, 0).dag()

        for t_idx in range(len(self.vqa.times)):
            # Apply circuit
            rho_evolved = self.vqa._apply_circuit(rho, theta, t_idx)

            # Apply noise (تعداد gates تقریبی)
            n_gates = self.vqa.depth * self.vqa.n_sites
            rho_noisy = self.noise.apply_noise_to_state(
                rho_evolved,
                gate_type='single',
                n_gates=n_gates
            )

            # Low-rank truncation
            rho_lowrank = self.vqa._lowrank_projection(rho_noisy)

            # Shot noise
            rho_shot = self.vqa._add_shot_noise(rho_lowrank)

            # Post-processing
            rho_final = self.vqa._postprocess(rho_shot)

            approx_states.append(rho_final)
            rho = rho_final

        return approx_states


# تست
if __name__ == "__main__":
    print("Testing Noise Models...")

    # Test state
    rho = basis(5, 0) * basis(5, 0).dag()

    # Ideal
    noise_ideal = NISQNoiseModel(backend='ideal')
    rho_ideal = noise_ideal.apply_noise_to_state(rho, 'single', n_gates=10)
    print(f"\nIdeal fidelity: {fidelity(rho, rho_ideal)**2:.6f}")

    # IBM Heron
    noise_heron = NISQNoiseModel(backend='ibm_heron')
    rho_heron = noise_heron.apply_noise_to_state(rho, 'single', n_gates=10)
    print(f"Heron fidelity: {fidelity(rho, rho_heron)**2:.6f}")

    print(f"\nExpected fidelity for depth=6: {noise_heron.get_effective_fidelity_reduction(6):.4f}")
