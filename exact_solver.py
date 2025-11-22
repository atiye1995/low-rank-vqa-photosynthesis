"""
exact_solver.py
===============
حل دقیق Lindblad master equation با QuTiP
"""

import numpy as np
from qutip import *
import time
from tqdm import tqdm

class ExactLindbladSolver:
    """حل دقیق معادله Lindblad"""

    def __init__(self, hamiltonian, lindblad_ops, initial_state=None):
        """
        Args:
            hamiltonian: System Hamiltonian (Qobj)
            lindblad_ops: لیست Lindblad operators
            initial_state: Initial density matrix (اگر None باشه، |0><0| است)
        """
        self.H = hamiltonian
        self.c_ops = lindblad_ops
        self.n_sites = hamiltonian.dims[0][0]

        if initial_state is None:
            # Initial excitation on site 0
            self.rho0 = basis(self.n_sites, 0) * basis(self.n_sites, 0).dag()
        else:
            self.rho0 = initial_state

    def evolve(self, times, progress=True):
        """
        حل معادله Lindblad

        Args:
            times: آرایه زمان‌ها (به ثانیه)
            progress: نمایش progress bar

        Returns:
            states: لیست density matrices در زمان‌های مختلف
            populations: dict شامل population هر site
            fidelity_trace: trace fidelity در طول زمان
        """

        if progress:
            print(f"Solving Lindblad equation for {self.n_sites} sites, {len(times)} time steps...")

        start_time = time.time()

        # حل با mesolve
        # options برای دقت بالا
        opts = Options(
            atol=1e-8,
            rtol=1e-6,
            nsteps=50000
        )

        result = mesolve(
            self.H,
            self.rho0,
            times,
            self.c_ops,
            options=opts,
            progress_bar=progress
        )

        elapsed = time.time() - start_time

        if progress:
            print(f"✓ Solved in {elapsed:.2f} seconds")

        # محاسبه populations
        populations = self._compute_populations(result.states)

        # محاسبه fidelity با initial state
        fidelity_trace = [fidelity(state, self.rho0)**2 for state in result.states]

        return {
            'states': result.states,
            'populations': populations,
            'fidelity_trace': fidelity_trace,
            'computation_time': elapsed,
            'times': times
        }

    def _compute_populations(self, states):
        """محاسبه population هر site در طول زمان"""
        populations = {}

        for site in range(self.n_sites):
            projector = basis(self.n_sites, site) * basis(self.n_sites, site).dag()
            populations[f'site_{site}'] = [
                expect(projector, state) for state in states
            ]

        return populations

    def compute_transfer_time(self, populations, donor_site=0, acceptor_site=2, threshold=0.5):
        """
        محاسبه زمان transfer از donor به acceptor

        Args:
            populations: dict of populations
            donor_site: site اولیه
            acceptor_site: site نهایی
            threshold: آستانه برای تعریف "transfer شده" (0.5 = 50% population)

        Returns:
            transfer_time: زمان transfer (به ps)
        """
        acceptor_pop = np.array(populations[f'site_{acceptor_site}'])

        # پیدا کردن اولین زمانی که population > threshold
        idx = np.where(acceptor_pop > threshold)[0]

        if len(idx) > 0:
            transfer_time = self.times[idx[0]]
            return transfer_time * 1e12  # به picosecond
        else:
            return np.nan


# تست
if __name__ == "__main__":
    from fmo_hamiltonian import FMOHamiltonian

    print("Testing Exact Lindblad Solver...")

    # FMO-7
    H, E, J = FMOHamiltonian.get_hamiltonian(7)
    c_ops = FMOHamiltonian.get_lindblad_operators(7, gamma=0.1)

    # Time points (0-500 fs)
    times = np.linspace(0, 500e-15, 200)

    # حل
    solver = ExactLindbladSolver(H, c_ops)
    result = solver.evolve(times)

    print(f"\n✓ Computation completed")
    print(f"  Final state trace: {result['states'][-1].tr():.6f}")
    print(f"  Site 0 population (t=0): {result['populations']['site_0'][0]:.4f}")
    print(f"  Site 0 population (t=500fs): {result['populations']['site_0'][-1]:.4f}")
