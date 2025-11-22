"""
fmo_hamiltonian.py
==================
FMO complex Hamiltonians با پارامترهای تجربی
References:
- Adolphs & Renger, Biophys. J. 2006
- Schmidt am Busch et al., J. Phys. Chem. Lett. 2011
"""

import numpy as np
from qutip import *

class FMOHamiltonian:
    """FMO complex Hamiltonian generator"""

    # پارامترهای تجربی (cm^-1)
    FMO_PARAMS = {
        5: {
            'energies': np.array([12410, 12530, 12210, 12320, 12480]),
            'couplings': np.array([
                [0,    -87.7, 5.5,   -5.9,  6.7],
                [-87.7, 0,    30.8,  8.2,   0.7],
                [5.5,   30.8, 0,     -53.5, -2.2],
                [-5.9,  8.2,  -53.5, 0,     -70.7],
                [6.7,   0.7,  -2.2,  -70.7, 0]
            ])
        },
        7: {
            'energies': np.array([12410, 12530, 12210, 12320, 12480, 12630, 12440]),
            'couplings': np.array([
                [0,    -87.7, 5.5,   -5.9,  6.7,   -13.7, -9.9],
                [-87.7, 0,    30.8,  8.2,   0.7,   11.4,  4.7],
                [5.5,   30.8, 0,     -53.5, -2.2,  -9.6,  6.0],
                [-5.9,  8.2,  -53.5, 0,     -70.7, -17.0, -63.3],
                [6.7,   0.7,  -2.2,  -70.7, 0,     81.1,  -1.3],
                [-13.7, 11.4, -9.6,  -17.0, 81.1,  0,     39.7],
                [-9.9,  4.7,  6.0,   -63.3, -1.3,  39.7,  0]
            ])
        },
        8: {
            'energies': np.array([12410, 12530, 12210, 12320, 12480, 12630, 12440, 12250]),
            'couplings': np.array([
                [0,    -87.7, 5.5,   -5.9,  6.7,   -13.7, -9.9,  -3.0],
                [-87.7, 0,    30.8,  8.2,   0.7,   11.4,  4.7,   5.0],
                [5.5,   30.8, 0,     -53.5, -2.2,  -9.6,  6.0,   -8.0],
                [-5.9,  8.2,  -53.5, 0,     -70.7, -17.0, -63.3, -1.0],
                [6.7,   0.7,  -2.2,  -70.7, 0,     81.1,  -1.3,  28.0],
                [-13.7, 11.4, -9.6,  -17.0, 81.1,  0,     39.7,  -5.0],
                [-9.9,  4.7,  6.0,   -63.3, -1.3,  39.7,  0,     32.0],
                [-3.0,  5.0,  -8.0,  -1.0,  28.0,  -5.0,  32.0,  0]
            ])
        }
    }

    @classmethod
    def get_hamiltonian(cls, n_sites):
        """
        ساخت Hamiltonian برای n_sites chromophore

        Args:
            n_sites: تعداد chromophore ها (5, 7, 8, 10, 12)

        Returns:
            H: Qobj Hamiltonian
            energies: Site energies
            couplings: Coupling matrix
        """

        if n_sites in cls.FMO_PARAMS:
            # استفاده از پارامترهای تجربی
            energies = cls.FMO_PARAMS[n_sites]['energies']
            couplings = cls.FMO_PARAMS[n_sites]['couplings']
        else:
            # Extrapolation برای سایزهای بزرگتر
            energies, couplings = cls._extrapolate_parameters(n_sites)

        # ساخت Hamiltonian
        H = cls._build_hamiltonian(energies, couplings)

        return H, energies, couplings

    @staticmethod
    def _build_hamiltonian(energies, couplings):
        """ساخت Hamiltonian matrix"""
        n = len(energies)
        H = 0

        # Site energies (diagonal)
        for i in range(n):
            H += energies[i] * basis(n, i) * basis(n, i).dag()

        # Couplings (off-diagonal)
        for i in range(n):
            for j in range(i+1, n):
                if np.abs(couplings[i, j]) > 0.1:  # فقط couplingهای قوی
                    H += couplings[i, j] * (
                        basis(n, i) * basis(n, j).dag() +
                        basis(n, j) * basis(n, i).dag()
                    )

        return H

    @staticmethod
    def _extrapolate_parameters(n_sites):
        """
        Extrapolation برای سایزهای بزرگتر (10, 12)
        بر اساس dipole-dipole interaction model
        """
        # Site energies با توزیع گاوسی
        mean_energy = 12400  # cm^-1
        std_energy = 120
        energies = mean_energy + std_energy * np.random.randn(n_sites)

        # Couplings بر اساس فاصله
        # فرض: chromophoreها روی یک شبکه منظم
        positions = np.linspace(0, n_sites-1, n_sites)
        couplings = np.zeros((n_sites, n_sites))

        for i in range(n_sites):
            for j in range(i+1, n_sites):
                distance = np.abs(positions[i] - positions[j])
                # Dipole-dipole coupling: J ~ 1/r^3
                J_ij = 100 / (distance ** 3 + 0.5)  # +0.5 برای جلوگیری از divergence
                # اضافه کردن disorder
                J_ij *= (1 + 0.2 * np.random.randn())
                couplings[i, j] = J_ij
                couplings[j, i] = J_ij

        return energies, couplings

    @staticmethod
    def get_lindblad_operators(n_sites, gamma=0.1):
        """
        ساخت Lindblad operators برای pure dephasing

        Args:
            n_sites: تعداد sites
            gamma: dephasing rate (cm^-1)

        Returns:
            c_ops: لیست Lindblad operators
        """
        c_ops = []
        for i in range(n_sites):
            L_i = np.sqrt(gamma) * basis(n_sites, i) * basis(n_sites, i).dag()
            c_ops.append(L_i)

        return c_ops


# تست
if __name__ == "__main__":
    print("Testing FMO Hamiltonian...")

    for n in [5, 7, 8]:
        H, E, J = FMOHamiltonian.get_hamiltonian(n)
        print(f"\nFMO-{n}:")
        print(f"  Hamiltonian dimension: {H.dims}")
        print(f"  Site energies: {E}")
        print(f"  Average coupling: {np.mean(np.abs(J[J!=0])):.2f} cm^-1")

        # Eigenspectrum
        evals = H.eigenenergies()
        print(f"  Energy gap (min): {np.min(np.diff(evals)):.2f} cm^-1")
