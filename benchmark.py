"""
benchmark.py
============
Statistical benchmarking با 50 independent trials
"""

import numpy as np
from scipy import stats
import pandas as pd
import time
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

class StatisticalBenchmark:
    """
    Run multiple trials و آنالیز آماری
    """

    def __init__(self, n_sites, n_trials=50, n_cores=None):
        """
        Args:
            n_sites: تعداد chromophores
            n_trials: تعداد trials (default: 50)
            n_cores: تعداد هسته‌ها برای parallelization (None = auto)
        """
        self.n_sites = n_sites
        self.n_trials = n_trials
        self.n_cores = n_cores or mp.cpu_count()

        self.results = []

    def run_trials(self, noisy=False, maxiter=500, verbose=True):
        """
        اجرای n_trials تست مستقل

        Args:
            noisy: اگر True، با noise model اجرا می‌شه
            maxiter: تعداد iterations هر trial
            verbose: نمایش progress

        Returns:
            results_dict: نتایج آماری
        """

        if verbose:
            print(f"\n{'='*70}")
            print(f"Statistical Benchmark: FMO-{self.n_sites}")
            print(f"{'='*70}")
            print(f"  Trials: {self.n_trials}")
            print(f"  Noisy: {noisy}")
            print(f"  Cores: {self.n_cores}")
            print(f"  Iterations per trial: {maxiter}")
            print(f"{'='*70}\n")

        start_time = time.time()

        # Parallel execution
        if self.n_cores > 1:
            with mp.Pool(processes=self.n_cores) as pool:
                func = partial(self._single_trial, noisy=noisy, maxiter=maxiter)
                results = list(tqdm(
                    pool.imap(func, range(self.n_trials)),
                    total=self.n_trials,
                    desc="Running trials",
                    disable=not verbose
                ))
        else:
            # Sequential
            results = []
            for seed in tqdm(range(self.n_trials), desc="Running trials", disable=not verbose):
                results.append(self._single_trial(seed, noisy, maxiter))

        elapsed = time.time() - start_time

        # Extract data
        fidelities = [r['mean_fidelity'] for r in results]
        computation_times = [r['computation_time'] for r in results]
        convergence_iters = [r.get('n_iterations', maxiter) for r in results]

        # Statistical analysis
        stats_dict = self._compute_statistics(fidelities)

        # ANOVA if multiple groups (placeholder)
        stats_dict['anova_p_value'] = self._compute_anova([fidelities])

        # Add metadata
        stats_dict.update({
            'n_sites': self.n_sites,
            'n_trials': self.n_trials,
            'noisy': noisy,
            'all_fidelities': fidelities,
            'all_times': computation_times,
            'all_iterations': convergence_iters,
            'total_benchmark_time': elapsed,
            'raw_results': results
        })

        if verbose:
            self._print_summary(stats_dict)

        self.results = stats_dict
        return stats_dict

    def _single_trial(self, seed, noisy, maxiter):
        """
        یک trial مستقل
        """
        from fmo_hamiltonian import FMOHamiltonian
        from exact_solver import ExactLindbladSolver
        from lowrank_vqa import LowRankVQA
        from noise_models import NISQNoiseModel, NoiseSimulator

        # Set seed برای reproducibility
        np.random.seed(seed)

        # Setup
        H, E, J = FMOHamiltonian.get_hamiltonian(self.n_sites)
        c_ops = FMOHamiltonian.get_lindblad_operators(self.n_sites, gamma=0.1)

        # Exact solution
        times = np.linspace(0, 500e-15, 200)
        exact_solver = ExactLindbladSolver(H, c_ops)
        exact_result = exact_solver.evolve(times, progress=False)

        # VQA
        vqa = LowRankVQA(
            H, c_ops,
            exact_result['states'],
            times,
            rank=10,
            depth=max(6, self.n_sites // 2),
            shots=3000
        )

        # Run with or without noise
        if noisy:
            noise_model = NISQNoiseModel(backend='ibm_heron')
            simulator = NoiseSimulator(vqa, noise_model)
            result = simulator.run_noisy_optimization(maxiter=maxiter, verbose=False)
        else:
            result = vqa.optimize(maxiter=maxiter, verbose=False)

        return {
            'seed': seed,
            'mean_fidelity': result['mean_fidelity'],
            'computation_time': result['computation_time'],
            'n_iterations': len(result['cost_history']),
            'final_cost': result['cost_history'][-1] if result['cost_history'] else np.nan
        }

    def _compute_statistics(self, data):
        """
        محاسبه آمار توصیفی
        """
        data = np.array(data)

        mean = np.mean(data)
        std = np.std(data, ddof=1)
        se = std / np.sqrt(len(data))
        ci_95 = 1.96 * se

        return {
            'mean': mean,
            'std': std,
            'se': se,
            'ci_95': ci_95,
            'ci_95_lower': mean - ci_95,
            'ci_95_upper': mean + ci_95,
            'median': np.median(data),
            'min': np.min(data),
            'max': np.max(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75)
        }

    def _compute_anova(self, groups):
        """
        One-way ANOVA برای مقایسه groups
        """
        if len(groups) < 2:
            return np.nan

        try:
            F_stat, p_value = stats.f_oneway(*groups)
            return p_value
        except:
            return np.nan

    def _print_summary(self, stats_dict):
        """
        چاپ خلاصه نتایج
        """
        print(f"\n{'='*70}")
        print(f"✓ Benchmark Completed")
        print(f"{'='*70}")
        print(f"  Mean Fidelity: {stats_dict['mean']:.6f}")
        print(f"  Std Deviation: {stats_dict['std']:.6f}")
        print(f"  95% CI: [{stats_dict['ci_95_lower']:.6f}, {stats_dict['ci_95_upper']:.6f}]")
        print(f"  Range: [{stats_dict['min']:.6f}, {stats_dict['max']:.6f}]")
        print(f"  Total Time: {stats_dict['total_benchmark_time']:.2f} seconds")
        print(f"  Time per Trial: {stats_dict['total_benchmark_time']/self.n_trials:.2f} seconds")
        print(f"{'='*70}\n")

    def export_to_excel(self, filename='benchmark_results.xlsx'):
        """
        Export نتایج به Excel
        """
        if not self.results:
            print("No results to export. Run benchmark first.")
            return

        # DataFrame اصلی
        df_main = pd.DataFrame({
            'Trial': range(self.n_trials),
            'Fidelity': self.results['all_fidelities'],
            'Time_seconds': self.results['all_times'],
            'Iterations': self.results['all_iterations']
        })

        # Summary statistics
        df_stats = pd.DataFrame({
            'Metric': ['Mean', 'Std', 'SE', '95% CI', 'Median', 'Min', 'Max'],
            'Value': [
                self.results['mean'],
                self.results['std'],
                self.results['se'],
                self.results['ci_95'],
                self.results['median'],
                self.results['min'],
                self.results['max']
            ]
        })

        # Export
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df_main.to_excel(writer, sheet_name='Trial_Results', index=False)
            df_stats.to_excel(writer, sheet_name='Statistics', index=False)

        print(f"✓ Results exported to {filename}")


def run_comprehensive_benchmark(systems=[5, 7, 8], n_trials=50, noisy=False):
    """
    اجرای benchmark برای چند سیستم

    Args:
        systems: لیست system sizes
        n_trials: تعداد trials هر سیستم
        noisy: اگر True، با noise اجرا می‌شه

    Returns:
        all_results: dict شامل نتایج همه سیستم‌ها
    """

    all_results = {}

    for n_sites in systems:
        print(f"\n{'#'*70}")
        print(f"# System: FMO-{n_sites}")
        print(f"{'#'*70}")

        benchmark = StatisticalBenchmark(n_sites, n_trials=n_trials)
        results = benchmark.run_trials(noisy=noisy, maxiter=500)

        # Export
        filename = f'FMO{n_sites}_{"noisy" if noisy else "noiseless"}_{n_trials}trials.xlsx'
        benchmark.export_to_excel(filename)

        all_results[f'FMO-{n_sites}'] = results

    return all_results


# تست
if __name__ == "__main__":
    print("Testing Statistical Benchmark...")

    # Quick test با 5 trials
    benchmark = StatisticalBenchmark(n_sites=5, n_trials=5, n_cores=2)
    results = benchmark.run_trials(noisy=False, maxiter=100)

    print("\n✓ Test completed")
