import csv
import time
from typing import List, Dict, Any
import numpy as np
import matplotlib as plt
import pandas as pd

from bestsolutions import BestSolutions
from greedyeffective import GreedyEffectiveSolver
from solution import Solution
from instance import parse_instance
from greedymultifacility import GreedyMultiFacilitySolver
from extendedrankgreedy import ExtendedRankGreedySolver


class Runner:
    def __init__(self, reference_file: str, instances_folder: str):
        self.reference_file = reference_file
        self.instances_folder = instances_folder
        self.reference_results = self._load_reference_results()
        self.tau_cache = {}

    def _load_reference_results(self) -> dict:
        reference = {}
        try:
            with open(self.reference_file, mode="r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    inst = row.get("inst", row.get("instance", ""))
                    ref_min = float(row.get("min", row.get("best", 0)))
                    ref_avg = float(row.get("avg", 0))
                    reference[inst] = (ref_min, ref_avg)
        except Exception as e:
            print(f"[ERROR] Could not load reference file: {e}")
        return reference

    def _compute_and_cache_tau(self, instance_name: str, problem) -> float:
        if instance_name not in self.tau_cache:
            temp_solver = GreedyEffectiveSolver(problem, Solution(problem))
            temp_solver.precompute_effective_costs()
            tau = temp_solver.compute_adaptive_tau()
            self.tau_cache[instance_name] = tau
        return self.tau_cache[instance_name]

    def compare_instance(self,
                         instance_name: str,
                         time_limit: float = None,
                         iteration_limit: int = None,
                         include_all_heuristics: bool = True,
                         output_csv: str = "comparison_results_final.csv"):

        filename = f"{self.instances_folder}/{instance_name}.dzn"
        try:
            problem = parse_instance(filename)
        except FileNotFoundError:
            print(f"[ERROR] Instance file not found: {filename}")
            return

        num_facilities = getattr(problem, 'num_facilities', 0)
        ref_min, ref_avg = self.reference_results.get(instance_name, (0.0, 0.0))
        tau = self._compute_and_cache_tau(instance_name, problem)

        print(f"\n{'=' * 100}")
        print(
            f"INSTANCE: {instance_name} (Size: {num_facilities}) | Time Lim: {time_limit}s | Iter Lim: {iteration_limit}")
        print(f"{'=' * 100}")

        alg_configs = []
        alg_configs.append({
            'name': "MF Basic",
            'class': GreedyMultiFacilitySolver,
            'method': 'solve_greedy_multiple_facility',
            'kwargs': {},
            'run_args': {'mode': 'basic'}
        })
        alg_configs.append({
            'name': "MF Global",
            'class': GreedyMultiFacilitySolver,
            'method': 'solve_greedy_multiple_facility',
            'kwargs': {},
            'run_args': {'mode': 'global'}
        })
        alg_configs.append({
            'name': "MF Local (5)",
            'class': GreedyMultiFacilitySolver,
            'method': 'solve_greedy_multiple_facility',
            'kwargs': {},
            'run_args': {'mode': 'local', 'k': 5}
        })
        alg_configs.append({
            'name': "Ext. AOC (1)",
            'class': ExtendedRankGreedySolver,
            'method': 'solve_extended_greedy',
            'kwargs': {'heuristic_mode': 1, 'rank_cutoff_X': 0.2},
            'run_args': {}
        })

        if include_all_heuristics:
            alg_configs.append({
                'name': "Ext. Trans (2)",
                'class': ExtendedRankGreedySolver,
                'method': 'solve_extended_greedy',
                'kwargs': {'heuristic_mode': 2, 'rank_cutoff_X': 0.2},
                'run_args': {}
            })
            alg_configs.append({
                'name': "Ext. Top5 (3)",
                'class': ExtendedRankGreedySolver,
                'method': 'solve_extended_greedy',
                'kwargs': {'heuristic_mode': 3, 'rank_cutoff_X': 0.2},
                'run_args': {}
            })

        results_data = {}
        results_data["Reference"] = {
            'facility_count': num_facilities,
            'best_cost': ref_min,
            'avg_cost': ref_avg,
            'iterations': 0,
            'avg_time_per_iter': 0
        }

        for config in alg_configs:
            name = config['name']
            print(f"Running {name}...", end=" ", flush=True)

            best_solutions = BestSolutions(10)
            total_iters = 0
            start_t = time.time()
            rng_seed = 42

            while True:
                elapsed = time.time() - start_t
                if time_limit and elapsed >= time_limit:
                    break
                if iteration_limit and total_iters >= iteration_limit:
                    break
                if not time_limit and not iteration_limit and total_iters >= 1:
                    break

                problem.reset()
                solver = config['class'](
                    problem,
                    Solution(problem),
                    rng_seed=rng_seed + total_iters,
                    **config['kwargs']
                )
                method = getattr(solver, config['method'])
                method(**config['run_args'])

                if solver.solution.is_valid():
                    best_solutions.add(solver.solution)
                total_iters += 1

            actual_time = time.time() - start_t
            avg_time_per_iter = actual_time / total_iters if total_iters > 0 else 0

            print(f"Done ({total_iters} iters in {actual_time:.2f}s)")

            if len(best_solutions) > 0:
                best_cost = best_solutions.best().total_cost()
                avg_cost = sum(s.total_cost() for s in best_solutions.get_solutions()) / len(best_solutions)
            else:
                best_cost = float('inf')
                avg_cost = float('inf')

            results_data[name] = {
                'facility_count': num_facilities,
                'best_cost': best_cost,
                'avg_cost': avg_cost,
                'iterations': total_iters,
                'avg_time_per_iter': avg_time_per_iter
            }

        alg_names = ["Reference"] + [cfg['name'] for cfg in alg_configs]
        rows_definitions = [
            ('Facility Count', 'facility_count', '{:d}'),
            ('Best Cost', 'best_cost', '{:,.2f}'),
            ('Avg Cost', 'avg_cost', '{:,.2f}'),
            ('Iterations', 'iterations', '{:,d}'),
            ('Avg Time/Iter', 'avg_time_per_iter', '{:.6f}')
        ]

        header_str = f"{'Metric':<15}"
        for alg in alg_names:
            header_str += f" | {alg:<12}"
        print("-" * len(header_str))
        print(header_str)
        print("-" * len(header_str))

        for row_label, data_key, fmt in rows_definitions:
            row_str = f"{row_label:<15}"
            for alg in alg_names:
                val = results_data[alg].get(data_key, 0)
                if alg == "Reference" and data_key in ['iterations', 'avg_time_per_iter']:
                    val_str = " - "
                else:
                    val_str = fmt.format(val)
                row_str += f" | {val_str:>12}"
            print(row_str)
        print("-" * len(header_str))

        file_exists = False
        try:
            with open(output_csv, 'r') as f:
                file_exists = True
        except FileNotFoundError:
            pass

        with open(output_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                headers = ["Instance", "Metric"] + alg_names
                writer.writerow(headers)
            for row_label, data_key, fmt in rows_definitions:
                row_data = [instance_name, row_label]
                for alg in alg_names:
                    val = results_data[alg].get(data_key, 0)
                    if alg == "Reference" and data_key in ['iterations', 'avg_time_per_iter']:
                        row_data.append("-")
                    else:
                        row_data.append(val)
                writer.writerow(row_data)
            writer.writerow([])

    def run_single_instance(self, instance_name: str, time_limit: float = 60.0, output_csv: str = None):
        if output_csv is None:
            output_csv = f"results_{instance_name}.csv"

        print(f"\n[INFO] Starting single instance run: {instance_name}")

        self.compare_instance(
            instance_name=instance_name,
            time_limit=time_limit,
            include_all_heuristics=True,
            output_csv=output_csv
        )

        print(f"[SUCCESS] Results for {instance_name} saved to {output_csv}")

    def compare_all(self,
                    time_limit: float = None,
                    iteration_limit: int = None,
                    include_all_heuristics: bool = True,
                    output_csv: str = "comparison_results_final.csv"):

        for inst in self.reference_results.keys():
            self.compare_instance(
                instance_name=inst,
                time_limit=time_limit,
                iteration_limit=iteration_limit,
                include_all_heuristics=include_all_heuristics,
                output_csv=output_csv
            )

    def plot_performance(csv_file):
        df = pd.read_csv(csv_file)

        counts = df[df['Metric'] == 'Facility Count'].copy()
        times = df[df['Metric'] == 'Avg Time/Iter'].copy()

        algorithms = [col for col in df.columns if col not in ['Instance', 'Metric', 'Reference']]

        plt.figure(figsize=(10, 6))

        for alg in algorithms:
            plot_df = pd.DataFrame({
                'Size': pd.to_numeric(counts[alg], errors='coerce'),
                'Time': pd.to_numeric(times[alg], errors='coerce')
            })

            plot_df = plot_df.sort_values(by='Size')

            plt.plot(plot_df['Size'], plot_df['Time'], marker='o', label=alg, linewidth=2)

        plt.title('Algorithm Scalability: Average Time per Iteration', fontsize=14)
        plt.xlabel('Instance Size (Number of Facilities)', fontsize=12)
        plt.ylabel('Average Time per Iteration (Seconds)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.tight_layout()
        plt.savefig('scalability_plot.png')
        plt.show()
