import csv
import time
from typing import List, Dict, Any
import numpy as np

from bestsolutions import BestSolutions
from greedyeffective import GreedyEffectiveSolver
from solution import Solution
from instance import parse_instance
from greedymultifacility import GreedyMultiFacilitySolver
from extendedrankgreedy import ExtendedRankGreedySolver


class Runner:
    def __init__(self, reference_file: str, instances_folder: str):
        """
        Args:
            reference_file: Path to CSV with reference results (references2.csv)
            instances_folder: Path to folder with .dzn instance files
        """
        self.reference_file = reference_file
        self.instances_folder = instances_folder
        self.reference_results = self._load_reference_results()
        self.tau_cache = {}

    def _load_reference_results(self) -> dict:
        """Loads reference Best Known values."""
        reference = {}
        try:
            with open(self.reference_file, mode="r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Adjust column names based on your actual CSV format
                    # Assuming keys are 'inst', 'min' (or 'best'), 'avg'
                    inst = row.get("inst", row.get("instance", ""))
                    ref_min = float(row.get("min", row.get("best", 0)))
                    ref_avg = float(row.get("avg", 0))
                    reference[inst] = (ref_min, ref_avg)
        except Exception as e:
            print(f"[ERROR] Could not load reference file: {e}")
        return reference

    def _compute_and_cache_tau(self, instance_name: str, problem) -> float:
        """Computes tau once per instance for the Effective cost heuristics."""
        if instance_name not in self.tau_cache:
            temp_solver = GreedyEffectiveSolver(problem, Solution(problem))
            temp_solver.precompute_effective_costs()
            tau = temp_solver.compute_adaptive_tau()
            self.tau_cache[instance_name] = tau
        return self.tau_cache[instance_name]

    def compare_instance(self,
                         instance_name: str,
                         time_limit: float = 120.0,
                         include_all_heuristics: bool = True,
                         output_csv: str = "comparison_results5.csv"):
        """
        Runs comparison for a single instance.

        Args:
            instance_name: e.g., "wlp01"
            time_limit: Time in seconds (default 300s = 5mins)
            include_all_heuristics: If True, runs Extended Greedy modes 1, 2, and 3.
                                    If False, only runs mode 1.
        """

        # 1. Setup Instance
        filename = f"{self.instances_folder}/{instance_name}.dzn"
        try:
            problem = parse_instance(filename)
        except FileNotFoundError:
            print(f"[ERROR] Instance file not found: {filename}")
            return

        ref_min, ref_avg = self.reference_results.get(instance_name, (0.0, 0.0))
        tau = self._compute_and_cache_tau(instance_name, problem)

        print(f"\n{'=' * 100}")
        print(f"INSTANCE: {instance_name} | Limit: {time_limit}s | Ref Best: {ref_min}")
        print(f"{'=' * 100}")

        # 2. Define Algorithms to Run
        # Structure: (DisplayName, SolverClass, MethodName, InitKwargs, RunKwargs)

        alg_configs = []

        # A. Greedy Effective
        alg_configs.append({
            'name': "Global Eff.",
            'class': GreedyEffectiveSolver,
            'method': 'solve_greedy_global_effective',
            'kwargs': {},
            'run_args': {'tau': tau}
        })

        # B. Multi Facility
        alg_configs.append({
            'name': "Multi Fac.",
            'class': GreedyMultiFacilitySolver,
            'method': 'solve_greedy_multiple_facility',
            'kwargs': {},
            'run_args': {}
        })

        # C. Extended Greedy (Mode 1)
        alg_configs.append({
            'name': "Ext. AOC (1)",
            'class': ExtendedRankGreedySolver,
            'method': 'solve_extended_greedy',
            'kwargs': {'heuristic_mode': 1, 'rank_cutoff_X': 0.2},
            'run_args': {}
        })

        # D. Optional Extended Modes
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

        # 3. Run Algorithms
        results_data = {}  # Key: Alg Name, Value: Dict of metrics

        # Add Reference Data manually
        results_data["Reference"] = {
            'best_cost': ref_min,
            'avg_cost': ref_avg,
            'iterations': 0,
            #'time': 0,
            #'gap': 0.0
        }

        for config in alg_configs:
            name = config['name']
            print(f"Running {name}...", end=" ", flush=True)

            # Run Logic
            best_solutions = BestSolutions(10)
            total_iters = 0
            start_t = time.time()
            deadline = start_t + time_limit

            # Simple seed generation
            rng_seed = 42

            while time.time() < deadline:
                problem.reset()

                # Instantiate
                solver = config['class'](
                    problem,
                    Solution(problem),
                    rng_seed=rng_seed + total_iters,
                    **config['kwargs']
                )

                # Run method
                method = getattr(solver, config['method'])
                method(**config['run_args'])

                if solver.solution.is_valid():
                    best_solutions.add(solver.solution)

                total_iters += 1

            actual_time = time.time() - start_t
            print(f"Done ({total_iters} iters)")

            # Process Results
            if len(best_solutions) > 0:
                best_cost = best_solutions.best().total_cost()
                avg_cost = sum(s.total_cost() for s in best_solutions.get_solutions()) / len(best_solutions)
                # Calculate Gap % vs Reference Min
                #gap = ((best_cost - ref_min) / ref_min * 100) if ref_min > 0 else 0.0
            else:
                best_cost = float('inf')
                avg_cost = float('inf')
                #gap = float('inf')

            results_data[name] = {
                'best_cost': best_cost,
                'avg_cost': avg_cost,
                'iterations': total_iters,
                #'time': actual_time,
                #'gap': gap
            }

        # 4. Format Output (Rows = Metrics, Cols = Algs)

        # Define the column order
        alg_names = ["Reference"] + [cfg['name'] for cfg in alg_configs]

        # Define rows to display
        rows_definitions = [
            ('Best Cost', 'best_cost', '{:,.2f}'),
            #('Gap (%)', 'gap', '{:+.2f}%'),
            ('Avg Cost', 'avg_cost', '{:,.2f}'),
            ('Iterations', 'iterations', '{:,d}'),
            #('Time (s)', 'time', '{:.1f}')
        ]

        # -- Print to Console --
        # Header
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

                # Handle Reference special cases (Time/Iter is N/A)
                if alg == "Reference" and data_key in ['time', 'iterations']:
                    val_str = " - "
                elif alg == "Reference" and data_key == 'gap':
                    val_str = " 0.00% "
                else:
                    val_str = fmt.format(val)

                row_str += f" | {val_str:>12}"
            print(row_str)
        print("-" * len(header_str))

        # -- Save to CSV (Transposed) --
        # We append to the file so we can run multiple instances in a loop outside
        file_exists = False
        try:
            with open(output_csv, 'r') as f:
                file_exists = True
        except FileNotFoundError:
            pass

        with open(output_csv, mode='a', newline='') as f:
            writer = csv.writer(f)

            # If new file, write header with first column being Instance + Metric
            if not file_exists:
                headers = ["Instance", "Metric"] + alg_names
                writer.writerow(headers)

            # Write data
            for row_label, data_key, fmt in rows_definitions:
                row_data = [instance_name, row_label]
                for alg in alg_names:
                    val = results_data[alg].get(data_key, 0)
                    if alg == "Reference" and data_key in ['time', 'iterations']:
                        row_data.append("-")
                    else:
                        row_data.append(val)
                writer.writerow(row_data)

            writer.writerow([])

    def compare_all(self, time_limit: float = 100.0, include_all_heuristics: bool = True):
        """Loop through all loaded reference instances."""
        for inst in self.reference_results.keys():
            self.compare_instance(inst, time_limit, include_all_heuristics)