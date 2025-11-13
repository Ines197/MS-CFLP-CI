from bestsolutions import BestSolutions
from greedyeffective import GreedyEffectiveSolver
from solution import Solution
from solver import Solver
from instance import parse_instance
import csv
import time
import math


class Runner:
    def __init__(self, reference_file: str, instances_folder: str, cpu_adjustment: float = 2.5):
        """
        Args:
            reference_file: Path to CSV with reference results
            instances_folder: Path to folder with .dzn instance files
            cpu_adjustment: Multiplier for timeout (default 2.5x for i5-1135G7 vs Threadripper)
        """
        self.reference_file = reference_file
        self.instances_folder = instances_folder
        self.cpu_adjustment = cpu_adjustment
        self.reference_results = self._load_reference_results()

    def _load_reference_results(self) -> dict:
        reference = {}
        with open(self.reference_file, mode="r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                inst = row["inst"]
                ref_min = float(row["min"])
                ref_avg = float(row["avg"])
                reference[inst] = (ref_min, ref_avg)
        return reference

    def _get_timeout(self, num_facilities: int, mode: str = 'competition', adjust: bool = True) -> float:
        """
        Calculate timeout based on paper's methodology.

        Args:
            num_facilities: Number of facilities (J) in the instance
            mode: 'competition' (10√J) or 'linear' (J)
            adjust: Whether to apply CPU speed adjustment

        Returns:
            Timeout in seconds
        """
        if mode == 'competition':
            base_timeout = 10 * math.sqrt(num_facilities)
        elif mode == 'linear':
            base_timeout = num_facilities
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'competition' or 'linear'")

        if adjust:
            return base_timeout * self.cpu_adjustment
        return base_timeout

    def profile(fnc):
        def inner(*args, **kwargs):
            import cProfile, pstats, io
            pr = cProfile.Profile()
            pr.enable()
            retval = fnc(*args, **kwargs)
            pr.disable()
            s = io.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())
            return retval

        return inner

    @profile
    def run_all(self, output_file: str = "comparison.csv",
                num_runs: int = 10,
                timeout_mode: str = 'competition',
                adjust_for_cpu: bool = True,
                top_k: int = 10):
        """
        Run solver for all instances with time-based limits (like in the paper).

        Args:
            output_file: CSV file to save results
            num_runs: Number of independent runs (paper uses 10)
            timeout_mode: 'competition' (10√J) or 'linear' (J seconds)
            adjust_for_cpu: Whether to adjust timeout for CPU speed difference
            top_k: Keep best K solutions for averaging
        """
        with open(output_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "inst", "J", "best_cost", "avg_bestK", "ref_min", "ref_avg",
                "gap_min(%)", "gap_avg(%)", "timeout(s)", "actual_time(s)", "iterations"
            ])

            for inst_name, (ref_min, ref_avg) in self.reference_results.items():
                filename = f"{self.instances_folder}/{inst_name}.dzn"
                problem = parse_instance(filename)

                num_facilities = len(list(problem.facilities.all()))
                timeout = self._get_timeout(num_facilities, timeout_mode, adjust_for_cpu)

                print(f"\n{'=' * 70}")
                print(f"Instance: {inst_name}")
                print(f"Facilities: {num_facilities}, Timeout: {timeout:.1f}s")
                print(f"{'=' * 70}")

                best_solutions = BestSolutions(top_k)
                total_iterations = 0

                run_start = time.time()

                for run_num in range(num_runs):
                    problem.reset()

                    # Time limit for this single run
                    run_timeout = timeout / num_runs  # Divide total time among runs
                    run_deadline = time.time() + run_timeout

                    iterations_this_run = 0

                    # Keep running solver until time limit
                    while time.time() < run_deadline:
                        problem.reset()
                        solver = GreedyEffectiveSolver(
                            problem,
                            Solution(problem),
                            rng_seed=int(time.time() * 1000 + run_num)
                        )
                        solver.solve_greedy_global_effective()

                        if solver.solution.is_valid():
                            best_solutions.add(solver.solution)

                        iterations_this_run += 1
                        total_iterations += 1

                    print(f"  Run {run_num + 1}/{num_runs}: {iterations_this_run} iterations")

                run_end = time.time()
                actual_time = run_end - run_start

                if len(best_solutions) == 0:
                    print(f"[WARN] No valid solutions for {inst_name}")
                    continue

                best_cost = best_solutions.best().total_cost()
                avg_best = sum(sol.total_cost() for sol in best_solutions.get_solutions()) / len(best_solutions)

                gap_min = (best_cost - ref_min) / ref_min * 100
                gap_avg = (avg_best - ref_avg) / ref_avg * 100

                print(f"\n📊 Results:")
                print(f"  Best cost: {best_cost:,.2f} (ref: {ref_min:,.2f})")
                print(f"  Avg top-{top_k}: {avg_best:,.2f} (ref: {ref_avg:,.2f})")
                print(f"  Gap min: {gap_min:+.2f}%")
                print(f"  Gap avg: {gap_avg:+.2f}%")
                print(f"  Total iterations: {total_iterations}")

                writer.writerow([
                    inst_name,
                    num_facilities,
                    round(best_cost, 2),
                    round(avg_best, 2),
                    ref_min,
                    ref_avg,
                    round(gap_min, 2),
                    round(gap_avg, 2),
                    round(timeout, 1),
                    round(actual_time, 1),
                    total_iterations
                ])

        print(f"\n[INFO] Results saved to {output_file}")

    @profile
    def run_one(self, instance_name: str = "wlp01",
                output_file: str = "comparison_single.csv",
                timeout_mode: str = 'competition',
                adjust_for_cpu: bool = True,
                num_runs: int = 10,
                top_k: int = 10):
        """
        Run solver for a single instance with time-based limit.

        Args:
            instance_name: Name of instance (e.g., 'wlp01')
            output_file: CSV file to save results
            timeout_mode: 'competition' (10√J) or 'linear' (J seconds)
            adjust_for_cpu: Whether to adjust timeout for CPU speed difference
            num_runs: Number of independent runs (paper uses 10)
            top_k: Keep best K solutions for averaging
        """
        if instance_name not in self.reference_results:
            print(f"[ERROR] Instance {instance_name} not in reference results")
            return

        ref_min, ref_avg = self.reference_results[instance_name]
        filename = f"{self.instances_folder}/{instance_name}.dzn"
        problem = parse_instance(filename)

        num_facilities = len(list(problem.facilities.all()))
        timeout = self._get_timeout(num_facilities, timeout_mode, adjust_for_cpu)

        best_solutions = BestSolutions(top_k)
        total_iterations = 0
        costs_per_run = []

        overall_start = time.time()

        for run_num in range(num_runs):
            problem.reset()

            # Time limit for this single run
            run_timeout = timeout / num_runs
            run_deadline = time.time() + run_timeout

            iterations_this_run = 0
            best_cost_this_run = float('inf')

            # Keep running solver until time limit
            while time.time() < run_deadline:
                problem.reset()
                solver = GreedyEffectiveSolver(
                    problem,
                    Solution(problem),
                    rng_seed=int(time.time() * 1000000 + run_num * 1000 + iterations_this_run)
                )
                solver.solve_greedy_global_effective()

                if solver.solution.is_valid():
                    cost = solver.solution.total_cost()
                    best_solutions.add(solver.solution)
                    best_cost_this_run = min(best_cost_this_run, cost)

                iterations_this_run += 1
                total_iterations += 1

            costs_per_run.append(best_cost_this_run)
            print(f"Run {run_num + 1:2d}/{num_runs}: {iterations_this_run:4d} iterations, "
                  f"best: {best_cost_this_run:,.2f}")

        overall_end = time.time()
        actual_time = overall_end - overall_start

        if len(best_solutions) == 0:
            print(f"\n[ERROR] No valid solutions found!")
            return

        best_cost = best_solutions.best().total_cost()
        avg_best = sum(sol.total_cost() for sol in best_solutions.get_solutions()) / len(best_solutions)
        avg_per_run = sum(costs_per_run) / len(costs_per_run)

        gap_min = (best_cost - ref_min) / ref_min * 100
        gap_avg = (avg_best - ref_avg) / ref_avg * 100

        print(f"\n{'=' * 70}")
        print(f"📊 FINAL RESULTS FOR {instance_name}")
        print(f"{'=' * 70}")
        print(f"Best solution found:     {best_cost:>12,.2f}")
        print(f"Avg of top-{top_k:2d}:          {avg_best:>12,.2f}")
        print(f"Avg across runs:         {avg_per_run:>12,.2f}")
        print(f"")
        print(f"Reference min (paper):   {ref_min:>12,.2f}")
        print(f"Reference avg (paper):   {ref_avg:>12,.2f}")
        print(f"")
        print(f"Gap vs ref min:          {gap_min:>11.2f}%")
        print(f"Gap vs ref avg:          {gap_avg:>11.2f}%")
        print(f"")
        print(f"Total iterations:        {total_iterations:>12,d}")
        print(f"Iterations per second:   {total_iterations / actual_time:>12,.1f}")
        print(f"Actual time:             {actual_time:>12.1f}s")
        print(f"{'=' * 70}\n")

        if best_cost < ref_min:
            print("✅ Our solution is BETTER than reference min!")
        elif best_cost > ref_min:
            print(f"❌ Our solution is {gap_min:.2f}% worse than reference min.")
        else:
            print("➖ Our solution equals reference min.")

        # Save results
        with open(output_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "inst", "J", "best_cost", "avg_bestK", "avg_per_run",
                "ref_min", "ref_avg", "gap_min(%)", "gap_avg(%)",
                "timeout(s)", "actual_time(s)", "iterations", "iter_per_sec"
            ])
            writer.writerow([
                instance_name,
                num_facilities,
                round(best_cost, 2),
                round(avg_best, 2),
                round(avg_per_run, 2),
                ref_min,
                ref_avg,
                round(gap_min, 2),
                round(gap_avg, 2),
                round(timeout, 1),
                round(actual_time, 1),
                total_iterations,
                round(total_iterations / actual_time, 1)
            ])

        print(f"[INFO] Results saved to {output_file}")