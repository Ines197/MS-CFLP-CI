from bestsolutions import BestSolutions
from solution import Solution
from solver import Solver
from instance import parse_instance
import csv, time

class Runner:
    def __init__(self, reference_file: str, instances_folder: str):
        self.reference_file = reference_file
        self.instances_folder = instances_folder
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
    def run_all(self, output_file: str = "comparison.csv", grasp_runs: int = 30, top_k: int = 5):
        """Pokreni solver za sve instance više puta i sačuvaj najboljih K rešenja."""
        with open(output_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["inst", "best_cost", "avg_bestK", "ref_min", "ref_avg",
                             "gap_min(%)", "gap_avg(%)", "time(s)"])

            for inst_name, (ref_min, ref_avg) in self.reference_results.items():
                filename = f"{self.instances_folder}/{inst_name}.dzn"
                problem = parse_instance(filename)

                best_solutions = BestSolutions(top_k)

                start = time.time()
                for _ in range(grasp_runs):
                    solver = Solver(problem)
                    solver.solve_grasp()
                    if solver.solution.is_valid():
                        best_solutions.add(solver.solution)
                end = time.time()

                if len(best_solutions) == 0:
                    print(f"[WARN] Nema validnih rešenja za {inst_name}")
                    continue

                best_cost = best_solutions.best().total_cost()
                avg_best = sum(sol.total_cost() for sol in best_solutions.get_solutions()) / len(best_solutions)

                gap_min = (best_cost - ref_min) / ref_min * 100
                gap_avg = (avg_best - ref_avg) / ref_avg * 100

                writer.writerow([
                    inst_name,
                    round(best_cost, 2),
                    round(avg_best, 2),
                    ref_min,
                    ref_avg,
                    round(gap_min, 2),
                    round(gap_avg, 2),
                    round(end - start, 3)
                ])

        print(f"[INFO] Rezultati su sačuvani u {output_file}")

    @profile
    def run_one(self, output_file: str = "comparison.csv", grasp_runs: int = 20, top_k: int = 5):
        """Pokreni solver samo za prvu instancu (wlp01) više puta i uporedi sa referentnim rešenjima."""
        wanted = ["wlp01"]  # samo prva instanca

        with open(output_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["inst", "best_cost", "avg_bestK", "ref_min", "ref_avg",
                             "gap_min(%)", "gap_avg(%)", "time(s)"])

            for inst_name in wanted:
                if inst_name not in self.reference_results:
                    print(f"[WARN] Instance {inst_name} nije u reference_results")
                    continue

                ref_min, ref_avg = self.reference_results[inst_name]
                filename = f"{self.instances_folder}/{inst_name}.dzn"
                problem = parse_instance(filename)

                best_solutions = BestSolutions(top_k)

                start = time.time()
                for _ in range(grasp_runs):
                    solver = Solver(problem)
                    solver.solve_grasp()
                    # ako želiš da preskoči nevalidna rešenja, odkomentariši sledeću liniju
                    # if solver.solution.is_valid():
                    best_solutions.add(solver.solution)
                end = time.time()

                best_grasp_solution = best_solutions.best()
                best_cost = best_grasp_solution.total_cost()
                avg_best = sum(sol.total_cost() for sol in best_solutions.get_solutions()) / len(best_solutions)

                print(f"\n=== Instance {inst_name} ===")
                print(f"Naše najbolje GRASP rešenje: {best_cost:.2f}")
                print(f"Referentno rešenje min: {ref_min:.2f}, avg: {ref_avg:.2f}")

                if best_cost < ref_min:
                    print("✅ Naše rešenje je BOLJE od referentnog min!")
                elif best_cost > ref_min:
                    print("❌ Naše rešenje je GORE od referentnog min.")
                else:
                    print("➖ Naše rešenje je jednako referentnom min.")

                gap_min = (best_cost - ref_min) / ref_min * 100
                gap_avg = (avg_best - ref_avg) / ref_avg * 100
                writer.writerow([
                    inst_name,
                    round(best_cost, 2),
                    round(avg_best, 2),
                    ref_min,
                    ref_avg,
                    round(gap_min, 2),
                    round(gap_avg, 2),
                    round(end - start, 3)
                ])

        print(f"[INFO] Rezultati za wlp01 su sačuvani u {output_file}")
