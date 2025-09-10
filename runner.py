import csv
import time
from instance import parse_instance
from solver import Solver

class Runner:
    def __init__(self, reference_file: str, instances_folder: str):
        self.reference_file = reference_file
        self.instances_folder = instances_folder
        self.reference_results = self._load_reference_results()

    def _load_reference_results(self) -> dict:
        """Učitaj csv fajl sa rezultatima iz rada (min i avg)."""
        reference = {}
        with open(self.reference_file, mode="r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                inst = row["inst"]
                ref_min = float(row["min"])
                ref_avg = float(row["avg"])
                reference[inst] = (ref_min, ref_avg)
        return reference

    def run_all(self, output_file: str = "comparison.csv", max_iterations: int = 50):
        """Pokreni solver za sve instance i snimi poređenje u CSV."""
        with open(output_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["inst", "my_cost", "ref_min", "ref_avg",
                             "gap_min(%)", "gap_avg(%)", "time(s)"])

            for inst_name, (ref_min, ref_avg) in self.reference_results.items():
                filename = f"{self.instances_folder}/{inst_name}.dzn"
                problem = parse_instance(filename)

                solver = Solver(problem)

                start = time.time()
                solver.solve_grasp()
                cost = solver.solution.total_cost()
                end = time.time()

                gap_min = (cost - ref_min) / ref_min * 100
                gap_avg = (cost - ref_avg) / ref_avg * 100

                writer.writerow([
                    inst_name,
                    round(cost, 2),
                    ref_min,
                    ref_avg,
                    round(gap_min, 2),
                    round(gap_avg, 2),
                    round(end - start, 3)
                ])

        print(f"[INFO] Rezultati su sačuvani u {output_file}")
