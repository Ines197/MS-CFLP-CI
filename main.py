# main.py
import instance
from solver import Solver

from runner import Runner

if __name__ == "__main__":
    runner = Runner(
        reference_file="references.csv",
        instances_folder="Instances"
    )

    # Pokreni i snimi rezultate u novi csv
    runner.run_all(
        output_file="my_reference.csv",
        max_iterations=3  # koliko puta da radi GRASP
    )
