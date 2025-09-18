from runner import Runner

if __name__ == "__main__":
    runner = Runner(
        reference_file="references.csv",
        instances_folder="Instances"
    )

    # Pokreni i snimi rezultate u novi csv
    runner.run_two(
        output_file="two_ref_solution.csv"
    )
