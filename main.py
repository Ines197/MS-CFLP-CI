from runner import Runner

if __name__ == "__main__":
    runner = Runner(
        reference_file="references2.csv",
        instances_folder="Instances"
    )

    # Pokreni i snimi rezultate u novi csv
    runner.compare_all()
