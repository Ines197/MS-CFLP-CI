from runner import Runner

if __name__ == "__main__":
    runner = Runner(
        reference_file="references2.csv",
        instances_folder="Instances"
    )

    runner.compare_all(None, 100)