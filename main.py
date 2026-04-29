from runner import Runner

if __name__ == "__main__":
    runner = Runner(
        reference_file="references2.csv",
        instances_folder="InstancesDoha"
    )

    runner.compare_instance(instance_name="doha_1", time_limit=100.0)