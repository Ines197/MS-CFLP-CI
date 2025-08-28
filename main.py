# main.py
import instance
from solver import Solver

def main():
    problem = instance.parse_instance("test.dzn")

    s = Solver(problem)

    s.solve_greedy()

    s.solution.print_solution()

    s.solve_local_search()

    s.solution.print_solution()
if __name__ == "__main__":
    main()
