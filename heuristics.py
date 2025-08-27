from customer_rcl import CustomerRCL

class Heuristics:
    def __init__(self, solver):
        self.solver = solver
        self.problem = solver.problem
        self.solution = solver.solution
        self.customer_rcl = CustomerRCL(self.problem, self.solver)

    def rcl(self, facility, rcl_size):
        return self.customer_rcl.get_top(facility, rcl_size)