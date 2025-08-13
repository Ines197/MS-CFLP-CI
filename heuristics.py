import random
import sys

class Heuristics:
    def __init__(self, solver):
        self.solver = solver
        self.problem = solver.problem
        self.solution = solver.solution

    def rcl(self, facility, rcl_size):
        best_customers = []
        already_assigned = self.solution.get_assigned_customers_for_facility(facility.id)

        for cust in self.problem.customers.customers_with_unmet_demand():
            if self.solver.has_conflict(cust.id, already_assigned):
                continue

            # dodaj kupca
            best_customers.append(cust)

            # sortiraj po shipping cost
            best_customers.sort(key=lambda c: self.problem.shipping_costs[(c.id, facility.id)])

            # ako ima više od 5 kupaca, izbaci najskupljeg (poslednji u listi)
            if len(best_customers) > rcl_size:
                best_customers.pop()

        return best_customers
