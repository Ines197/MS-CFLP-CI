import sys
import problem
import solution
import random
import heuristics

class Solver:
    def __init__(self, problem_instance):
        self.heuristics = heuristics.Heuristics(self)
        self.problem = problem_instance
        self.solution = solution.Solution(problem_instance)

    def has_conflict(self, cust_id, already_assigned_ids):
        for other_cust_id in already_assigned_ids:
            if (cust_id, other_cust_id) in self.problem.incompatibilities or \
                    (other_cust_id, cust_id) in self.problem.incompatibilities:
                return True
        return False

    def _assign_customer_to_facility(self, customer, facility):
        assign_amount = min(facility.remaining_capacity, customer.remaining_demand)

        self.solution.add_assignment(customer.id, facility.id, assign_amount)

        facility.remaining_capacity -= assign_amount
        customer.remaining_demand -= assign_amount

    def solve_greedy(self):
        sorted_facilities = self.problem.facilities.sort_by_cost_capacity_ratio()
        while sorted_facilities:
            if self.problem.customers.total_remaining_demand() == 0:
                break

            cheapest_facility = sorted_facilities.pop(0)
            cheapest_facility.open()

            while cheapest_facility.remaining_capacity:
                best_customers = self.heuristics.rcl(cheapest_facility, 5)
                if not best_customers:
                    break

                chosen_customer = random.choice(best_customers)
                self._assign_customer_to_facility(chosen_customer, cheapest_facility)

            sorted_facilities = self.problem.facilities.sort_by_cost_capacity_ratio()

        return self.solution



