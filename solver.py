import solution
import random
import heuristics

class Solver:
    def __init__(self, problem_instance):
        self.problem = problem_instance
        self.solution = solution.Solution(problem_instance)
        self.heuristics = heuristics.Heuristics(self)

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
        # Sort facilities once
        sorted_facilities = self.problem.facilities.sort_by_cost_capacity_ratio()

        for fac in sorted_facilities:
            if self.problem.customers.total_remaining_demand() == 0:
                break

            fac.open()

            while fac.remaining_capacity > 0:
                best_customers = self.heuristics.rcl(fac, rcl_size=5)

                if not best_customers:
                    break

                chosen_cust = random.choice(best_customers)

                # Assign using existing method
                self._assign_customer_to_facility(chosen_cust, fac)

                # Update RCL
                self.heuristics.customer_rcl.update_after_assignment(fac, chosen_cust)
                if chosen_cust.remaining_demand == 0:
                    self.heuristics.customer_rcl.remove_customer(chosen_cust.id)
