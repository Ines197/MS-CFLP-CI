from typing import Optional, Dict, List, Tuple

import solution
import random
import heuristics

class Solver:
    def __init__(self, problem_instance):
        self.problem = problem_instance
        self.solution = solution.Solution(problem_instance)
        self.heuristics = heuristics.Heuristics(self)
        self.rng = random.Random(53)

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
                best_customers = self.heuristics.rcl(fac, 5)

                if not best_customers:
                    break

                chosen_cust = random.choice(best_customers)

                # Assign using existing method
                self._assign_customer_to_facility(chosen_cust, fac)

                # Update RCL
                self.heuristics.customer_rcl.update_after_assignment(fac, chosen_cust)
                if chosen_cust.remaining_demand == 0:
                    self.heuristics.customer_rcl.remove_customer(chosen_cust.id)

        if not self.solution.is_valid():
            print("Greedy solution is invalid!")
        else:
            print("Greedy solution is valid!")

    def solve_local_search(self, max_passes: int = 10, step: float = float("inf")) -> None:


        inst = self.problem

        fac_by_id = {fac.id: fac for fac in inst.facilities}

        def unit_cost(cust_id: int, fac_id: int) -> float:
            return inst.shipping_costs[(cust_id, fac_id)]

        def remaining_capacity(fac_id: int) -> float:
            used = self.solution.facility_used_capacity.get(fac_id, 0)
            return fac_by_id[fac_id].capacity - used

        def _remove_assignment(cust_id: int, fac_id: int, amount: float) -> float:
            key = (cust_id, fac_id)
            old = self.solution.assignments.get(key, 0.0)
            take = min(old, amount)
            if take <= 0:
                return 0.0


            new_amt = old - take
            if new_amt > 0:
                self.solution.assignments[key] = new_amt
            else:
                self.solution.assignments.pop(key, None)


            self.solution.facility_used_capacity[fac_id] = self.solution.facility_used_capacity.get(fac_id, 0.0) - take
            if self.solution.facility_used_capacity[fac_id] <= 0:
                self.solution.facility_used_capacity.pop(fac_id, None)

            self.solution.customer_supply[cust_id] = self.solution.customer_supply.get(cust_id, 0.0) - take
            if self.solution.customer_supply[cust_id] <= 0:
                self.solution.customer_supply.pop(cust_id, None)


            still_used = any(f_id == fac_id and amt > 0 for (c_id, f_id), amt in self.solution.assignments.items())
            if not still_used and fac_id in self.solution.facilities_open:
                self.solution.facilities_open.remove(fac_id)

            return take

        EPS = 1e-12
        passes = 0
        improved_globally = True

        while improved_globally and passes < max_passes:
            improved_globally = False
            passes += 1

            for A in inst.facilities:
                A_id = A.id
                pairs = [(cust_id, amt) for (cust_id, f_id), amt in self.solution.assignments.items()
                         if f_id == A_id and amt > 0]
                if not pairs:
                    continue

                for cust_id, amt in pairs:
                    cA = unit_cost(cust_id, A_id)

                    best_delta = 0.0
                    best_move = None

                    for B in inst.facilities:
                        B_id = B.id
                        if B_id == A_id:
                            continue
                        capB = remaining_capacity(B_id)
                        if capB <= 0:
                            continue
                        cB = unit_cost(cust_id, B_id)
                        if cB >= cA:
                            continue

                        move_amt = min(amt, capB, step)
                        if move_amt <= 0:
                            continue

                        delta_shipping = (cB - cA) * move_amt

                        open_penalty = 0.0
                        if B_id not in self.solution.facilities_open:
                            open_penalty = fac_by_id[B_id].opening_cost

                        used_A = self.solution.facility_used_capacity.get(A_id, 0.0)
                        close_saving = fac_by_id[A_id].opening_cost if abs(used_A - move_amt) <= EPS else 0.0

                        total_delta = delta_shipping + open_penalty - close_saving

                        if total_delta < best_delta:
                            best_delta = total_delta
                            best_move = (B_id, move_amt, open_penalty, close_saving)

                    if best_move is not None:
                        B_id, move_amt, open_penalty, close_saving = best_move

                        taken = _remove_assignment(cust_id, A_id, move_amt)
                        if taken <= 0:
                            continue

                        self.solution.add_assignment(cust_id, B_id, taken)

                        improved_globally = True
        if not self.solution.is_valid():
            print("Local search solution is invalid!")
        else:
            print("Local search solution is valid!")