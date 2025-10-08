from customer_rcl import CustomerRCL
import random

class Heuristics:
    def __init__(self, solver):
        self.solver = solver
        self.problem = solver.problem
        self.solution = solver.solution
        self.customer_rcl = CustomerRCL(self.problem, self.solver)
        self.rng = solver.rng

    def rcl(self, facility, rcl_size):
        return self.customer_rcl.get_top(facility, rcl_size)

    def close_one_facility(self):
        # Pick open facilities supplying only one customer
        candidates = [
            fac for fac in self.problem.facilities.all() if fac.id in self.solution.facilities_open
                                                            and len(
                self.solution.get_assigned_customers_for_facility(fac.id)) == 1
        ]
        if not candidates:
            return False  # nothing to do

        fac_to_close = random.choice(candidates)
        cust_id = self.solution.get_assigned_customers_for_facility(fac_to_close.id)[0]

        # Remove assignments from this facility
        amount = self.solution.assignments.pop((cust_id, fac_to_close.id))
        self.solution.facility_used_capacity[fac_to_close.id] -= amount
        self.solution.facilities_open.discard(fac_to_close.id)

        # Reassign this customer's demand to other open facilities
        remaining_demand = self.problem.customers[cust_id].demand - self.solution.get_total_assigned_to_customer(
            cust_id)
        open_facilities = [f for f in self.problem.facilities.all() if f.id in self.solution.facilities_open]
        open_facilities.sort(key=lambda f: self.problem.shipping_costs[(cust_id, f.id)])

        for f in open_facilities:
            cap = f.capacity - self.solution.facility_used_capacity.get(f.id, 0)
            if cap <= 0 or remaining_demand <= 0:
                continue
            assign_amt = min(cap, remaining_demand)
            self.solution.add_assignment(cust_id, f.id, assign_amt)
            remaining_demand -= assign_amt

    def open_one_facility(self):
        closed_facilities = [f for f in self.problem.facilities.all() if f.id not in self.solution.facilities_open]
        if not closed_facilities:
            return False
        fac_to_open = random.choice(closed_facilities)
        self.solution.facilities_open.add(fac_to_open.id)

    def _are_customers_incompatible(self, cust_set_a, cust_set_b):
        """
        Defensive check for incompatibilities between two sets of customers.
        Supports several representations on self.problem:
         - a method are_incompatible(i,j)
         - an attribute incompatible_pairs (set of frozensets or tuple pairs)
         - an attribute incompatibilities which can be:
             * dict: mapping a -> set(of incompatible customers)
             * set: set of frozensets/tuples representing incompatible pairs
        If none exist, assumes no incompatibilities (returns False).
        """
        prob = self.problem

        # method on problem: are_incompatible(i,j)
        if hasattr(prob, "are_incompatible") and callable(prob.are_incompatible):
            for a in cust_set_a:
                for b in cust_set_b:
                    if prob.are_incompatible(a, b):
                        return True
            return False

        # incompatible_pairs as set (of frozenset or tuple) or similar
        if hasattr(prob, "incompatible_pairs"):
            pairs = prob.incompatible_pairs
            # If it's a dict-like structure, try membership appropriately,
            # but most common case is a set of pairs.
            try:
                # treat pairs as a container supporting "in"
                for a in cust_set_a:
                    for b in cust_set_b:
                        if frozenset((a, b)) in pairs or (a, b) in pairs or (b, a) in pairs:
                            return True
                return False
            except Exception:
                # fallback to continue checking other representations
                pass

        # incompatibilities attribute can be dict-like or set-like
        if hasattr(prob, "incompatibilities"):
            inc = prob.incompatibilities
            # dict-like mapping a -> set(of incompatible customers)
            if hasattr(inc, "get") and callable(getattr(inc, "get")):
                for a in cust_set_a:
                    bs = inc.get(a, set())
                    for b in cust_set_b:
                        if b in bs:
                            return True
                return False

            # if it's a set of pairs (frozenset or tuple), check membership
            if isinstance(inc, set):
                for a in cust_set_a:
                    for b in cust_set_b:
                        if frozenset((a, b)) in inc or (a, b) in inc or (b, a) in inc:
                            return True
                return False

            # otherwise try to treat as mapping-like (index access)
            try:
                for a in cust_set_a:
                    bs = inc[a]  # may raise
                    for b in cust_set_b:
                        if b in bs:
                            return True
                return False
            except Exception:
                # unable to interpret, assume no incompatibilities
                return False

        # fallback: assume no incompatibilities
        return False

    def close_one_open_one(self):
        """
        3. Close one facility and open one facility.
        Randomly choose open j1 and closed j2 such that s_j2 >= sum_i x_{i j1}.
        Close j1 and open j2, then reassign all supplies from j1 to j2.
        """
        sol = self.solution
        prob = self.problem

        open_facs = [f for f in prob.facilities.all() if f.id in sol.facilities_open]
        closed_facs = [f for f in prob.facilities.all() if f.id not in sol.facilities_open]

        # build candidate pairs (j1 open, j2 closed) where capacity j2 >= total assigned to j1
        candidates = []
        for j1 in open_facs:
            assigned = sol.get_assigned_customers_for_facility(j1.id)
            total_assigned = sum(sol.assignments.get((c_id, j1.id), 0) for c_id in assigned)
            for j2 in closed_facs:
                if j2.capacity >= total_assigned:
                    candidates.append((j1, j2))

        if not candidates:
            return False

        j1, j2 = random.choice(candidates)

        # Move assignments from j1 to j2 (capacity guaranteed)
        assigned_customers = list(sol.get_assigned_customers_for_facility(j1.id))
        for c_id in assigned_customers:
            amt = sol.assignments.pop((c_id, j1.id), 0)
            # reduce old capacity usage
            sol.facility_used_capacity[j1.id] = sol.facility_used_capacity.get(j1.id, 0) - amt
            sol.customer_supply[c_id] = sol.customer_supply.get(c_id, 0) - amt

            # add to new facility
            sol.add_assignment(c_id, j2.id, amt)

        # close j1, open j2
        sol.facilities_open.discard(j1.id)
        sol.facilities_open.add(j2.id)

        # ensure facility_used_capacity tidy
        if sol.facility_used_capacity.get(j1.id, 0) <= 0:
            sol.facility_used_capacity.pop(j1.id, None)

        return True

    def close_one_open_two(self):
        """
        4. Close one facility and open two facilities.
        Select open j1 and closed j2,j3 such that s_j2 + s_j3 >= total assigned to j1
        and the opening cost improvement is maximum.
        Close j1 and open j2 & j3, then reassign all supplies from j1 to j2/j3
        by cheapest feasible assignments.
        """
        sol = self.solution
        prob = self.problem

        open_facs = [f for f in prob.facilities.all() if f.id in sol.facilities_open]
        closed_facs = [f for f in prob.facilities.all() if f.id not in sol.facilities_open]

        best = None  # (improvement, j1, j2, j3)
        for j1 in open_facs:
            assigned_customers = sol.get_assigned_customers_for_facility(j1.id)
            total_assigned = sum(sol.assignments.get((c_id, j1.id), 0) for c_id in assigned_customers)
            # consider all pairs of closed facilities
            n = len(closed_facs)
            for i in range(n):
                for j in range(i + 1, n):
                    j2 = closed_facs[i]
                    j3 = closed_facs[j]
                    combined_cap = j2.capacity + j3.capacity
                    if combined_cap >= total_assigned:
                        # improvement: how much opening cost decreases by replacing j1 with j2+j3
                        # interpretation: improvement = opening_cost(j1) - (opening_cost(j2)+opening_cost(j3))
                        improvement = j1.opening_cost - (j2.opening_cost + j3.opening_cost)
                        if best is None or improvement > best[0]:
                            best = (improvement, j1, j2, j3)

        if best is None:
            return False

        _, j1, j2, j3 = best

        # open j2 and j3, close j1
        sol.facilities_open.discard(j1.id)
        sol.facilities_open.add(j2.id)
        sol.facilities_open.add(j3.id)

        # initialize used capacities for new facilities
        sol.facility_used_capacity[j2.id] = sol.facility_used_capacity.get(j2.id, 0)
        sol.facility_used_capacity[j3.id] = sol.facility_used_capacity.get(j3.id, 0)

        # Reassign customers from j1 to j2/j3 by cheapest feasible assignments
        assigned_customers = list(sol.get_assigned_customers_for_facility(j1.id))
        # compute remaining capacities
        cap2 = j2.capacity - sol.facility_used_capacity.get(j2.id, 0)
        cap3 = j3.capacity - sol.facility_used_capacity.get(j3.id, 0)

        for c_id in assigned_customers:
            amt = sol.assignments.pop((c_id, j1.id), 0)
            if amt <= 0:
                continue
            sol.customer_supply[c_id] = sol.customer_supply.get(c_id, 0) - amt
            sol.facility_used_capacity[j1.id] = sol.facility_used_capacity.get(j1.id, 0) - amt

            remaining = amt
            # order j2/j3 by shipping cost for this customer
            costs = [
                (prob.shipping_costs[(c_id, j2.id)], j2),
                (prob.shipping_costs[(c_id, j3.id)], j3)
            ]
            costs.sort(key=lambda x: x[0])
            for _, chosen in costs:
                if remaining <= 0:
                    break
                if chosen.id == j2.id:
                    avail = cap2
                else:
                    avail = cap3
                if avail <= 0:
                    continue
                assign_amt = min(avail, remaining)
                sol.add_assignment(c_id, chosen.id, assign_amt)
                remaining -= assign_amt
                if chosen.id == j2.id:
                    cap2 -= assign_amt
                else:
                    cap3 -= assign_amt

            if remaining > 1e-9:
                # This should not happen because combined cap >= total_assigned,
                # but keep fallback: try other open facilities (very unlikely)
                other_open = [f for f in prob.facilities.all() if f.id in sol.facilities_open and f.id not in (j2.id, j3.id)]
                other_open.sort(key=lambda f: prob.shipping_costs[(c_id, f.id)])
                for f in other_open:
                    avail = f.capacity - sol.facility_used_capacity.get(f.id, 0)
                    if avail <= 0:
                        continue
                    amt2 = min(avail, remaining)
                    sol.add_assignment(c_id, f.id, amt2)
                    remaining -= amt2
                    if remaining <= 0:
                        break
            # done with this customer

        # tidy j1 used capacity if zero
        if sol.facility_used_capacity.get(j1.id, 0) <= 0:
            sol.facility_used_capacity.pop(j1.id, None)

        return True

    def open_one_close_two(self):
        """
        5. Open one facility and close two facilities.
        Select closed j1 and open j2,j3 such that:
          s_j1 >= sum_i (x_{i j2} + x_{i j3}),
          there are no incompatibilities between customers of j2 and j3,
          and opening cost improvement is maximum.
        Open j1, close j2 and j3, then reassign all supplies from j2 and j3 to j1.
        """
        sol = self.solution
        prob = self.problem

        open_facs = [f for f in prob.facilities.all() if f.id in sol.facilities_open]
        closed_facs = [f for f in prob.facilities.all() if f.id not in sol.facilities_open]

        best = None  # (improvement, j1, j2, j3)
        for j1 in closed_facs:
            for i in range(len(open_facs)):
                for j in range(i + 1, len(open_facs)):
                    j2 = open_facs[i]
                    j3 = open_facs[j]
                    # customers served by j2 and j3
                    custs_j2 = set(sol.get_assigned_customers_for_facility(j2.id))
                    custs_j3 = set(sol.get_assigned_customers_for_facility(j3.id))
                    combined_demand = 0
                    for c in custs_j2:
                        combined_demand += sol.assignments.get((c, j2.id), 0)
                    for c in custs_j3:
                        combined_demand += sol.assignments.get((c, j3.id), 0)

                    # capacity check
                    if j1.capacity < combined_demand:
                        continue

                    # incompatibility check
                    if self._are_customers_incompatible(custs_j2, custs_j3):
                        continue

                    # opening cost improvement: replace j2+j3 by j1:
                    # improvement = (opening_cost_j2 + opening_cost_j3) - opening_cost_j1
                    improvement = (j2.opening_cost + j3.opening_cost) - j1.opening_cost
                    if best is None or improvement > best[0]:
                        best = (improvement, j1, j2, j3)

        if best is None:
            return False

        _, j1, j2, j3 = best

        # open j1, close j2 and j3
        sol.facilities_open.add(j1.id)
        sol.facilities_open.discard(j2.id)
        sol.facilities_open.discard(j3.id)

        # prepare capacity accounting
        sol.facility_used_capacity[j1.id] = sol.facility_used_capacity.get(j1.id, 0)
        remaining_cap_j1 = j1.capacity - sol.facility_used_capacity[j1.id]

        # move all assignments from j2 and j3 to j1 (j1 capacity guaranteed)
        for src in (j2, j3):
            assigned_customers = list(sol.get_assigned_customers_for_facility(src.id))
            for c_id in assigned_customers:
                amt = sol.assignments.pop((c_id, src.id), 0)
                if amt <= 0:
                    continue
                sol.customer_supply[c_id] = sol.customer_supply.get(c_id, 0) - amt
                sol.facility_used_capacity[src.id] = sol.facility_used_capacity.get(src.id, 0) - amt

                assign_amt = min(remaining_cap_j1, amt)
                if assign_amt > 0:
                    sol.add_assignment(c_id, j1.id, assign_amt)
                    remaining_cap_j1 -= assign_amt
                    amt -= assign_amt

                if amt > 1e-9:
                    # fallback: should not happen because capacity checked earlier,
                    # but try to distribute remaining to other open facilities
                    other_open = [f for f in prob.facilities.all() if f.id in sol.facilities_open and f.id != j1.id]
                    other_open.sort(key=lambda f: prob.shipping_costs[(c_id, f.id)])
                    for f in other_open:
                        avail = f.capacity - sol.facility_used_capacity.get(f.id, 0)
                        if avail <= 0:
                            continue
                        take = min(avail, amt)
                        sol.add_assignment(c_id, f.id, take)
                        amt -= take
                        if amt <= 0:
                            break

        # tidy up used capacities for closed j2/j3
        for src in (j2, j3):
            if sol.facility_used_capacity.get(src.id, 0) <= 0:
                sol.facility_used_capacity.pop(src.id, None)

        return True


    def large_neighborhood_search(self, destruction_pct=(0.01, 0.05), ls_type=None):
        inst = self.problem
        sol = self.solution
        rng = self.rng

        # Nasumično biranje LS tipa ako nije definisano
        if ls_type is None:
            ls_type = rng.choice(["LS1", "LS2", "LS3"])

        # -------------------
        # Destruction phase
        # -------------------
        open_facs = list(sol.facilities_open)
        num_open = len(open_facs)
        num_to_remove = max(1, int(num_open * rng.uniform(*destruction_pct)))

        if ls_type == "LS1":
            # zatvori fabrike i ukloni sve dodeljene količine
            facs_to_close = rng.sample(open_facs, min(num_to_remove, len(open_facs)))
            for f_id in facs_to_close:
                assigned_customers = sol.get_assigned_customers_for_facility(f_id)
                for c_id in assigned_customers:
                    amt = sol.assignments.pop((c_id, f_id), 0)
                    sol.customer_supply[c_id] -= amt
                sol.facility_used_capacity.pop(f_id, None)
                sol.facilities_open.discard(f_id)

        elif ls_type == "LS2":
            # ostavi fabrike otvorene, ukloni do 50% potražnje kupaca
            facs_to_modify = rng.sample(open_facs, min(num_to_remove, len(open_facs)))
            for f_id in facs_to_modify:
                assigned_customers = sol.get_assigned_customers_for_facility(f_id)
                for c_id in assigned_customers:
                    amt = sol.assignments[(c_id, f_id)]
                    remove_amt = amt * rng.uniform(0, 0.5)
                    sol.assignments[(c_id, f_id)] -= remove_amt
                    sol.customer_supply[c_id] -= remove_amt
                    sol.facility_used_capacity[f_id] -= remove_amt

        elif ls_type == "LS3":
            # potpuno ukloni nekoliko kupaca
            all_customers = [c.id for c in inst.customers]
            custs_to_remove = rng.sample(
                all_customers, max(1, int(len(all_customers) * rng.uniform(*destruction_pct)))
            )
            for c_id in custs_to_remove:
                for f_id in list(sol.facilities_open):
                    amt = sol.assignments.pop((c_id, f_id), 0)
                    sol.customer_supply[c_id] -= amt
                    if f_id in sol.facility_used_capacity:
                        sol.facility_used_capacity[f_id] -= amt
                        if sol.facility_used_capacity[f_id] <= 0:
                            sol.facility_used_capacity.pop(f_id)
                            sol.facilities_open.discard(f_id)

        # -------------------
        # Repair phase
        # -------------------
        unassigned_customers = [
            c for c in inst.customers if sol.get_total_assigned_to_customer(c.id) < c.demand
        ]

        for cust in unassigned_customers:
            remaining_demand = cust.demand - sol.get_total_assigned_to_customer(cust.id)

            # Set 1: otvorene fabrike + zatvorene sa dovoljno kapaciteta
            set1 = []
            set2 = []

            for fac in inst.facilities:
                avail = fac.capacity - sol.facility_used_capacity.get(fac.id, 0)
                if fac.id in sol.facilities_open or avail >= remaining_demand:
                    set1.append(fac)
                else:
                    set2.append(fac)

            # sortiranje po unit cost
            def unit_cost(fac):
                shipping = inst.shipping_costs[(cust.id, fac.id)]
                opening = 0 if fac.id in sol.facilities_open else fac.opening_cost / max(1.0, remaining_demand)
                return shipping + opening

            set1.sort(key=unit_cost)
            set2.sort(key=unit_cost)

            # prvo Set 1, pa Set 2
            for fac in set1 + set2:
                if remaining_demand <= 0:
                    break
                avail = fac.capacity - sol.facility_used_capacity.get(fac.id, 0)
                if avail <= 0:
                    continue
                assign_amt = min(remaining_demand, avail)
                sol.add_assignment(cust.id, fac.id, assign_amt)
                remaining_demand -= assign_amt