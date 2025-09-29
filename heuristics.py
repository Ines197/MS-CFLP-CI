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