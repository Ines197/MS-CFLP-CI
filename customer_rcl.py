class CustomerRCL:
    def __init__(self, problem, solver):
        self.problem = problem
        self.solver = solver
        self.eligible = {}
        self._compute_eligible()

    def _compute_eligible(self):
        for fac in self.problem.facilities.all():
            already_assigned = self.solver.solution.get_assigned_customers_for_facility(fac.id)
            customers = [
                c for c in self.problem.customers.customers_with_unmet_demand()
                if not self.solver.has_conflict(c.id, already_assigned)
            ]
            customers.sort(key=lambda c: self.problem.shipping_costs[(c.id, fac.id)])
            self.eligible[fac.id] = customers

    def get_candidates(self, facility, tau=1.0):
        if facility.id not in self.eligible:
            return []

        candidates = []
        U_f = [f for f in self.problem.facilities.all() if f.id != facility.id]

        for cust in self.eligible[facility.id]:
            eff_cost = (facility.opening_cost / facility.capacity) + self.problem.shipping_costs[(cust.id, facility.id)]
            best_alt = min(
                (f.opening_cost / f.capacity) + self.problem.shipping_costs[(cust.id, f.id)]
                for f in U_f
            ) if U_f else eff_cost

            if eff_cost <= tau * best_alt:
                candidates.append(cust)

        return candidates

    def remove_customer(self, cust_id):
        for fac_id in self.eligible:
            self.eligible[fac_id] = [c for c in self.eligible[fac_id] if c.id != cust_id]

    def get_top(self, facility, rcl_size=None):
        if facility.id not in self.eligible:
            return []
        if rcl_size is None:
            rcl_size = 5
        return self.eligible[facility.id][:rcl_size]

    def update_after_assignment(self, facility, customer):
        self.eligible[facility.id] = [
            c for c in self.eligible[facility.id] if c.id != customer.id
        ]
        already_assigned = self.solver.solution.get_assigned_customers_for_facility(facility.id)
        self.eligible[facility.id] = [
            c for c in self.eligible[facility.id]
            if not self.solver.has_conflict(c.id, already_assigned)
        ]

    def reset(self):
        self._compute_eligible()