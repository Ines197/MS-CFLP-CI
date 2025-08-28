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

    def get_top(self, facility, rcl_size=None):
        if facility.id not in self.eligible:
            return []
        if rcl_size is None:
            return self.eligible[facility.id]
        return self.eligible[facility.id][:rcl_size]

    def remove_customer(self, customer_id):
        for fac_id in self.eligible:
            self.eligible[fac_id] = [c for c in self.eligible[fac_id] if c.id != customer_id]

    def update_after_assignment(self, facility, customer):
        self.eligible[facility.id] = [
            c for c in self.eligible[facility.id] if c.id != customer.id
        ]
        already_assigned = self.solver.solution.get_assigned_customers_for_facility(facility.id)
        self.eligible[facility.id] = [
            c for c in self.eligible[facility.id]
            if not self.solver.has_conflict(c.id, already_assigned)
        ]
