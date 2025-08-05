class Solution:
    def __init__(self, instance):
        self.instance = instance
        self.assignments = {}
        self.facility_used_capacity = {}
        self.customer_supply = {}
        self.facilities_open = set()

    def add_assignment(self, cust_id, fac_id, amount):
        key = (cust_id, fac_id)
        self.assignments[key] = self.assignments.get(key, 0) + amount
        self.facility_used_capacity[fac_id] = self.facility_used_capacity.get(fac_id, 0) + amount
        self.customer_supply[cust_id] = self.customer_supply.get(cust_id, 0) + amount
        self.facilities_open.add(fac_id)

    def get_assigned_customers_for_facility(self, fac_id):
        return [
            cust_id for (cust_id, f_id), amount in self.assignments.items()
            if f_id == fac_id and amount > 0
        ]

    def get_total_assigned_to_customer(self, cust_id):
        return self.customer_supply.get(cust_id, 0)

    def total_cost(self):
        shipping = sum(
            self.instance.shipping_costs[(cust_id, fac_id)] * amount
            for (cust_id, fac_id), amount in self.assignments.items()
        )
        opening = sum(
            self.instance.facilities[fac_id].opening_cost
            for fac_id in self.facilities_open
        )
        return shipping + opening

    def is_valid(self):
        # 1. Svi kupci zadovoljeni
        for cust in self.instance.customers:
            if self.get_total_assigned_to_customer(cust.id) < cust.demand:
                return False

        # 2. Kapaciteti fabrika
        for fac in self.instance.facilities:
            used = self.facility_used_capacity.get(fac.id, 0)
            if used > fac.capacity:
                return False

        # 3. Inkompatibilnosti
        for fac_id in self.facilities_open:
            assigned = self.get_assigned_customers_for_facility(fac_id)
            for i in range(len(assigned)):
                for j in range(i + 1, len(assigned)):
                    pair = (assigned[i], assigned[j])
                    rev_pair = (assigned[j], assigned[i])
                    if pair in self.instance.incompatibilities or rev_pair in self.instance.incompatibilities:
                        return False

        return True

    def print_solution(self):
        print("Assignments:")
        for (cust, fac), amt in self.assignments.items():
            print(f"  Customer {cust} → Facility {fac}: {amt}")
        print(f"Total cost: {self.total_cost():.2f}")