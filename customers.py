class Customers:
    def __init__(self, customer_list):
        self.customers = {c.id: c for c in customer_list}

    def __getitem__(self, customer_id):
        return self.customers[customer_id]

    def get_by_id(self, customer_id):
        return self.customers.get(customer_id)

    def all(self):
        return list(self.customers.values())

    def customers_with_unmet_demand(self):
        return [c for c in self.customers.values() if c.remaining_demand > 0]

    def total_demand(self):
        return sum(c.demand for c in self.customers.values())

    def total_remaining_demand(self):
        return sum(c.remaining_demand for c in self.customers.values())

    def sort_by_demand_descending(self):
        return sorted(self.customers.values(), key=lambda c: c.demand, reverse=True)

    def reset(self):
        for c in self.customers.values():
            c.reset()

    def __iter__(self):
        return iter(self.customers.values())

    def __len__(self):
        return len(self.customers)

    def __contains__(self, customer_id):
        return customer_id in self.customers