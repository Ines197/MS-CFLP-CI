class Customer:
    def __init__(self, customer_id, demand):
        self.id = customer_id
        self.demand = demand
        self.remaining_demand = demand

    def reset(self):
        self.remaining_demand = self.demand