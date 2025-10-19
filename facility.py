class Facility:
    def __init__(self, facility_id, capacity, opening_cost, customers=None):
        self.id = facility_id
        self.capacity = capacity
        self.remaining_capacity = capacity
        self.opening_cost = opening_cost
        self.is_open = False
        self.customers = customers  # može biti None, kasnije setovati
        self.instance = None        # kasnije možeš dodati reference na instance

    def set_instance(self, instance):
        self.instance = instance

    def set_customers(self, customers):
        self.customers = customers

    def open(self):
        self.is_open = True

    def reset(self):
        self.remaining_capacity = self.capacity
        self.is_open = False

    def second_part(self):
        size_of_unassigned_cust_set = len(self.customers.customers_with_unmet_demand())
        s = 0
        for c in self.customers.customers_with_unmet_demand():
            s = s + self.instance.shipping_costs[(c.id, self.id)]
        return 1/size_of_unassigned_cust_set*s