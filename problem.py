import facilities
import customers

class Problem:
    def __init__(self, facilities, customers, shipping_costs, incompatibilities):
        self.facilities = facilities.Facilities(facilities)
        self.customers = customers.Customers(customers)
        self.shipping_costs = shipping_costs  # dict: (cust_id, fac_id) → cost
        self.incompatibilities = incompatibilities  # set of (cust1, cust2)

    def reset(self):
        self.facilities.reset()
        self.customers.reset()