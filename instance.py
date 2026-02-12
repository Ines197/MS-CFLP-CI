import re
from facility import Facility
from customer import Customer
from facilities import Facilities
from customers import Customers

class Instance:
    def __init__(self, facilities, customers, shipping_costs, incompatibilities):
        self.facilities = facilities          # Facilities wrapper
        self.customers = customers            # Customers wrapper
        self.shipping_costs = shipping_costs  # dict[(cust_id, fac_id)] = cost
        self.incompatibilities = incompatibilities

    def reset(self):
        self.facilities.reset()
        self.customers.reset()


import re


def parse_instance(filename):
    with open(filename, "r") as f:
        text = f.read()

    def safe_search(pattern, content, name):
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            # Instead of crashing, we raise a helpful error or return empty
            print(f"[WARNING] Could not find {name} in {filename}")
            return None
        return match.group(1)

    # 1. Extract counts
    n_fac_match = re.search(r"Warehouses\s*=\s*(\d+);", text)
    n_cust_match = re.search(r"Stores\s*=\s*(\d+);", text)

    if not n_fac_match or not n_cust_match:
        raise ValueError(f"Missing Warehouse/Store counts in {filename}")

    n_fac = int(n_fac_match.group(1))
    n_cust = int(n_cust_match.group(1))

    # 2. Parse arrays (Added \s* for safety)
    capacity = list(map(int, safe_search(r"Capacity\s*=\s*\[(.*?)\]\s*;", text, "Capacity").split(",")))
    fixed_cost = list(map(int, safe_search(r"FixedCost\s*=\s*\[(.*?)\]\s*;", text, "FixedCost").split(",")))
    demand = list(map(int, safe_search(r"Goods\s*=\s*\[(.*?)\]\s*;", text, "Goods").split(",")))

    # 3. Parse SupplyCost block (matrix)
    # Updated pattern to handle whitespace like | ];
    supply_text = safe_search(r"SupplyCost\s*=\s*\[\|(.*?)\|\s*\]\s*;", text, "SupplyCost")
    rows = [row.strip(" |") for row in supply_text.strip().splitlines()]
    supply_matrix = [list(map(int, row.split(","))) for row in rows]

    # 4. Parse incompatibilities (The 2.1M pairs block)
    # This regex is now flexible with whitespaces
    incomp_text = safe_search(r"IncompatiblePairs\s*=\s*\[\|(.*?)\|\s*\]\s*;", text, "IncompatiblePairs")

    incompatibilities = set()
    if incomp_text:
        # Using finditer is more memory-efficient for 2 million entries
        pairs = re.finditer(r"(\d+),\s*(\d+)", incomp_text)
        for match in pairs:
            a, b = int(match.group(1)) - 1, int(match.group(2)) - 1
            incompatibilities.add((a, b))
            incompatibilities.add((b, a))

    # 5. Build objects
    customers_list = [Customer(i, demand[i]) for i in range(n_cust)]
    customers_wrapper = Customers(customers_list)

    facilities_list = [
        Facility(i, capacity[i], fixed_cost[i], customers_wrapper)
        for i in range(n_fac)
    ]
    facilities_wrapper = Facilities(facilities_list)

    # 6. Build shipping cost dictionary
    shipping_costs = {
        (cust_id, fac_id): supply_matrix[cust_id][fac_id]
        for cust_id in range(n_cust)
        for fac_id in range(n_fac)
    }

    instance_obj = Instance(facilities_wrapper, customers_wrapper, shipping_costs, incompatibilities)

    for f in facilities_wrapper.all():
        f.set_instance(instance_obj)

    return instance_obj