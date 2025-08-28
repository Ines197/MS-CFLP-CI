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


def parse_instance(filename):
    with open(filename, "r") as f:
        text = f.read()

    # Extract counts
    n_fac = int(re.search(r"Warehouses\s*=\s*(\d+);", text).group(1))
    n_cust = int(re.search(r"Stores\s*=\s*(\d+);", text).group(1))

    # Parse arrays
    capacity = list(map(int, re.search(r"Capacity\s*=\s*\[(.*?)\];", text).group(1).split(",")))
    fixed_cost = list(map(int, re.search(r"FixedCost\s*=\s*\[(.*?)\];", text).group(1).split(",")))
    demand = list(map(int, re.search(r"Goods\s*=\s*\[(.*?)\];", text).group(1).split(",")))

    # Parse SupplyCost block (matrix)
    supply_text = re.search(r"SupplyCost\s*=\s*\[\|(.*?)\|\];", text, re.DOTALL).group(1)
    rows = [row.strip(" |") for row in supply_text.strip().splitlines()]
    supply_matrix = [list(map(int, row.split(","))) for row in rows]

    # Parse incompatibilities
    incomp_text = re.search(r"IncompatiblePairs\s*=\s*\[\|(.*?)\|\];", text, re.DOTALL).group(1)
    pairs = re.findall(r"(\d+),\s*(\d+)", incomp_text)
    # convert to 0-based indices
    incompatibilities = [(int(a) - 1, int(b) - 1) for a, b in pairs]

    # --- Build objects ---
    facilities = [Facility(i, capacity[i], fixed_cost[i]) for i in range(n_fac)]
    customers = [Customer(i, demand[i]) for i in range(n_cust)]

    # Wrap into containers
    facilities = Facilities(facilities)
    customers = Customers(customers)

    # Build shipping cost dictionary
    shipping_costs = {
        (cust_id, fac_id): supply_matrix[cust_id][fac_id]
        for cust_id in range(n_cust)
        for fac_id in range(n_fac)
    }

    return Instance(facilities, customers, shipping_costs, incompatibilities)
