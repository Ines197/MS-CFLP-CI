import re

from customer import Customer
from facility import Facility


class Instance:
    def __init__(self, facilities, customers, shipping_costs, incompatibilities):
        self.facilities = facilities
        self.customers = customers
        self.shipping_costs = shipping_costs  # dict {(cust_id, fac_id): cost}
        self.incompatibilities = incompatibilities

    def get_facility_by_id(self, fac_id):
        return next(f for f in self.facilities if f.id == fac_id)

    def get_customer_by_id(self, cust_id):
        return next(c for c in self.customers if c.id == cust_id)


def parse_instance(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    # Helper to extract arrays like [1,2,3]
    def parse_array(pattern):
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            return []
        arr_str = match.group(1)
        # remove line breaks, pipes and extra spaces
        arr_str = arr_str.replace("|", "").replace("\n", " ")
        return [int(x) for x in re.findall(r"-?\d+", arr_str)]

    # Scalars
    n_fac = int(re.search(r"Warehouses\s*=\s*(\d+);", content).group(1))
    n_cust = int(re.search(r"Stores\s*=\s*(\d+);", content).group(1))

    # Arrays
    capacities = parse_array(r"Capacity\s*=\s*\[(.*?)\];")
    fixed_costs = parse_array(r"FixedCost\s*=\s*\[(.*?)\];")
    demands = parse_array(r"Goods\s*=\s*\[(.*?)\];")

    # Supply cost matrix
    match = re.search(r"SupplyCost\s*=\s*\[\|(.*?)\|\];", content, re.DOTALL)
    supply_costs = []
    if match:
        rows = match.group(1).strip().split("\n")
        for row in rows:
            row = row.strip().lstrip("|").rstrip("|").strip()
            if row:
                supply_costs.append([int(x) for x in re.findall(r"-?\d+", row)])

    # Incompatibilities
    incompatibilities = []
    match = re.search(r"IncompatiblePairs\s*=\s*\[\|(.*?)\|\];", content, re.DOTALL)
    if match:
        pairs_str = match.group(1)
        pairs = re.findall(r"(\d+)\s*,\s*(\d+)", pairs_str)
        incompatibilities = [(int(a)-1, int(b)-1) for a, b in pairs]  # zero-based

    # Build objects
    facilities = [
        Facility(i, capacities[i], fixed_costs[i])
        for i in range(n_fac)
    ]
    customers = [
        Customer(i, demands[i])
        for i in range(n_cust)
    ]

    # shipping_costs: map (cust, fac) → cost
    shipping_costs = {}
    for cust_id in range(n_cust):
        for fac_id in range(n_fac):
            shipping_costs[(cust_id, fac_id)] = supply_costs[cust_id][fac_id]

    return Instance(facilities, customers, shipping_costs, incompatibilities)
