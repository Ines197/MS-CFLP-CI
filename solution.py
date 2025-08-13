import matplotlib as plt
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

    def visualize(self):
        """
        Simple 2D visualization:
        - Facilities are squares
        - Customers are circles
        - Lines show assignments
        """
        plt.figure(figsize=(8, 6))

        # fabrike
        for fac in self.instance.facilities:
            x, y = fac.x, fac.y
            color = 'green' if fac.id in self.facilities_open else 'red'
            plt.scatter(x, y, s=200, c=color, marker='s', label=f'Facility {fac.id}' if fac.id == 0 else "")
            plt.text(x, y + 0.5, f'F{fac.id}', ha='center')

        # kupci
        for cust in self.instance.customers:
            x, y = cust.x, cust.y
            plt.scatter(x, y, s=100, c='blue', marker='o', label=f'Customer {cust.id}' if cust.id == 0 else "")
            plt.text(x, y + 0.3, f'C{cust.id}', ha='center')

        # assignments
        for (cust_id, fac_id), amount in self.assignments.items():
            cust = self.instance.customers.get_by_id(cust_id)
            fac = self.instance.facilities[fac_id]
            plt.plot([cust.x, fac.x], [cust.y, fac.y], 'k--', alpha=0.5)
            mid_x = (cust.x + fac.x) / 2
            mid_y = (cust.y + fac.y) / 2
            plt.text(mid_x, mid_y, f'{amount}', color='purple', fontsize=8)

        plt.title(f"Solution Visualization - Total Cost: {self.total_cost():.2f}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()

    def print_solution(self):
        print("Assignments:")
        for (cust, fac), amt in self.assignments.items():
            print(f"  Customer {cust} → Facility {fac}: {amt}")
        print(f"Total cost: {self.total_cost():.2f}")