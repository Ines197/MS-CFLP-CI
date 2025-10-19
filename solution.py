import plotly.graph_objects as go
import random

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
                print("Nije ispunjen demand")
                return False

        # 2. Kapaciteti fabrika
        for fac in self.instance.facilities:
            used = self.facility_used_capacity.get(fac.id, 0)
            if used > fac.capacity:
                print("Uzeli smo vise nego sto smo smeli")
                return False

        # 3. Inkompatibilnosti
        for fac_id in self.facilities_open:
            assigned = self.get_assigned_customers_for_facility(fac_id)
            for i in range(len(assigned)):
                for j in range(i + 1, len(assigned)):
                    pair = (assigned[i], assigned[j])
                    rev_pair = (assigned[j], assigned[i])
                    if pair in self.instance.incompatibilities or rev_pair in self.instance.incompatibilities:
                        print("Nekompatibilnost")
                        return False

        return True

    def visualize(self):
        """
        Vizualizacija pomoću Plotly:
        - Fabrike = kvadrati (zelene ako su otvorene, crvene ako nisu)
        - Kupci = krugovi (plavi)
        - Grane = dodele (sive linije sa etiketom količine)
        """
        nodes_x, nodes_y, nodes_text, nodes_color, nodes_symbol = [], [], [], [], []
        edges_x, edges_y, edge_text = [], [], []

        # layout koordinata (Plotly nema automatski layout kao NetworkX spring_layout)
        # pa ćemo staviti fabrike gore, kupce dole
        fac_y, cust_y = 1, 0
        fac_step = 1 / (len(self.instance.facilities) + 1)
        cust_step = 1 / (len(self.instance.customers) + 1)

        fac_pos, cust_pos = {}, {}

        # fabrike
        for i, fac in enumerate(self.instance.facilities, start=1):
            x, y = i * fac_step, fac_y
            fac_pos[fac.id] = (x, y)
            nodes_x.append(x)
            nodes_y.append(y)
            nodes_text.append(f"F{fac.id}<br>Cap={fac.capacity}")
            nodes_color.append("green" if fac.id in self.facilities_open else "red")
            nodes_symbol.append("square")

        # kupci
        for i, cust in enumerate(self.instance.customers, start=1):
            x, y = i * cust_step, cust_y
            cust_pos[cust.id] = (x, y)
            nodes_x.append(x)
            nodes_y.append(y)
            nodes_text.append(f"C{cust.id}<br>Demand={cust.demand}")
            nodes_color.append("blue")
            nodes_symbol.append("circle")

        # grane
        for (cust_id, fac_id), amount in self.assignments.items():
            if amount > 0:
                x0, y0 = cust_pos[cust_id]
                x1, y1 = fac_pos[fac_id]
                edges_x += [x0, x1, None]
                edges_y += [y0, y1, None]
                mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
                edge_text.append((mid_x, mid_y, str(amount)))

        # trace za grane
        edge_trace = go.Scatter(
            x=edges_x, y=edges_y,
            line=dict(width=1, color="gray"),
            hoverinfo="none",
            mode="lines"
        )

        # trace za čvorove
        node_trace = go.Scatter(
            x=nodes_x, y=nodes_y,
            mode="markers+text",
            text=nodes_text,
            textposition="top center",
            marker=dict(
                color=nodes_color,
                size=20,
                symbol=nodes_symbol,
                line=dict(width=2, color="black")
            ),
            hoverinfo="text"
        )

        # figure
        fig = go.Figure(data=[edge_trace, node_trace])

        # dodaj oznake količina na grane
        for x, y, label in edge_text:
            fig.add_annotation(
                x=x, y=y, text=label,
                showarrow=False,
                font=dict(size=10, color="purple")
            )

        fig.update_layout(
            title=f"Solution Visualization - Total Cost: {self.total_cost():.2f}",
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False)
        )

        fig.show()

    def print_solution(self):
        print("Assignments:")
        for (cust, fac), amt in self.assignments.items():
            print(f"  Customer {cust} → Facility {fac}: {amt}")

        print(f"Total cost: {self.total_cost():.2f}")

    def customers_of_facility(self, id):
        opa = []
        for (f, c) in self.assignments.keys():
            opa.append(c)
        return opa

    def copy(self):
        new_sol = Solution(self.instance)
        new_sol.assignments = self.assignments.copy()
        new_sol.facility_used_capacity = self.facility_used_capacity.copy()
        new_sol.customer_supply = self.customer_supply.copy()
        new_sol.facilities_open = self.facilities_open.copy()
        return new_sol

    def reset(self):
        self.assignments.clear()
        self.facility_used_capacity.clear()
        self.customer_supply.clear()
        self.facilities_open.clear()

