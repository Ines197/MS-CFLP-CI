import math
import sys
import solution
import problem
import facilities
import solver
import customer
import facility
import heuristics
import solution
import customers

def parse_dzn_file(filename):
    data = {}
    with open(filename, 'r') as f:
        content = f.read()
    content = content.replace('\n', '').replace(' ', '')
    statements = content.split(';')
    for stmt in statements:
        if '=' in stmt:
            key, value = stmt.split('=', 1)
            if value.startswith('[') and value.endswith(']'):
                data[key] = list(map(int, value.strip('[]').replace('|', '').split(',')))
            elif value.startswith('array2d'):
                _, values = value.split(',', 1)
                values = values.strip(')').strip('[').strip(']').replace('|', '')
                data[key] = list(map(int, values.split(',')))
            else:
                data[key] = int(value)
    return data

data = parse_dzn_file("instances/wlp02.dzn")

nf = 4
nc = 10

Capacity = [100, 40, 60, 60]
FixedCost = [860, 350, 440, 580]

print(data)
facilities1 = data["Warehouses"]
customers1 = data["Stores"]
capacity = data["Capacity"]
fixed_cost = data["FixedCost"]
goods = data["Goods"]
shipping_costs = data["SupplyCost"]
incompatibilities = data["Incompatibilities"]
incompatible_pairs = data["IncompatiblePairs"]

facs = []
rusty = []

for i in range(0, facilities1):
    facility_new = facility.Facility(i, capacity[i], fixed_cost[i])
    facs.append(facility_new)

facilities_1= facilities.Facilities(facs)


for i in range(0, customers1):
    customer_new = customer.Customer(i, goods[i])
    rusty.append(customer_new)

custo = customers.Customers(rusty)

print(customers1)
print(len(shipping_costs))
shipping_cost = {}
print(math.floor(len(shipping_costs)//customers1))
for i in range(0, customers1):
    for j in range(0, math.floor(len(shipping_costs)/customers1)):
        shipping_cost[str((i))+str((j))] = shipping_costs[i*customers1 + j]
print(shipping_cost)
p = problem.Problem(facilities_1, custo, shipping_costs, incompatibilities)
s = solver.Solver(p)

#s.solve_greedy()

print(s.solution.facility_used_capacity)
print(s.solution.customer_supply)
print(data)
