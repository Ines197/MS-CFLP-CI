import instance


instance = instance.parse_instance("test.dzn")

print("Facilities:")
for fac in instance.facilities:
    print(f"  Facility {fac.id} cap={fac.capacity}, cost={fac.opening_cost}")

print("Customers:")
for cust in instance.customers:
    print(f"  Customer {cust.id} demand={cust.demand}")

print("Shipping cost (Customer 0 → Facility 0):", instance.shipping_costs[(0,0)])
print("Incompatibilities:", instance.incompatibilities)

#s.solve_greedy()


