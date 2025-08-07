import sys
import solution
import problem

p = problem()

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

data = parse_dzn_file("Instances/wlp01.dzn")
print(data)

# I) biramo fabriku koja ima najnizu cenu otvaranja po jedinici
# II) u petlji se biraju korisnici koji imaju najpovoljniji trosak transporta

sorted_facilities = p.facilities.sort_by_cost_capacity_ratio()

while sorted_facilities:
    if p.customers.total_remaining_demand() == 0:
        break

    cheapest_facility = sorted_facilities.pop(0)
    cheapest_facility.open()

    while cheapest_facility.remaining_capacity:
        min_shipping_cost = sys.maxsize
        customer_with_min_shipping = None

        #nalazimo korisnika sa najmanjim troskom transporta do facilitya
        for cust in p.customers.customers_with_unmet_demand():
            if p.shipping_costs[cust.id, cheapest_facility.id] < min_shipping_cost:
                min_shipping_cost = p.shipping_costs[cust.id, cheapest_facility.id]
                customer_with_min_shipping = cust

        if customer_with_min_shipping is None:
            break;

        assign_amount = min(cheapest_facility.remaining_capacity, customer_with_min_shipping.remaining_demand)

        #oduzimamo iz kapaciteta ono sto je korisnik kupio
        cheapest_facility.remaining_capacity -= assign_amount
        #oduzimamo koliko je korisniku potrebno resursa
        customer_with_min_shipping.remaining_demand -= assign_amount
