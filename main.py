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
