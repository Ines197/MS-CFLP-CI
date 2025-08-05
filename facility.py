class Facility:
    def __init__(self, facility_id, capacity, opening_cost):
        self.id = facility_id
        self.capacity = capacity
        self.opening_cost = opening_cost
        self.is_open = False
        self.remaining_capacity = capacity

    def open(self):
        self.is_open = True

    def reset(self):
        self.is_open = False
        self.remaining_capacity = self.capacity