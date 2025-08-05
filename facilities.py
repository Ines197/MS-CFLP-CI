class Facilities:
    def __init__(self, facility_list):
        # facility_list je lista objekata klase Facility
        self.facilities = {f.id: f for f in facility_list}

    def __getitem__(self, facility_id):
        # Omogućava pristup kao u dict-u: facilities[3]
        return self.facilities[facility_id]

    def get_by_id(self, facility_id):
        # Alternativa ako želiš sigurniji pristup (možeš koristiti get())
        return self.facilities.get(facility_id)

    def all(self):
        # Vraća listu svih Facility objekata
        return list(self.facilities.values())

    def open_facilities(self):
        return [f for f in self.facilities.values() if f.is_open]

    def closed_facilities(self):
        return [f for f in self.facilities.values() if not f.is_open]

    def facility_with_min_opening_cost(self):
        return min(self.facilities.values(), key=lambda f: f.opening_cost)

    def facility_with_max_capacity(self):
        return max(self.facilities.values(), key=lambda f: f.capacity)

    def sort_by_cost_capacity_ratio(self):
        return sorted(self.facilities.values(), key=lambda f: f.opening_cost / f.capacity)

    def reset(self):
        for f in self.facilities.values():
            f.reset()

    def __iter__(self):
        # Omogućava: for f in facilities
        return iter(self.facilities.values())

    def __len__(self):
        return len(self.facilities)

    def __contains__(self, facility_id):
        return facility_id in self.facilities