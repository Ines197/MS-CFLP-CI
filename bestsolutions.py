import heapq
import random
from solution import Solution
import itertools

_counter = itertools.count()  # globalni brojač za tie-breaker

class BestSolutions:
    def __init__(self, n: int):
        self.n = n
        # heap čuvamo kao (-cost, counter, solution)
        self._heap = []

    def add(self, solution: Solution):
        cost = solution.total_cost()
        entry = (-cost, next(_counter), solution.copy())
        if len(self._heap) < self.n:
            heapq.heappush(self._heap, entry)
        else:
            worst_cost, _, _ = self._heap[0]
            worst_cost = -worst_cost
            if cost < worst_cost:
                heapq.heapreplace(self._heap, entry)

    def get_solutions(self):
        # vraća sortirano po rastućem cost-u
        return [sol for _, _, sol in sorted(self._heap, key=lambda x: -x[0])]

    def random_solution(self):
        if not self._heap:
            return None
        _, _, sol = random.choice(self._heap)
        return sol

    def best(self):
        if not self._heap:
            return None
        return min(self._heap, key=lambda x: -x[0])[2]

    def worst(self):
        if not self._heap:
            return None
        return max(self._heap, key=lambda x: -x[0])[2]

    def __len__(self):
        return len(self._heap)
