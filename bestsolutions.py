import heapq
import random
from solution import Solution

class BestSolutions:
    def __init__(self, n: int):
        self.n = n
        # heap čuvamo kao (-cost, solution), jer heapq u Python-u pravi min-heap
        # a mi hoćemo da najveći cost bude na vrhu (da ga najlakše izbacimo)
        self._heap = []

    def add(self, solution: Solution):
        cost = solution.total_cost()
        # ako još nema n rešenja, ubaci odmah
        if len(self._heap) < self.n:
            heapq.heappush(self._heap, (-cost, solution.copy()))
        else:
            # pogledaj najgore (najveći cost)
            worst_cost, _ = self._heap[0]
            worst_cost = -worst_cost
            if cost < worst_cost:
                # izbaci najgore i ubaci novo
                heapq.heapreplace(self._heap, (-cost, solution.copy()))

    def get_solutions(self):
        # vraća sortirano po rastućem cost-u
        return [sol for _, sol in sorted(self._heap, key=lambda x: -x[0])]

    def random_solution(self):
        if not self._heap:
            return None
        _, sol = random.choice(self._heap)
        return sol

    def best(self):
        if not self._heap:
            return None
        return min(self._heap, key=lambda x: -x[0])[1]

    def worst(self):
        if not self._heap:
            return None
        return max(self._heap, key=lambda x: -x[0])[1]

    def __len__(self):
        return len(self._heap)
