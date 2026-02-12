import numpy as np
import random
import time
import heuristics
from numba import njit
from typing import Dict, Any


@njit(cache=True)
def check_conflicts_numba(cust_idx: int, assigned: np.ndarray, n_assigned: int, incompat: np.ndarray) -> bool:
    for i in range(n_assigned):
        if incompat[cust_idx, assigned[i]]:
            return True
    return False


@njit(cache=True)
def solve_facility_numba(
        fac_idx: int,
        capacity: float,
        sorted_indices: np.ndarray,
        rem_demand: np.ndarray,
        incompat: np.ndarray,
        assigned_tracker: np.ndarray,
        seed: int
):
    np.random.seed(seed)

    rem_cap = capacity
    n_assigned = 0
    assigned_ids = assigned_tracker

    out_cust_indices = np.zeros(len(sorted_indices), dtype=np.int32)
    out_amounts = np.zeros(len(sorted_indices), dtype=np.float64)

    TOL = 1e-9
    k = 5
    max_len = len(sorted_indices)
    local_indices = sorted_indices.copy()

    i = 0
    while rem_cap > TOL and i < max_len:
        window_end = min(i + k, max_len)
        if window_end > i + 1:
            offset = np.random.randint(0, window_end - i)
            pick_idx = i + offset
            if pick_idx != i:
                temp = local_indices[i]
                local_indices[i] = local_indices[pick_idx]
                local_indices[pick_idx] = temp

        cust_idx = local_indices[i]

        if rem_demand[cust_idx] <= TOL:
            i += 1
            continue

        has_conflict = False
        if incompat.shape[0] > 0 and n_assigned > 0:
            if check_conflicts_numba(cust_idx, assigned_ids, n_assigned, incompat):
                has_conflict = True

        if has_conflict:
            i += 1
            continue

        amt = min(rem_demand[cust_idx], rem_cap)
        if amt > TOL:
            rem_cap -= amt
            assigned_ids[n_assigned] = cust_idx
            out_cust_indices[n_assigned] = cust_idx
            out_amounts[n_assigned] = amt
            n_assigned += 1
        i += 1

    return out_cust_indices[:n_assigned], out_amounts[:n_assigned]


class GreedyMultiFacilitySolver:
    def __init__(self, problem, solution, rng_seed: int = 53):
        self.problem = problem
        self.solution = solution
        self.rng = random.Random(rng_seed)

        self.fac_index = {f.id: i for i, f in enumerate(self.problem.facilities.all())}
        self.cust_index = {c.id: i for i, c in enumerate(self.problem.customers.all())}

        self.incompat_array = np.zeros((len(self.cust_index), len(self.cust_index)), dtype=np.int8)
        self._initialize_incompatibilities()

        self.sorted_customers_by_facility = None
        self.facility_aoc = None
        self.facility_order = None
        self.cost_matrix = None

        self.heuristics = heuristics.Heuristics(self)

        self.stats = {
            'precompute_time': 0.0,
            'iterations': 0,
            'facilities_opened': 0
        }

    def _initialize_incompatibilities(self):
        if hasattr(self.problem, 'incompatibilities') and self.problem.incompatibilities:
            for cust1_id, cust2_id in self.problem.incompatibilities:
                if cust1_id in self.cust_index and cust2_id in self.cust_index:
                    i, j = self.cust_index[cust1_id], self.cust_index[cust2_id]
                    self.incompat_array[i, j] = 1
                    self.incompat_array[j, i] = 1

    def _get_cost_matrix(self):
        if self.cost_matrix is not None:
            return self.cost_matrix

        facilities = list(self.problem.facilities.all())
        customers = list(self.problem.customers.all())
        F, C = len(facilities), len(customers)

        self.cost_matrix = np.zeros((F, C), dtype=np.float64)
        for f_idx, fac in enumerate(facilities):
            for c_idx, cust in enumerate(customers):
                self.cost_matrix[f_idx, c_idx] = self.problem.shipping_costs[cust.id, fac.id]

        return self.cost_matrix

    def precompute_facility_ratios(self, mode='basic', k_nearest=5):
        facilities = list(self.problem.facilities.all())
        opening_costs = np.array([getattr(f, "opening_cost", 0.0) for f in facilities])
        capacities = np.maximum(np.array([getattr(f, "capacity", 1.0) for f in facilities]), 1.0)

        aoc = opening_costs / capacities

        if mode in ['global', 'local']:
            cm = self._get_cost_matrix()
            if mode == 'global':
                aoc += np.mean(cm, axis=1)
            elif mode == 'local':
                k = min(k_nearest, cm.shape[1])
                partitioned = np.partition(cm, k - 1, axis=1)[:, :k]
                aoc += np.mean(partitioned, axis=1)

        self.facility_aoc = aoc
        self.facility_order = np.argsort(self.facility_aoc)

    def precompute_customer_sorting(self):
        cm = self._get_cost_matrix()
        F = cm.shape[0]
        self.sorted_customers_by_facility = {}
        for fac_idx in range(F):
            sorted_indices = np.argsort(cm[fac_idx, :])
            self.sorted_customers_by_facility[fac_idx] = (cm[fac_idx, sorted_indices], sorted_indices)

    def solve_greedy_multiple_facility(self, mode='basic', k=5):
        start_time = time.time()
        self.cost_matrix = None

        self.precompute_facility_ratios(mode=mode, k_nearest=k)
        self.precompute_customer_sorting()

        facilities = list(self.problem.facilities.all())
        customers = list(self.problem.customers.all())
        rem_demand = np.array([getattr(c, "demand", 0.0) for c in customers], dtype=float)

        jit_incompat = self.incompat_array if self.incompat_array is not None else np.zeros((0, 0), dtype=np.int8)
        dummy_buffer = np.zeros(len(customers), dtype=np.int32)

        for fac_idx in self.facility_order:
            cap = getattr(facilities[fac_idx], "capacity", 0.0)
            if cap <= 1e-9: continue

            _, sorted_cust_indices = self.sorted_customers_by_facility[fac_idx]
            seed = self.rng.randint(0, 1000000)

            assigned_indices, assigned_amounts = solve_facility_numba(
                fac_idx, cap, sorted_cust_indices, rem_demand,
                jit_incompat, dummy_buffer, seed
            )

            if len(assigned_indices) > 0:
                for i in range(len(assigned_indices)):
                    c_idx, amt = assigned_indices[i], assigned_amounts[i]
                    rem_demand[c_idx] -= amt
                    self.solution.add_assignment(customers[c_idx].id, facilities[fac_idx].id, amt)
                self.stats['facilities_opened'] += 1

        self.stats['total_time'] = time.time() - start_time
        return self.solution

    def has_conflict(self, cust_id, already_assigned_ids):
        if self.incompat_array is not None and cust_id in self.cust_index:
            cust_idx = self.cust_index[cust_id]
            for other_id in already_assigned_ids:
                if other_id in self.cust_index:
                    if self.incompat_array[cust_idx, self.cust_index[other_id]] == 1:
                        return True
        return False