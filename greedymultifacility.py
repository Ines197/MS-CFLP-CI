import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
import heuristics
from typing import Dict, Any, List, Set, Tuple
from numba import njit
import time

@njit(cache=True)
def check_conflicts_numba(cust_idx: int, assigned: np.ndarray, n_assigned: int, incompat: np.ndarray) -> bool:
    """Fast conflict checking using Numba JIT compilation."""
    for i in range(n_assigned):
        if incompat[cust_idx, assigned[i]]:
            return True
    return False


class GreedyMultiFacilitySolver:
    def __init__(self, problem, solution, rng_seed: int = 53):
        self.sorted_customers_by_facility = None
        self.facility_aoc = None
        self.facility_order = None
        self.problem = problem
        self.solution = solution
        self.fac_index: Dict[Any, int] = {}
        self.cust_index: Dict[Any, int] = {}
        self.rng = random.Random(rng_seed)
        self.heuristics = heuristics.Heuristics(self)
        self.incompat_array = None

        # Benchmarking
        self.stats = {
            'precompute_time': 0.0,
            'iterations': 0,
            'tau_filter_hits': 0,
            'conflict_checks': 0,
            'min_cost_updates': 0,
            'facilities_opened': 0
        }

    def precompute_facility_ratios(self):
        """Vectorized facility ratio computation."""
        start_time = time.time()

        facilities = list(self.problem.facilities.all())
        F = len(facilities)

        self.fac_index = {f.id: i for i, f in enumerate(facilities)}

        # Vectorized computation
        opening_costs = np.array([getattr(f, "opening_cost", 0.0) for f in facilities])
        capacities = np.maximum(np.array([getattr(f, "capacity", 1.0) for f in facilities]), 1.0)

        self.facility_aoc = opening_costs / capacities
        self.facility_order = np.argsort(self.facility_aoc)

        self.stats['precompute_time'] = time.time() - start_time

    def precompute_customer_sorting(self):
        """Optimized customer sorting with cost matrix."""
        start_time = time.time()

        facilities = list(self.problem.facilities.all())
        customers = list(self.problem.customers.all())
        F = len(facilities)
        C = len(customers)

        self.cust_index = {c.id: i for i, c in enumerate(customers)}

        # Build cost matrix once (F x C)
        cost_matrix = np.zeros((F, C), dtype=float)
        for fac_idx, fac in enumerate(facilities):
            for cust_idx, cust in enumerate(customers):
                cost_matrix[fac_idx, cust_idx] = self.problem.get_shipping_costs(cust.id, fac.id)

        # Sort customers for each facility using argsort
        self.sorted_customers_by_facility = {}
        for fac_idx in range(F):
            sorted_indices = np.argsort(cost_matrix[fac_idx, :])
            # Store as numpy arrays for faster iteration
            self.sorted_customers_by_facility[fac_idx] = (
                cost_matrix[fac_idx, sorted_indices],  # costs
                sorted_indices  # customer indices
            )

        self.stats['customer_sort_time'] = time.time() - start_time

    def precompute_incompatibilities(self):
        """Build incompatibility matrix for O(1) lookups."""
        customers = list(self.problem.customers.all())
        C = len(customers)

        self.incompat_array = np.zeros((C, C), dtype=np.int8)  # Use int8 to save memory

        if hasattr(self.problem, 'incompatibilities') and self.problem.incompatibilities:
            if isinstance(self.problem.incompatibilities, set):
                for cust1_id, cust2_id in self.problem.incompatibilities:
                    if cust1_id in self.cust_index and cust2_id in self.cust_index:
                        i = self.cust_index[cust1_id]
                        j = self.cust_index[cust2_id]
                        self.incompat_array[i, j] = 1
                        self.incompat_array[j, i] = 1

    def solve_greedy_multiple_facility(self):
        """Optimized greedy construction with minimal overhead."""
        start_time = time.time()

        self.precompute_facility_ratios()
        self.precompute_customer_sorting()
        self.precompute_incompatibilities()

        facilities = list(self.problem.facilities.all())
        customers = list(self.problem.customers.all())

        F = len(facilities)
        C = len(customers)

        # Initialize remaining capacities and demands
        rem_cap = np.array([getattr(f, "capacity", 0.0) for f in facilities], dtype=float)
        rem_demand = np.array([getattr(c, "demand", 0.0) for c in customers], dtype=float)

        # Use numpy arrays instead of sets for faster operations
        facility_assignments = [np.empty(C, dtype=np.int32) for _ in range(F)]
        assignment_counts = np.zeros(F, dtype=np.int32)

        # Pre-allocate for batch assignments
        TOL = 1e-9

        # Process facilities in order of increasing AOC
        for fac_idx in self.facility_order:
            self.stats['iterations'] += 1

            # Check if facility is full
            if rem_cap[fac_idx] <= TOL:
                continue

            # Get pre-sorted customers for this facility
            costs, sorted_cust_indices = self.sorted_customers_by_facility[fac_idx]

            assigned_customers = facility_assignments[fac_idx]
            n_assigned = assignment_counts[fac_idx]

            # Process customers in order of increasing transportation cost
            for i in range(len(sorted_cust_indices)):
                cust_idx = sorted_cust_indices[i]

                # Skip customers with no remaining demand
                if rem_demand[cust_idx] <= TOL:
                    continue

                # Check if facility is full
                if rem_cap[fac_idx] <= TOL:
                    break

                # Fast conflict check using Numba
                if self.incompat_array is not None and n_assigned > 0:
                    if check_conflicts_numba(cust_idx, assigned_customers, n_assigned, self.incompat_array):
                        self.stats['conflict_checks'] += 1
                        continue

                # Assign as much as possible
                amt = min(rem_demand[cust_idx], rem_cap[fac_idx])

                if amt > TOL:
                    # Record assignment
                    self.solution.assign(fac_idx, cust_idx, amt)

                    # Update remaining capacity and demand
                    rem_cap[fac_idx] -= amt
                    rem_demand[cust_idx] -= amt

                    # Track assignment
                    assigned_customers[n_assigned] = cust_idx
                    assignment_counts[fac_idx] += 1
                    n_assigned += 1

                    self.stats['facilities_opened'] += 1

        self.stats['total_time'] = time.time() - start_time
        return self.solution

    def has_conflict(self, cust_id, already_assigned_ids):
        """
        Compatibility method for external usage.
        """
        if not hasattr(self.problem, 'incompatibilities') or not self.problem.incompatibilities:
            return False

        if isinstance(self.problem.incompatibilities, set):
            return any((cust_id, other) in self.problem.incompatibilities or
                       (other, cust_id) in self.problem.incompatibilities
                       for other in already_assigned_ids)

        if self.incompat_array is not None and cust_id in self.cust_index:
            cust_idx = self.cust_index[cust_id]
            for other_id in already_assigned_ids:
                if other_id in self.cust_index:
                    other_idx = self.cust_index[other_id]
                    if self.incompat_array[cust_idx, other_idx] == 1:
                        return True
            return False

        return False