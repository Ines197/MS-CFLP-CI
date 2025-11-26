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
        assigned_tracker: np.ndarray,  # Array to track who is assigned to this facility
        seed: int
):
    """
    Numba-optimized inner loop for a single facility with RCL.
    Returns: (indices_assigned, amounts_assigned, count)
    """
    np.random.seed(seed)

    # 1. Setup local variables
    rem_cap = capacity
    n_assigned = 0
    assigned_ids = assigned_tracker  # Re-use array to save alloc

    # Output storage (max possible is all customers)
    out_cust_indices = np.zeros(len(sorted_indices), dtype=np.int32)
    out_amounts = np.zeros(len(sorted_indices), dtype=np.float64)

    TOL = 1e-9
    k = 5
    max_len = len(sorted_indices)

    # Make a local copy of indices so we can SWAP without ruining the global sort for other facilities
    # Note: If memory is tight, we can skip copy and re-sort later, but copy is safer.
    local_indices = sorted_indices.copy()

    i = 0
    while rem_cap > TOL and i < max_len:

        # --- OPTIMIZED RCL (SWAP METHOD) ---
        # Define window end
        window_end = min(i + k, max_len)

        # If we have a choice range
        if window_end > i + 1:
            # Pick random offset from 0 to (window_len - 1)
            offset = np.random.randint(0, window_end - i)
            pick_idx = i + offset

            # SWAP: Move chosen candidate to current position 'i'
            # The one currently at 'i' moves to 'pick_idx' to be processed later
            if pick_idx != i:
                temp = local_indices[i]
                local_indices[i] = local_indices[pick_idx]
                local_indices[pick_idx] = temp

        # Current candidate is now guaranteed to be at local_indices[i]
        cust_idx = local_indices[i]

        # Check demand
        if rem_demand[cust_idx] <= TOL:
            i += 1
            continue

        # Check conflicts
        # Assuming incompat is passed. If None, pass dummy array or handle logic outside.
        # Here assuming incompat is a valid 2D array or handling logic exists.
        has_conflict = False
        if incompat.shape[0] > 0 and n_assigned > 0:
            if check_conflicts_numba(cust_idx, assigned_ids, n_assigned, incompat):
                has_conflict = True

        if has_conflict:
            i += 1
            continue

        # Assign
        amt = min(rem_demand[cust_idx], rem_cap)
        if amt > TOL:
            rem_cap -= amt
            # We don't update rem_demand here directly if we want to return results,
            # but since we are sequential, we can return the decrement needed.

            assigned_ids[n_assigned] = cust_idx

            out_cust_indices[n_assigned] = cust_idx
            out_amounts[n_assigned] = amt
            n_assigned += 1

        i += 1

    return out_cust_indices[:n_assigned], out_amounts[:n_assigned]


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
        start_time = time.time()

        facilities = list(self.problem.facilities.all())
        F = len(facilities)

        self.fac_index = {f.id: i for i, f in enumerate(facilities)}

        opening_costs = np.array([getattr(f, "opening_cost", 0.0) for f in facilities])
        capacities = np.maximum(np.array([getattr(f, "capacity", 1.0) for f in facilities]), 1.0)

        self.facility_aoc = opening_costs / capacities
        self.facility_order = np.argsort(self.facility_aoc)

        self.stats['precompute_time'] = time.time() - start_time

    def precompute_customer_sorting(self):
        start_time = time.time()

        facilities = list(self.problem.facilities.all())
        customers = list(self.problem.customers.all())
        F = len(facilities)
        C = len(customers)

        self.cust_index = {c.id: i for i, c in enumerate(customers)}

        cost_matrix = np.zeros((F, C), dtype=float)
        for fac_idx, fac in enumerate(facilities):
            for cust_idx, cust in enumerate(customers):
                cost_matrix[fac_idx, cust_idx] = self.problem.shipping_costs[cust.id, fac.id]

        self.sorted_customers_by_facility = {}
        for fac_idx in range(F):
            sorted_indices = np.argsort(cost_matrix[fac_idx, :])
            self.sorted_customers_by_facility[fac_idx] = (
                cost_matrix[fac_idx, sorted_indices],  # costs
                sorted_indices  # customer indices
            )

        self.stats['customer_sort_time'] = time.time() - start_time


    def precompute_incompatibilities(self):
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
        start_time = time.time()

        # Ensure data setup
        self.precompute_facility_ratios()
        self.precompute_customer_sorting()
        self.precompute_incompatibilities()

        facilities = list(self.problem.facilities.all())
        customers = list(self.problem.customers.all())

        rem_demand = np.array([getattr(c, "demand", 0.0) for c in customers], dtype=float)

        # Handle incompat array for Numba (pass empty if None)
        if self.incompat_array is None:
            jit_incompat = np.zeros((0, 0), dtype=np.int8)
        else:
            jit_incompat = self.incompat_array

        # Re-usable buffer for conflict checking
        dummy_buffer = np.zeros(len(customers), dtype=np.int32)

        for fac_idx in self.facility_order:
            cap = getattr(facilities[fac_idx], "capacity", 0.0)
            if cap <= 1e-9: continue

            # Get sorted indices for this facility
            _, sorted_cust_indices = self.sorted_customers_by_facility[fac_idx]

            # --- CALL NUMBA ---
            # Generate a distinct seed per facility to ensure randomness varies
            seed = self.rng.randint(0, 1000000)

            assigned_indices, assigned_amounts = solve_facility_numba(
                fac_idx,
                cap,
                sorted_cust_indices,
                rem_demand,
                jit_incompat,
                dummy_buffer,
                seed
            )

            # --- UPDATE PYTHON OBJECTS ---
            # Update state based on Numba results
            for i in range(len(assigned_indices)):
                c_idx = assigned_indices[i]
                amt = assigned_amounts[i]

                # Update demand vector
                rem_demand[c_idx] -= amt

                # Update Solution Object
                self.solution.add_assignment(c_idx, fac_idx, amt)
                self.stats['facilities_opened'] += 1  # Approximation

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