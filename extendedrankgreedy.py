import numpy as np
import random
import time
from typing import Dict, Any
from numba import njit, int32, float64, boolean, uint32


# ==========================================
# 1. OPTIMIZED NUMBA KERNELS
# ==========================================

@njit(cache=True)
def _get_random_index(state: uint32, limit: int32) -> (uint32, int32):
    """
    Fast Xorshift RNG to replace the memory-heavy matrix lookup.
    Returns new state and a random integer in [0, limit-1].
    """
    x = state
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    # fast modulo bias is acceptable for heuristic RCL
    return x, int32(x % uint32(limit))


@njit(cache=True)
def numba_main_loop(
        F: int32,
        C: int32,
        facility_order: np.ndarray,  # (F,) int32
        cost_sorted: np.ndarray,  # (F, C) int32
        rank_matrix: np.ndarray,  # (C, F) int32
        incompat_array: np.ndarray,  # (C, C) int8
        rem_cap: np.ndarray,  # (F,) float64
        rem_demand: np.ndarray,  # (C,) float64
        fac_ids: np.ndarray,  # (F,) int32
        cust_ids: np.ndarray,  # (C,) int32
        rank_threshold: int32,
        TOL: float64
):
    max_assign = F * C
    assigns = np.zeros((max_assign, 3), dtype=np.float64)
    assign_count = 0

    facility_assignments = np.full((F, C), -1, dtype=np.int32)
    assignment_counts = np.zeros(F, dtype=np.int32)

    # Track global demand to pass to fallback (Optimization)
    total_remaining_demand = 0.0
    for c in range(C):
        total_remaining_demand += rem_demand[c]

    for idx in range(F):
        fi = facility_order[idx]
        if rem_cap[fi] <= TOL:
            continue

        # Iterate through pre-sorted customers for this facility
        for k in range(C):
            cust_idx = cost_sorted[fi, k]
            if cust_idx < 0:
                break

            # Demand/Cap Check
            if rem_demand[cust_idx] <= TOL:
                continue
            if rem_cap[fi] <= TOL:
                break  # Facility full, move to next facility

            # Rank Logic
            if rank_matrix[cust_idx, fi] > rank_threshold:
                continue

            # Incompatibility Check
            n_assigned = assignment_counts[fi]
            conflict = False
            for a in range(n_assigned):
                other = facility_assignments[fi, a]
                if incompat_array[cust_idx, other] != 0:
                    conflict = True
                    break
            if conflict:
                continue

            # Assignment Logic
            d_amt = rem_demand[cust_idx]
            c_amt = rem_cap[fi]
            amt = d_amt if d_amt < c_amt else c_amt

            if amt > TOL:
                assigns[assign_count, 0] = cust_ids[cust_idx]
                assigns[assign_count, 1] = fac_ids[fi]
                assigns[assign_count, 2] = amt
                assign_count += 1

                rem_cap[fi] -= amt
                rem_demand[cust_idx] -= amt
                total_remaining_demand -= amt

                facility_assignments[fi, n_assigned] = cust_idx
                assignment_counts[fi] = n_assigned + 1

                if rem_cap[fi] <= TOL:
                    break

    return assigns, assign_count, rem_cap, rem_demand, total_remaining_demand, facility_assignments, assignment_counts


@njit(cache=True)
def numba_fallback_greedy_rcl(
        F: int32,
        C: int32,
        facility_order: np.ndarray,
        cost_sorted: np.ndarray,
        incompat_array: np.ndarray,
        rem_cap: np.ndarray,
        rem_demand: np.ndarray,
        fac_ids: np.ndarray,
        cust_ids: np.ndarray,
        # rand_choice_matrix REMOVED
        rcl_size: int32,
        TOL: float64,
        seed: uint32,  # NEW: Seed for internal RNG
        facility_assignments: np.ndarray,  # Passed from main loop to continue state
        assignment_counts: np.ndarray,
        total_remaining_demand: float64  # Passed from main loop
):
    max_assign = F * C
    # We can create a new buffer or reuse. Creating new for safety/simplicity
    assigns = np.zeros((max_assign, 3), dtype=np.float64)
    assign_count = 0

    # Optimization: Pre-allocate candidates buffer ONCE
    # This prevents memory thrashing inside the loop
    candidates = np.empty(rcl_size, dtype=np.int32)
    rng_state = seed

    for idx in range(F):
        # Global termination check (O(1) instead of O(C))
        if total_remaining_demand <= TOL:
            break

        fi = facility_order[idx]
        if rem_cap[fi] <= TOL:
            continue

        while rem_cap[fi] > TOL and total_remaining_demand > TOL:
            cand_count = 0

            # --- Build RCL ---
            for k in range(C):
                cust_idx = cost_sorted[fi, k]
                if cust_idx < 0:
                    break  # End of sorted list

                if rem_demand[cust_idx] <= TOL:
                    continue

                # Incompat check
                n_assigned = assignment_counts[fi]
                conflict = False
                for a in range(n_assigned):
                    other = facility_assignments[fi, a]
                    if incompat_array[cust_idx, other] != 0:
                        conflict = True
                        break

                if not conflict:
                    candidates[cand_count] = cust_idx
                    cand_count += 1
                    if cand_count >= rcl_size:
                        break

            # If no candidates found for this facility, move to next
            if cand_count == 0:
                break

            # --- RCL Selection (Optimized) ---
            # Use internal RNG instead of memory lookup
            rng_state, pick_index = _get_random_index(rng_state, int32(cand_count))
            chosen_cust = candidates[pick_index]

            # --- Apply Assignment ---
            d_amt = rem_demand[chosen_cust]
            c_amt = rem_cap[fi]
            amt = d_amt if d_amt < c_amt else c_amt

            if amt > TOL:
                assigns[assign_count, 0] = cust_ids[chosen_cust]
                assigns[assign_count, 1] = fac_ids[fi]
                assigns[assign_count, 2] = amt
                assign_count += 1

                rem_cap[fi] -= amt
                rem_demand[chosen_cust] -= amt
                total_remaining_demand -= amt

                n_assigned = assignment_counts[fi]
                facility_assignments[fi, n_assigned] = chosen_cust
                assignment_counts[fi] = n_assigned + 1
            else:
                break  # Should not happen due to TOL checks, but safe break

    return assigns, assign_count, rem_cap, rem_demand


# ==========================================
# 2. OPTIMIZED PYTHON CLASS
# ==========================================

class ExtendedRankGreedySolver:
    def __init__(self, problem, solution,
                 rank_cutoff_X: float = 0.2,
                 rng_seed: int = 53,
                 randomize_facility_order: bool = True,
                 randomize_customer_order: bool = True,
                 use_random_rcl: bool = True,
                 rcl_size: int = 3,
                 heuristic_mode: int = 1):

        self.problem = problem
        self.solution = solution
        self.rank_cutoff_X = rank_cutoff_X
        self.rng = random.Random(rng_seed)
        self.numpy_rng = np.random.RandomState(rng_seed)  # For seeds

        if not (0.0 < rank_cutoff_X <= 1.0):
            raise ValueError("rank_cutoff_X must be in (0, 1].")

        self.randomize_facility_order = randomize_facility_order
        self.randomize_customer_order = randomize_customer_order
        self.use_random_rcl = use_random_rcl
        self.rcl_size = int(rcl_size)
        self.heuristic_mode = heuristic_mode

        # Structures
        self.facility_order = None
        self.cost_matrix = None
        self.cost_sorted_array = None
        self.rank_matrix = None
        self.incompat_array = None
        self.cust_index = {}
        self.fac_index = {}

        self.stats = {
            'precompute_time': 0.0,
            'numba_main_time': 0.0,
            'numba_fallback_time': 0.0,
            'total_time': 0.0,
            'assignments_main': 0,
            'assignments_fallback': 0
        }

    def _randomize_within_groups(self, sorted_indices: np.ndarray, values: np.ndarray,
                                 tolerance: float = 0.05) -> np.ndarray:
        # Keeps Python implementation as this runs only once during setup
        result = []
        i = 0
        n = len(sorted_indices)
        while i < n:
            group = [sorted_indices[i]]
            base_val = values[sorted_indices[i]]
            j = i + 1
            # Avoid division by zero with safe denominator
            denom = abs(base_val) + 1e-9
            while j < n:
                curr_val = values[sorted_indices[j]]
                if abs(curr_val - base_val) / denom <= tolerance:
                    group.append(sorted_indices[j])
                    j += 1
                else:
                    break
            self.rng.shuffle(group)
            result.extend(group)
            i = j
        return np.array(result, dtype=np.int32)

    def build_cost_matrix(self):
        # Use simple lists for iteration to avoid Django QuerySet overhead
        facilities = list(self.problem.facilities.all())
        customers = list(self.problem.customers.all())
        F = len(facilities)
        C = len(customers)

        self.cust_index = {c.id: i for i, c in enumerate(customers)}
        self.fac_index = {f.id: i for i, f in enumerate(facilities)}

        # Initialize with infinity
        cost_matrix = np.full((F, C), np.inf, dtype=np.float64)

        # Optimized dictionary iteration
        shipping = self.problem.shipping_costs
        for (cust_id, fac_id), cost in shipping.items():
            # Direct lookup is faster than .get()
            try:
                c_i = self.cust_index[cust_id]
                f_i = self.fac_index[fac_id]
                cost_matrix[f_i, c_i] = cost
            except KeyError:
                continue

        self.cost_matrix = cost_matrix

    def compute_facility_priority(self):
        facilities = list(self.problem.facilities.all())
        F = len(facilities)

        # Vectorized property access
        opening_costs = np.array([getattr(f, "opening_cost", 0.0) for f in facilities], dtype=float)
        capacities = np.array([getattr(f, "capacity", 1.0) for f in facilities], dtype=float)
        np.maximum(capacities, 1.0, out=capacities)  # In-place max

        base_aoc = opening_costs / capacities
        final_scores = base_aoc.copy()

        # Heuristic Modes
        if self.heuristic_mode in [2, 3]:
            # Vectorized mean ignoring Infs
            masked_cost = np.ma.masked_invalid(self.cost_matrix)

            if self.heuristic_mode == 2:
                # Average of valid transport costs
                transport_avgs = np.mean(masked_cost, axis=1).filled(1e9)
                final_scores += transport_avgs

            elif self.heuristic_mode == 3:
                # Average of closest 5
                # Using partition is O(C) vs Sort O(C log C)
                k = 5
                transport_heur = np.full(F, 1e9, dtype=float)
                for f in range(F):
                    row = self.cost_matrix[f]
                    valid = row[row != np.inf]
                    if valid.size > 0:
                        if valid.size > k:
                            # small partition is very fast
                            transport_heur[f] = np.mean(np.partition(valid, k)[:k])
                        else:
                            transport_heur[f] = np.mean(valid)
                final_scores += transport_heur

        sorted_indices = np.argsort(final_scores)
        if self.randomize_facility_order:
            self.facility_order = self._randomize_within_groups(sorted_indices, final_scores)
        else:
            self.facility_order = sorted_indices.astype(np.int32)

    def prepare_customer_arrays(self):
        start = time.time()
        F, C = self.cost_matrix.shape

        # Pre-fill with -1
        cost_sorted = np.full((F, C), -1, dtype=np.int32)
        rank_matrix = np.zeros((C, F), dtype=np.int32)

        # 1. Sort Customers per Facility
        for f in range(F):
            row = self.cost_matrix[f]
            # stable argsort usually better for ties
            sorted_idx = np.argsort(row)

            if self.randomize_customer_order:
                sorted_idx = self._randomize_within_groups(sorted_idx, row)

            cost_sorted[f, :len(sorted_idx)] = sorted_idx

        self.cost_sorted_array = cost_sorted

        # 2. Sort Facilities per Customer (Ranking)
        # Transpose logic: iterate C to rank F
        for c in range(C):
            col = self.cost_matrix[:, c]
            sorted_facs = np.argsort(col)

            if self.randomize_customer_order:
                sorted_facs = self._randomize_within_groups(sorted_facs, col)

            # Map facility index to its Rank (1-based)
            # Efficient numpy assignment
            rank_matrix[c, sorted_facs] = np.arange(1, F + 1, dtype=np.int32)

        self.rank_matrix = rank_matrix
        self.stats['precompute_time'] += (time.time() - start)

    def precompute_incompatibilities(self):
        customers = list(self.problem.customers.all())
        C = len(customers)
        # Using dense matrix (int8) for O(1) lookup speed.
        # Note: 10k customers = 100MB RAM. Manageable.
        self.incompat_array = np.zeros((C, C), dtype=np.int8)

        if hasattr(self.problem, 'incompatibilities') and self.problem.incompatibilities:
            for c1, c2 in self.problem.incompatibilities:
                if c1 in self.cust_index and c2 in self.cust_index:
                    i, j = self.cust_index[c1], self.cust_index[c2]
                    self.incompat_array[i, j] = 1
                    self.incompat_array[j, i] = 1

    def solve_extended_greedy(self, use_fallback: bool = True):
        total_start = time.time()

        # 1. Pre-computation
        self.build_cost_matrix()
        self.compute_facility_priority()
        self.prepare_customer_arrays()
        self.precompute_incompatibilities()

        # 2. Extract Data for Numba (Ensure Contiguous Memory)
        facilities = list(self.problem.facilities.all())
        customers = list(self.problem.customers.all())
        F = len(facilities)
        C = len(customers)

        # Enforce C-order contiguous arrays for maximum SIMD/Cache performance
        rem_cap = np.ascontiguousarray([getattr(f, "capacity", 0.0) for f in facilities], dtype=np.float64)
        rem_demand = np.ascontiguousarray([getattr(c, "demand", 0.0) for c in customers], dtype=np.float64)
        fac_ids = np.ascontiguousarray([int(f.id) for f in facilities], dtype=np.int32)
        cust_ids = np.ascontiguousarray([int(c.id) for c in customers], dtype=np.int32)

        fac_order = np.ascontiguousarray(self.facility_order, dtype=np.int32)
        cost_sorted = np.ascontiguousarray(self.cost_sorted_array, dtype=np.int32)
        rank_mat = np.ascontiguousarray(self.rank_matrix, dtype=np.int32)
        incompat = np.ascontiguousarray(self.incompat_array, dtype=np.int8)

        rank_thresh = int(self.rank_cutoff_X * F)
        TOL = 1e-9

        # 3. Main Numba Kernel
        t0 = time.time()
        (assigns, count, rem_cap, rem_demand,
         total_rem_demand, fac_assigns, assign_counts) = numba_main_loop(
            np.int32(F), np.int32(C), fac_order, cost_sorted, rank_mat,
            incompat, rem_cap, rem_demand, fac_ids, cust_ids,
            np.int32(rank_thresh), float(TOL)
        )
        self.stats['numba_main_time'] = time.time() - t0
        self.stats['assignments_main'] = count

        # Apply Main Assignments
        for i in range(count):
            self.solution.add_assignment(int(assigns[i, 0]), int(assigns[i, 1]), float(assigns[i, 2]))

        # 4. Fallback Kernel (If needed)
        if use_fallback and total_rem_demand > TOL:
            t2 = time.time()

            # Use seed int instead of matrix
            seed = np.uint32(self.numpy_rng.randint(0, 2 ** 32 - 1))

            fb_assigns, fb_count, _, _ = numba_fallback_greedy_rcl(
                np.int32(F), np.int32(C), fac_order, cost_sorted, incompat,
                rem_cap, rem_demand, fac_ids, cust_ids,
                np.int32(self.rcl_size), float(TOL), seed,
                fac_assigns, assign_counts, total_rem_demand
            )
            self.stats['numba_fallback_time'] = time.time() - t2
            self.stats['assignments_fallback'] = fb_count

            # Apply Fallback Assignments
            for i in range(fb_count):
                self.solution.add_assignment(int(fb_assigns[i, 0]), int(fb_assigns[i, 1]), float(fb_assigns[i, 2]))

        self.stats['total_time'] = time.time() - total_start
        return self.solution