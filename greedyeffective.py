import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
import heuristics
from typing import Dict, Any
from numba import njit
import time

@njit
def has_conflict_numba(cust_id, assigned_ids, incompat_array):
    """Brza provera konflikta korišćenjem numba."""
    for other_id in assigned_ids:
        if incompat_array[cust_id, other_id] == 1:
            return True
    return False

@njit
def has_conflict_by_ids(cust_idx, assigned_indices, incompat_array):
    """
    Numba funkcija za proveru konflikta.
    Koristi indekse umesto ID-jeva.
    """
    for other_idx in assigned_indices:
        if incompat_array[cust_idx, other_idx] == 1:
            return True
    return False

class GreedyEffectiveSolver:
    def __init__(self, problem, solution, rng_seed: int = 53):
        self.problem = problem
        self.solution = solution
        self.effective_cost_matrix = None  # numpy array F x C
        self.fac_index: Dict[Any, int] = {}
        self.cust_index: Dict[Any, int] = {}
        self.rng = random.Random(rng_seed)
        self.heuristics = heuristics.Heuristics(self)

        # Inkompatibilnost array (inicijalizuje se u precompute)
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

    def compute_adaptive_tau(self, base_tau: float = 2.0) -> float:
        """
        Calculates adaptive tau based on instance characteristics, with aggressive
        sensitivity to the number of facilities (F) for increased exploration.

        Args:
            base_tau: Starting tau value (default: 2.0, effective for F=50)

        Returns:
            Adjusted tau value
        """
        facilities = list(self.problem.facilities.all())
        customers = list(self.problem.customers.all())
        F, C = len(facilities), len(customers)

        # --- Factor 1: Instance Size (Aggressive Facility Count Scaling) ---
        fac_factor = 1.0
        # Increased thresholds and multipliers for larger instances
        if F >= 800:
            fac_factor = 2.2  # Allows base_tau to double (2.0 * 2.0 = 4.0 before other factors)
        elif F > 500:
            fac_factor = 1.8
        elif F > 300:
            fac_factor = 1.52
        elif F > 200:
            fac_factor = 1.38
        elif F > 100:
            fac_factor = 1.25
        elif F > 50:
            fac_factor = 1.05
        # For F <= 50, fac_factor remains 1.0

        # Optional secondary scaling by total vars (F*C)
        total_vars = F * C
        size_factor_fc = 1.0
        if total_vars > 100000:
            size_factor_fc = 1.1
        elif total_vars > 50000:
            size_factor_fc = 1.05
        elif total_vars < 1000:
            size_factor_fc = 0.95

        size_factor = fac_factor * size_factor_fc

        # --- Factor 2: Facility/Customer ratio (no change) ---
        ratio = F / C if C > 0 else 1.0
        ratio_factor = 1.0
        if ratio > 0.5:
            ratio_factor = 1.15
        elif ratio < 0.1:
            ratio_factor = 1.2

        # --- Factor 3: Cost variance (no change) ---
        variance_factor = 1.0
        if self.effective_cost_matrix is not None:
            finite_costs = self.effective_cost_matrix[np.isfinite(self.effective_cost_matrix)]
            if len(finite_costs) > 0:
                cost_std = np.std(finite_costs)
                cost_mean = np.mean(finite_costs)
                cv = cost_std / cost_mean if cost_mean > 0 else 0

                if cv > 1.0:
                    variance_factor = 1.2
                elif cv > 0.5:
                    variance_factor = 1.1
                elif cv < 0.2:
                    variance_factor = 0.95

        # --- Factor 4: Capacity tightness (no change) ---
        total_demand = sum(c.demand for c in customers)
        total_capacity = sum(f.capacity for f in facilities)
        tightness = total_demand / total_capacity if total_capacity > 0 else 1.0

        tightness_factor = 1.0
        if tightness > 0.9:
            tightness_factor = 1.25
        elif tightness > 0.7:
            tightness_factor = 1.15
        elif tightness < 0.3:
            tightness_factor = 0.9

        # --- Factor 5: Incompatibility density (no change) ---
        incompat_factor = 1.0
        if hasattr(self.problem, 'incompatibilities') and self.problem.incompatibilities:
            num_incompats = len(self.problem.incompatibilities)
            max_possible = C * (C - 1) / 2
            density = num_incompats / max_possible if max_possible > 0 else 0

            if density > 0.1:
                incompat_factor = 1.3
            elif density > 0.05:
                incompat_factor = 1.15

        # --- Combine all factors ---
        adaptive_tau = (base_tau * size_factor * ratio_factor * variance_factor *
                        tightness_factor * incompat_factor)

        # Significantly increased max clamp to allow for high tau values
        adaptive_tau = max(1.2, min(7.0, adaptive_tau))

        return adaptive_tau

    def precompute_effective_costs(self):
        """Paralelno izračunavanje efektivnih troškova i inkompatibilnosti."""
        start_time = time.time()

        facilities = list(self.problem.facilities.all())
        customers = list(self.problem.customers.all())
        F, C = len(facilities), len(customers)

        self.fac_index = {f.id: i for i, f in enumerate(facilities)}
        self.cust_index = {c.id: j for j, c in enumerate(customers)}

        eff_matrix = np.zeros((F, C), dtype=float)
        shipping = getattr(self.problem, "shipping_costs", {})

        def compute_for_fac(fac):
            i = self.fac_index[fac.id]
            base_cost = (getattr(fac, "opening_cost", 0.0) /
                         max(getattr(fac, "capacity", 1.0), 1.0))
            for cust in customers:
                j = self.cust_index[cust.id]
                s_cost = shipping.get((cust.id, fac.id),
                                      shipping.get((fac.id, cust.id), 0.0))
                eff_matrix[i, j] = base_cost + s_cost

        max_workers = min(32, max(1, F))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(compute_for_fac, facilities))

        self.effective_cost_matrix = eff_matrix

        self.incompat_array = np.zeros((C, C), dtype=np.uint8)
        if hasattr(self.problem, "incompatibilities") and self.problem.incompatibilities:
            for a, b in self.problem.incompatibilities:
                if a in self.cust_index and b in self.cust_index:
                    i, j = self.cust_index[a], self.cust_index[b]
                    self.incompat_array[i, j] = 1
                    self.incompat_array[j, i] = 1

        self.stats['precompute_time'] = time.time() - start_time

    def solve_greedy_global_effective(self,
                                      alpha: float = 0.25,
                                      beta: float = 0.228,
                                      tau: float = None,
                                      top_k: int = 40,
                                      update_mode: str = 'incremental'):
        """
        Optimizovani Global Greedy sa tau filterom i hybrid fixom.

        Args:
            alpha: Penalizacija za zatvorene fabrike
            beta: RCL diversifikacija
            tau: Tau filter threshold (dozvoli troškove do tau × min)
            top_k: Broj najboljih kandidata za razmatranje
            update_mode: 'incremental', 'batch', 'lazy', ili 'hybrid'
        """

        #print(f"\n{'=' * 70}")
        #print(f"🚀 Optimizovani Global Greedy")
        #print(f"{'=' * 70}")
        #print(f"Parametri: α={alpha}, β={beta}, τ={tau}, top_k={top_k}")
        #print(f"Update mode: {update_mode}")
        #print(f"{'=' * 70}\n")

        start_time = time.time()
        self.precompute_effective_costs()

        if tau is None:
            tau = self.compute_adaptive_tau()
            #print(f"📊 Computed adaptive τ = {tau:.3f}")
        #else:
            #print(f"📊 Using fixed τ = {tau:.3f}")

        facilities = list(self.problem.facilities.all())
        customers = list(self.problem.customers.all())
        F, C = len(facilities), len(customers)

        # Inicijalizacija stanja
        fac_remaining = np.array([f.remaining_capacity for f in facilities], dtype=float)
        cust_remaining = np.array([c.remaining_demand for c in customers], dtype=float)
        open_fac_indices = set()
        facility_assignments = [set() for _ in range(F)]

        # ✅ FIX 2: Hybrid initialization
        # Base theoretical minimum (what we could achieve if all facilities were free to use)
        base_min_cost = np.min(self.effective_cost_matrix, axis=0).copy()
        # Actual achievable minimum (updated as facilities open)
        min_eff_cost = np.full(C, np.inf)

        active_custs = set(i for i in range(C) if cust_remaining[i] > 0)

        conflict_cache = {}
        newly_opened = []
        updates_since_refresh = 0
        update_threshold = max(3, min(10, F // 20)) # za hibrid mod
        iteration_count = 0

        # Glavna petlja
        while active_custs:
            self.stats['iterations'] += 1
            iteration_count += 1

            active_facs = np.array([i for i in range(F) if fac_remaining[i] > 0], dtype=int)
            active_cust_arr = np.array(list(active_custs), dtype=int)

            if len(active_facs) == 0 or len(active_cust_arr) == 0:
                break

            # Submatrica efektivnih troškova
            eff_sub = self.effective_cost_matrix[np.ix_(active_facs, active_cust_arr)].copy()

            # Penalizacija za zatvorene fabrike
            closed_mask = np.isin(active_facs, list(open_fac_indices), invert=True)
            if np.any(closed_mask):
                fac_caps = np.array([facilities[i].capacity for i in active_facs])
                fac_opening = np.array([facilities[i].opening_cost for i in active_facs])
                q_matrix = np.minimum(
                    cust_remaining[active_cust_arr][None, :],
                    fac_remaining[active_facs][:, None]
                )
                penalty = alpha * (q_matrix / fac_caps[:, None]) * fac_opening[:, None]
                eff_sub = eff_sub + penalty * closed_mask[:, None]

            # ✅ FIX 2: Adaptive tau filtering
            if len(open_fac_indices) == 0:
                # No facilities open yet: use base cost with relaxed tau
                effective_min = base_min_cost[active_cust_arr]
                effective_tau = tau * 3.0  # More relaxed for initial iterations
                #print(
                #    f"  Iteration {iteration_count}: Using relaxed tau ({effective_tau:.2f}) - no facilities open yet")
            else:
                # Use actual minimum from open facilities where available
                effective_min = np.where(
                    np.isfinite(min_eff_cost[active_cust_arr]),
                    min_eff_cost[active_cust_arr],
                    base_min_cost[active_cust_arr]
                )
                effective_tau = tau

            tau_threshold = effective_tau * effective_min[None, :]
            tau_mask = eff_sub <= tau_threshold
            eff_sub = np.where(tau_mask, eff_sub, np.inf)

            filtered_count = np.sum(tau_mask)
            self.stats['tau_filter_hits'] += (tau_mask.size - filtered_count)

            if not np.isfinite(eff_sub).any():
                # Lazy update: ako tau eliminiše sve, ažuriraj i pokušaj ponovo
                if update_mode == 'lazy' and len(open_fac_indices) > 0:
                    #print(f"⚙️ Lazy update triggered at iteration {self.stats['iterations']}")
                    active_cust_list = list(active_custs)
                    min_eff_cost[active_cust_list] = np.min(
                        self.effective_cost_matrix[np.ix_(
                            list(open_fac_indices),
                            active_cust_list
                        )], axis=0
                    )
                    continue
                else:
                    #print(f"⚠️ Iteracija {self.stats['iterations']}: Tau filter eliminisao sve kandidate!")
                    break

            # Top-k najboljih parova (RCL pristup)
            flat_eff = eff_sub.ravel()
            valid_count = np.isfinite(flat_eff).sum()
            top_k_actual = min(top_k, valid_count)

            if top_k_actual == 0:
                break

            best_indices = np.argpartition(flat_eff, top_k_actual - 1)[:top_k_actual]
            f_rel, c_rel = np.unravel_index(best_indices, eff_sub.shape)

            # Filtriranje po konfliktima
            valid_triples = []
            for ff, cc in zip(f_rel, c_rel):
                f_idx = active_facs[ff]
                c_idx = active_cust_arr[cc]
                cust_id = customers[c_idx].id

                cache_key = (c_idx, f_idx)
                conflict = conflict_cache.get(cache_key)

                if conflict is None:
                    assigned_arr = np.array(list(facility_assignments[f_idx]), dtype=np.int32)
                    conflict = has_conflict_by_ids(c_idx, assigned_arr, self.incompat_array)
                    conflict_cache[cache_key] = conflict
                    self.stats['conflict_checks'] += 1

                if not conflict:
                    eff_cost = eff_sub[ff, cc]
                    q = min(cust_remaining[c_idx], fac_remaining[f_idx])
                    if q > 0:
                        valid_triples.append((c_idx, f_idx, q, eff_cost))

            if not valid_triples:
                #print(f"⚠️ Iteracija {self.stats['iterations']}: Nema validnih parova bez konflikta!")
                break

            # RCL sa beta parametrom
            valid_triples.sort(key=lambda x: x[3])
            best_cost = valid_triples[0][3]
            rcl = [t for t in valid_triples if t[3] <= best_cost * (1 + beta)]
            c_idx, f_idx, q, chosen_cost = self.rng.choice(rcl)

            cust, fac = customers[c_idx], facilities[f_idx]

            # Otvaranje fabrike i ažuriranje min_eff_cost
            if f_idx not in open_fac_indices:
                fac.open()
                open_fac_indices.add(f_idx)
                self.solution.facilities_open.add(fac.id)
                newly_opened.append(f_idx)
                self.stats['facilities_opened'] += 1

                # ✅ AŽURIRANJE min_eff_cost (različiti načini)
                if update_mode == 'incremental':
                    # Brzo: samo nova fabrika
                    active_cust_list = list(active_custs)
                    new_costs = self.effective_cost_matrix[f_idx, active_cust_list]
                    min_eff_cost[active_cust_list] = np.minimum(
                        min_eff_cost[active_cust_list],
                        new_costs
                    )
                    self.stats['min_cost_updates'] += 1

                elif update_mode == 'batch':
                    # Batch: svako 5. otvaranje
                    if len(newly_opened) >= 5:
                        active_cust_list = list(active_custs)
                        min_eff_cost[active_cust_list] = np.min(
                            self.effective_cost_matrix[np.ix_(
                                list(open_fac_indices),
                                active_cust_list
                            )], axis=0
                        )
                        newly_opened.clear()
                        self.stats['min_cost_updates'] += 1

                elif update_mode == 'hybrid':
                    # Hibridno: inkrementalno + periodično puno
                    active_cust_list = list(active_custs)
                    new_costs = self.effective_cost_matrix[f_idx, active_cust_list]
                    min_eff_cost[active_cust_list] = np.minimum(
                        min_eff_cost[active_cust_list],
                        new_costs
                    )
                    updates_since_refresh += 1
                    self.stats['min_cost_updates'] += 1

                    if updates_since_refresh >= update_threshold:
                        min_eff_cost[active_cust_list] = np.min(
                            self.effective_cost_matrix[np.ix_(
                                list(open_fac_indices),
                                active_cust_list
                            )], axis=0
                        )
                        updates_since_refresh = 0


                elif update_mode == 'lazy':
                    # Lazy: only update on first facility open
                    if len(open_fac_indices) == 1:  # Just opened first facility
                        active_cust_list = list(active_custs)
                        min_eff_cost[active_cust_list] = np.min(
                            self.effective_cost_matrix[np.ix_(
                                list(open_fac_indices),
                                active_cust_list
                            )], axis=0
                        )
                        self.stats['min_cost_updates'] += 1

            # Dodela kupca fabrici
            self._assign_customer_to_facility(cust, fac)
            facility_assignments[f_idx].add(c_idx)
            fac_remaining[f_idx] = fac.remaining_capacity
            cust_remaining[c_idx] = cust.remaining_demand

            # ✅ FIX 1: Proper cache invalidation
            # Invalidate only customers that actually conflict with the newly assigned customer
            if c_idx < self.incompat_array.shape[0]:
                # Find all customers that have incompatibility with c_idx
                conflicting_custs = np.where(self.incompat_array[c_idx, :] == 1)[0]
                if len(conflicting_custs) > 0:
                    # Invalidate cache entries for conflicting customers at this facility
                    keys_to_remove = [
                        k for k in conflict_cache
                        if k[1] == f_idx and k[0] in conflicting_custs
                    ]
                    for k in keys_to_remove:
                        conflict_cache.pop(k, None)

            if cust_remaining[c_idx] <= 0:
                active_custs.discard(c_idx)
                min_eff_cost[c_idx] = np.inf  # Označi kao završeno

        # Fallback ako ostane neraspoređeno
        remaining = self.problem.customers.total_remaining_demand()
        if remaining > 0:
            #print(f"\n⚠️ Pokrećem fallback greedy (preostalo: {remaining:.2f})")
            self.solve_greedy()

        total_time = time.time() - start_time

        # Statistika
        #print(f"\n{'=' * 70}")
        #print(f"📊 REZULTATI I STATISTIKA")
        #print(f"{'=' * 70}")
        #print(f"✅ Validnost: {'VALID ✓' if self.solution.is_valid() else 'INVALID ✗'}")
        #print(f"💰 Ukupan trošak: {self.solution.total_cost():,.2f}")
        #print(f"🏭 Otvorenih fabrika: {self.stats['facilities_opened']}")
        #print(f"🔄 Iteracija: {self.stats['iterations']}")
        #print(f"📦 Preostala potražnja: {remaining:.2f}")
        #print(f"\n⏱️ PERFORMANSE:")
        #print(f"  • Ukupno vreme: {total_time:.3f}s")
        #print(f"  • Preračunavanje matrice: {self.stats['precompute_time']:.3f}s")
        #print(f"  • Vreme po iteraciji: {(total_time / max(self.stats['iterations'], 1)):.4f}s")
        #print(f"\n🔍 DETALJI:")
        #print(f"  • Tau filter eliminacija: {self.stats['tau_filter_hits']:,}")
        #print(f"  • Provera konflikata: {self.stats['conflict_checks']:,}")
        #print(f"  • Ažuriranja min_cost: {self.stats['min_cost_updates']}")
        #print(f"{'=' * 70}\n")

        return self.solution

    def _assign_customer_to_facility(self, customer, facility):
        """Dodeljuje kupca fabrici."""
        assign_amount = min(facility.remaining_capacity, customer.remaining_demand)
        if assign_amount <= 0:
            return

        self.solution.add_assignment(customer.id, facility.id, assign_amount)
        facility.remaining_capacity -= assign_amount
        customer.remaining_demand -= assign_amount

    def solve_greedy(self):
        """Fallback: jednostavni greedy za popunjavanje ostatka."""
        sorted_facilities = list(self.problem.facilities.sort_by_cost_capacity_ratio())

        for fac in sorted_facilities:
            if self.problem.customers.total_remaining_demand() == 0:
                break

            fac.open()
            while getattr(fac, "remaining_capacity", 0) > 0:
                best_customers = self.heuristics.rcl(fac, 3)
                if not best_customers:
                    break

                chosen_cust = self.rng.choice(best_customers)
                self._assign_customer_to_facility(chosen_cust, fac)

                try:
                    self.heuristics.customer_rcl.update_after_assignment(fac, chosen_cust)
                    if getattr(chosen_cust, "remaining_demand", 0) == 0:
                        self.heuristics.customer_rcl.remove_customer(chosen_cust.id)
                except Exception:
                    pass

        #print(f"Fallback greedy: {'✅ Valid' if self.solution.is_valid() else '❌ Invalid'}")

    def has_conflict(self, cust_id, already_assigned_ids):
        """
        Provera konflikta po ID-jevima (za kompatibilnost sa customer_rcl.py).

        Args:
            cust_id: ID kupca
            already_assigned_ids: Lista/set ID-jeva već dodeljenih kupaca

        Returns:
            True ako postoji konflikt, False inače
        """
        if not hasattr(self.problem, 'incompatibilities') or not self.problem.incompatibilities:
            return False

        # Ako su ID-jevi direktno u incompatibilities (tuple set)
        if isinstance(self.problem.incompatibilities, set):
            return any((cust_id, other) in self.problem.incompatibilities or
                       (other, cust_id) in self.problem.incompatibilities
                       for other in already_assigned_ids)

        # Ako imamo incompat_array, konvertuj ID → indeks
        if self.incompat_array is not None and cust_id in self.cust_index:
            cust_idx = self.cust_index[cust_id]
            for other_id in already_assigned_ids:
                if other_id in self.cust_index:
                    other_idx = self.cust_index[other_id]
                    if self.incompat_array[cust_idx, other_idx] == 1:
                        return True
            return False

        return False

    def has_conflict_fast(self, cust_idx, assigned_indices):
        """
        Brza provera konflikta korišćenjem indeksa (internal use).

        Args:
            cust_idx: Indeks kupca
            assigned_indices: numpy array indeksa već dodeljenih kupaca

        Returns:
            True ako postoji konflikt
        """
        if self.incompat_array is None:
            return False
        return has_conflict_by_ids(cust_idx, assigned_indices, self.incompat_array)
