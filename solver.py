from typing import Optional, Dict, List, Tuple

import solution
import random
import heuristics
import instance

class Solver:
    def __init__(self, problem_instance):
        self.problem = problem_instance
        self.solution = solution.Solution(problem_instance)
        self.rng = random.Random(53)
        self.heuristics = heuristics.Heuristics(self)

    def solve_grasp(self, number_of_iterations: int = 60):
        # Inicijalno resetovanje
        self.solution.reset()
        self.problem.facilities.reset()
        self.problem.customers.reset()
        self.heuristics.customer_rcl.reset()

        # Prvi greedy + lokalna pretraga kao početno rešenje
        self.solve_greedy()
        self.solve_local_search()

        best_cost = self.solution.total_cost()
        best_solution_snapshot = self.solution.copy()

        for iteration in range(number_of_iterations):
            # Reset pre svake iteracije
            self.solution.reset()
            self.problem.facilities.reset()
            self.problem.customers.reset()
            self.heuristics.customer_rcl.reset()

            # GRASP faze: konstrukcija + lokalno poboljšanje
            self.solve_greedy()

            # Primeni heuristike na nivou postrojenja
            heuristics_list = [
                self.heuristics.close_one_facility,
                self.heuristics.open_one_facility,
                self.heuristics.close_one_open_one,
                self.heuristics.close_one_open_two,
                self.heuristics.open_one_close_two,
            ]

            # Izaberi nasumično jednu heuristiku
            chosen_heuristic = random.choice(heuristics_list)

            # Pokušaj da je primeni
            try:
                chosen_heuristic()
            except Exception as e:
                print(f"Warning: heuristic {chosen_heuristic.__name__} failed with error {e}")

            self.solve_local_search()

            # Šira pretraga (LNS)
            self.heuristics.large_neighborhood_search()

            # Evaluiraj novu konfiguraciju
            current_cost = self.solution.total_cost()

            # Ako je bolje — ažuriraj najbolje rešenje
            if current_cost < best_cost:
                best_cost = current_cost
                best_solution_snapshot = self.solution.copy()

            print(
                f"Iteracija {iteration + 1}/{number_of_iterations} završena, trošak: {current_cost:.2f}, najbolji: {best_cost:.2f}")

        # Postavi najbolje rešenje
        self.solution = best_solution_snapshot
        print(f"GRASP završio. Najbolji trošak: {best_cost:.2f}")

    def has_conflict(self, cust_id, already_assigned_ids):
        return any((cust_id, other) in self.problem.incompatibilities for other in already_assigned_ids)

    def _assign_customer_to_facility(self, customer, facility):
        assign_amount = min(facility.remaining_capacity, customer.remaining_demand)

        self.solution.add_assignment(customer.id, facility.id, assign_amount)

        facility.remaining_capacity -= assign_amount
        customer.remaining_demand -= assign_amount

    def solve_greedy(self):
        # Sort facilities once
        sorted_facilities = self.problem.facilities.sort_by_cost_capacity_ratio()

        for fac in sorted_facilities:
            if self.problem.customers.total_remaining_demand() == 0:
                break

            fac.open()

            while fac.remaining_capacity > 0:
                best_customers = self.heuristics.rcl(fac, 3)

                if not best_customers:
                    break

                chosen_cust = random.choice(best_customers)

                # Assign using existing method
                self._assign_customer_to_facility(chosen_cust, fac)

                # Update RCL
                self.heuristics.customer_rcl.update_after_assignment(fac, chosen_cust)
                if chosen_cust.remaining_demand == 0:
                    self.heuristics.customer_rcl.remove_customer(chosen_cust.id)

        if not self.solution.is_valid():
            print("Greedy solution is invalid!")
        else:
            print("Greedy solution is valid!")

    def solve_local_search(self, max_passes: int = 10, step: float = float("inf")) -> None:
        inst = self.problem
        fac_by_id = {fac.id: fac for fac in inst.facilities}

        def unit_cost(cust_id: int, fac_id: int) -> float:
            return inst.shipping_costs[(cust_id, fac_id)]

        def remaining_capacity(fac_id: int) -> float:
            used = self.solution.facility_used_capacity.get(fac_id, 0)
            return fac_by_id[fac_id].capacity - used

        def customers_assigned_to(fac_id: int):
            return {c_id for (c_id, f_id), amt in self.solution.assignments.items() if f_id == fac_id and amt > 0}

        def incompatible_with_any(cust_id: int, other_customers: set) -> bool:
            if hasattr(inst, "incompatibility_graph"):
                return not inst.incompatibility_graph.get(cust_id, set()).isdisjoint(other_customers)
            else:
                for other in other_customers:
                    if (cust_id, other) in inst.incompatibilities or (other, cust_id) in inst.incompatibilities:
                        return True
                return False

        EPS = 1e-12
        passes = 0
        improved_globally = True

        while improved_globally and passes < max_passes:
            improved_globally = False
            passes += 1

            for A in inst.facilities:
                A_id = A.id
                pairs = [(cust_id, amt) for (cust_id, f_id), amt in self.solution.assignments.items()
                         if f_id == A_id and amt > 0]
                if not pairs:
                    continue

                for cust_id, amt in pairs:
                    cA = unit_cost(cust_id, A_id)
                    best_delta = 0.0
                    best_move = None

                    for B in inst.facilities:
                        B_id = B.id
                        if B_id == A_id:
                            continue

                        capB = remaining_capacity(B_id)
                        if capB <= 0:
                            continue

                        # incompatibility check
                        if incompatible_with_any(cust_id, customers_assigned_to(B_id)):
                            continue

                        cB = unit_cost(cust_id, B_id)
                        if cB >= cA:
                            continue

                        move_amt = min(amt, capB, step)
                        if move_amt <= 0:
                            continue

                        delta_shipping = (cB - cA) * move_amt

                        open_penalty = 0.0
                        if B_id not in self.solution.facilities_open:
                            open_penalty = fac_by_id[B_id].opening_cost

                        used_A = self.solution.facility_used_capacity.get(A_id, 0.0)
                        close_saving = fac_by_id[A_id].opening_cost if abs(used_A - move_amt) <= EPS else 0.0

                        total_delta = delta_shipping + open_penalty - close_saving

                        if total_delta < best_delta:
                            best_delta = total_delta
                            best_move = (B_id, move_amt)

                    if best_move is not None:
                        B_id, move_amt = best_move

                        # directly update solution (guaranteed valid due to checks)
                        self.solution.add_assignment(cust_id, B_id, move_amt)
                        self.solution.assignments[(cust_id, A_id)] -= move_amt
                        if self.solution.assignments[(cust_id, A_id)] <= 0:
                            self.solution.assignments.pop((cust_id, A_id))

                        self.solution.facility_used_capacity[A_id] -= move_amt
                        if self.solution.facility_used_capacity[A_id] <= 0:
                            self.solution.facility_used_capacity.pop(A_id, None)
                            self.solution.facilities_open.discard(A_id)

                        improved_globally = True

        if not self.solution.is_valid():
            print("Local search solution is INVALID!")
        else:
            print("Local search solution is VALID ✅")

    def solve_ga(
            self,
            generations: int = 150,
            pop_size: int = 40,
            tournament_k: int = 3,
            cx_prob: float = 0.9,
            mut_prob: float = 0.08,
            init_top_k: int = 3,
            verbose: bool = True,
    ):
        """
        Genetski algoritam za CFLP:
        - Hromozom: dict {cust_id -> fac_id} (primarna fabrika).
        - Dekodiranje: greedy punjenje izabrane fabrike, pa prelazak na sledeće najjeftinije.
        - Selekcija: turnir.
        - Crossover: jednotačkani.
        - Mutacija: promena fabrike u jednu od k najjeftinijih za tog kupca.
        """
        inst = self.problem
        rng = self.rng

        customer_ids = [c.id for c in inst.customers]
        facility_ids = [f.id for f in inst.facilities]

        cheap_fac_order: Dict[int, List[int]] = {
            c_id: sorted(facility_ids, key=lambda f_id: inst.shipping_costs[(c_id, f_id)])
            for c_id in customer_ids
        }

        def random_individual() -> Dict[int, int]:
            chrom = {}
            for c_id in customer_ids:
                top = cheap_fac_order[c_id][:max(1, min(init_top_k, len(facility_ids)))]
                chrom[c_id] = rng.choice(top)
            return chrom

        def decode_to_solution(chrom: Dict[int, int]) -> Tuple["solution.Solution", float]:
            """
            Pretvara hromozom u Solution:
            - ne dira self.problem.* remaining_* polja,
            - vodi sopstvene local 'remaining' mape,
            - penalizuje eventualno neisporučenu potražnju.
            """
            sol = solution.Solution(inst)
            # lokalni "remaining"
            rem_cap = {f.id: f.capacity for f in inst.facilities}
            rem_dem = {c.id: getattr(c, "demand", getattr(c, "remaining_demand", 0.0)) for c in inst.customers}

            for c_id in customer_ids:
                if rem_dem[c_id] <= 0:
                    continue
                f_id = chrom[c_id]
                take = min(rem_dem[c_id], rem_cap[f_id])
                if take > 0:
                    sol.add_assignment(c_id, f_id, take)
                    rem_dem[c_id] -= take
                    rem_cap[f_id] -= take

            for c_id in customer_ids:
                if rem_dem[c_id] <= 0:
                    continue
                for f_id in cheap_fac_order[c_id]:
                    if rem_dem[c_id] <= 0:
                        break
                    if rem_cap[f_id] <= 0:
                        continue
                    take = min(rem_dem[c_id], rem_cap[f_id])
                    if take > 0:
                        sol.add_assignment(c_id, f_id, take)
                        rem_dem[c_id] -= take
                        rem_cap[f_id] -= take

            base_cost = sol.total_cost()
            unmet = sum(max(0.0, d) for d in rem_dem.values())
            penalty = 1e6 * unmet
            return sol, base_cost + penalty

        def fitness(chrom: Dict[int, int]) -> float:
            _, fit = decode_to_solution(chrom)
            return fit

        def tournament_select(pop: List[Dict[int, int]]) -> Dict[int, int]:
            cand = rng.sample(pop, k=min(tournament_k, len(pop)))
            cand.sort(key=fitness)
            return cand[0]

        def one_point_crossover(p1: Dict[int, int], p2: Dict[int, int]) -> Tuple[Dict[int, int], Dict[int, int]]:
            if rng.random() > cx_prob or len(customer_ids) < 2:
                return p1.copy(), p2.copy()
            cut = rng.randrange(1, len(customer_ids))
            order = customer_ids[:]  # stabilno
            c1, c2 = {}, {}
            for i, c_id in enumerate(order):
                if i < cut:
                    c1[c_id] = p1[c_id]
                    c2[c_id] = p2[c_id]
                else:
                    c1[c_id] = p2[c_id]
                    c2[c_id] = p1[c_id]
            return c1, c2

        def mutate(chrom: Dict[int, int]) -> None:
            for c_id in customer_ids:
                if rng.random() < mut_prob:
                    options = cheap_fac_order[c_id][:max(1, min(init_top_k, len(facility_ids)))]
                    if len(options) <= 1:
                        chrom[c_id] = rng.choice(facility_ids)
                    else:
                        new_f = rng.choice([f for f in options if f != chrom[c_id]] or options)
                        chrom[c_id] = new_f

        population = [random_individual() for _ in range(pop_size)]

        best_chrom = None
        best_fit = float("inf")
        best_sol_snapshot = None

        for ind in population:
            sol_i, fit_i = decode_to_solution(ind)
            if fit_i < best_fit and sol_i.is_valid():
                best_fit = fit_i
                best_chrom = ind.copy()
                best_sol_snapshot = sol_i.copy()

        for gen in range(generations):
            new_pop: List[Dict[int, int]] = []

            # elita (1 komad) — čuvamo trenutno najbolji
            if best_chrom is not None:
                new_pop.append(best_chrom.copy())

            # generiši potomke
            while len(new_pop) < pop_size:
                p1 = tournament_select(population)
                p2 = tournament_select(population)
                c1, c2 = one_point_crossover(p1, p2)
                mutate(c1)
                mutate(c2)
                new_pop.append(c1)
                if len(new_pop) < pop_size:
                    new_pop.append(c2)

            population = new_pop

            # evaluacija + update elite
            for ind in population:
                sol_i, fit_i = decode_to_solution(ind)
                # preferiramo validna rešenja; penal već kažnjava nevalidna
                if fit_i < best_fit and sol_i.is_valid():
                    best_fit = fit_i
                    best_chrom = ind.copy()
                    best_sol_snapshot = sol_i.copy()

            if verbose and (gen % max(1, generations // 10) == 0 or gen == generations - 1):
                print(f"[GA] gen={gen + 1}/{generations} best_cost={best_fit:.6f}")

        # fallback: ako nismo imali validno kroz elite, uzmi trenutno najbolje dekodirano (iako penalizovano)
        if best_sol_snapshot is None and population:
            # nađi najmanji fitness pa postavi
            population.sort(key=fitness)
            bs, bf = decode_to_solution(population[0])
            best_sol_snapshot = bs
            best_fit = bf

        # postavi najbolje rešenje
        self.solution = best_sol_snapshot if best_sol_snapshot is not None else self.solution
        if self.solution.is_valid():
            print("GA solution is valid!")
        else:
            print("GA solution is invalid (penalized).")