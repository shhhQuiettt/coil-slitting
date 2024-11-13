from pymoo.util.dominator import Dominator
from copy import deepcopy
import numpy as np


def binary_tournament(pop, P, algorithm, **kwargs):
    n_tournaments, n_parents = P.shape

    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    tournament_type = algorithm.tournament_type
    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):

        a, b = P[i, 0], P[i, 1]
        a_cv, a_f, b_cv, b_f = pop[a].CV[0], pop[a].F, pop[b].CV[0], pop[b].F
        rank_a, cd_a = pop[a].get("rank", "crowding")
        rank_b, cd_b = pop[b].get("rank", "crowding")

        # if at least one solution is infeasible
        if a_cv > 0.0 or b_cv > 0.0:
            S[i] = compare(
                a,
                a_cv,
                b,
                b_cv,
                method="smaller_is_better",
                return_random_if_equal=True,
            )

        # both solutions are feasible
        else:

            if tournament_type == "comp_by_dom_and_crowding":
                rel = Dominator.get_relation(a_f, b_f)
                if rel == 1:
                    S[i] = a
                elif rel == -1:
                    S[i] = b

            elif tournament_type == "comp_by_rank_and_crowding":
                S[i] = compare(a, rank_a, b, rank_b, method="smaller_is_better")

            else:
                raise Exception("Unknown tournament type.")

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = compare(
                    a,
                    cd_a,
                    b,
                    cd_b,
                    method="larger_is_better",
                    return_random_if_equal=True,
                )

    return S[:, None].astype(int, copy=False)


def compare(a, a_val, b, b_val, method, return_random_if_equal=False):
    if method == "larger_is_better":
        if a_val > b_val:
            return deepcopy(a)
        elif a_val < b_val:
            return deepcopy(b)
        else:
            if return_random_if_equal:
                return deepcopy(np.random.choice([a, b]))
            else:
                return None
    elif method == "smaller_is_better":
        if a_val < b_val:
            return deepcopy(a)
        elif a_val > b_val:
            return deepcopy(b)
        else:
            if return_random_if_equal:
                return deepcopy(np.random.choice([a, b]))
            else:
                return None
    else:
        raise Exception("Unknown method.")
