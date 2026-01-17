from typing import List

from vnsopt.types import Neighbourhood, FitnessFunction, FitnessScore


def vnd[S](neighbourhoods: List[Neighbourhood[S]], s: S, objective: FitnessFunction[S]) -> tuple[S, FitnessScore]:
    """
    Variable Neighbourhood Descent (VND) algorithm.

    :param neighbourhoods:
    :param s:
    :param objective:
    :return:
    """
    k = 0
    best_s = s
    best_score = objective(best_s)
    # print(f"[VND] Initial score: {best_score}")

    while k < len(neighbourhoods):
        # print(f"[VND] k={k}")
        neighbourhood = neighbourhoods[k]

        while True:
            improved = False
            gen = neighbourhood(best_s)

            for new_s, delta in gen:
                # print(f"[VND] k={k}: Evaluated neighbour with delta {delta}")
                if delta < -0.0000001:
                    best_s = new_s
                    best_score = best_score + delta
                    # print(f"[VND] k={k}: improved solution to {best_score}")
                    improved = True
                    break
            if not improved:
                # print(f"[VND] k={k}: No improvement found in neighbourhood")
                break
        k += 1
    # print(f"[VND] Done. Best score: {best_score}")
    return best_s, best_score