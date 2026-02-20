from typing import List

from vnsopt.types import Neighbourhood, FitnessFunction, FitnessScore


def vnd[S](
    neighbourhoods: List[Neighbourhood[S]],
    s: S,
    objective: FitnessFunction[S],
    epsilon: float = 1e-6,
    verbosity: int = 0,
    maximize: bool = False,
    iterations: int = 200,
) -> tuple[S, FitnessScore]:
    """
    Variable Neighbourhood Descent (VND) algorithm.

    :param neighbourhoods: List of neighbourhoods to explore sequentially
    :param s: Initial solution
    :param objective: Fitness function, returns a score for solution. Iteration is towards a MINIMUM of this score.
    :param epsilon: Minimum improvement in fitness for a solution to be considered better
    :param verbosity: Log verbosity level. 0 or 1 currently.
    :param maximize: Whether to maximize the objective function instead of minimizing
    :param iterations : Maximum number of iterations per neighbourhood
    :return: Tuple of the improved solution and its fitness score
    """
    k = 0
    best_s = s
    best_score = objective(best_s)
    print(f"[VND] Initial score: {best_score}") if verbosity > 0 else None

    while k < len(neighbourhoods):
        print(f"[VND] k={k}") if verbosity > 0 else None
        neighbourhood = neighbourhoods[k]

        i = 0

        while True:
            improved = False
            gen = neighbourhood(best_s)

            for new_s, delta in gen:
                if i > iterations:
                    print(f"[VND] k={k}: Reached maximum number of iterations") if verbosity > 0 else None
                    improved = False
                    break

                i += 1

                print(f"[VND] k={k}: Evaluated neighbour with delta {delta}") if verbosity > 1 else None
                if (not maximize and delta < -epsilon) or (maximize and delta > epsilon):
                    best_s = new_s
                    best_score = best_score + delta
                    print(f"[VND] k={k}, i={i}: improved solution to {best_score}") if verbosity > 0 else None
                    improved = True
                    break
            if not improved:
                print(f"[VND] k={k}: No improvement found in neighbourhood") if verbosity > 1 else None
                break
        k += 1
    print(f"[VND] Done. Best score: {best_score}") if verbosity > 0 else None
    return best_s, best_score
