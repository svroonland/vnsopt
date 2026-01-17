from typing import Callable, Optional

from vnsopt.types import Neighbourhood, FitnessScore, FitnessFunction


def variable_neighbourhood_search[S](
    start: S,
    neighbourhoods: list[Neighbourhood[S]],
    local_search: Callable[[S], tuple[S, FitnessScore]],
    objective: FitnessFunction[S],
    iterations_vns: int = 200,
    max_trials_per_neighbourhood: int = 5,
    verbose: bool = False,
    on_iteration_complete: Callable[[S, FitnessScore], None] = lambda sol, fitness: None,
    convergence_threshold: Optional[int] = None,
    max_iterations_without_improvement: Optional[int] = None,
) -> tuple[S, FitnessScore]:
    """
    An enhancement on top of a hill climbing (local search) algorithm that helps getting out of local optima.

    Based on the idea that a local optimum is with respect to some definition of locality / neighbourhood and that
    a local optimum in one neighbourhood is not necessarily a local optimum in another neighbourhood.

    When hill climbing gets stuck in a local optimum, the solution is perturbed by randomly selecting a 'neighbouring'
    solution from a neighbourhood function. When after hill climbing again that fails to give better solutions, the VNS
    moves on to the next neighbourhood. When it does give a better solution, the first neighbourhood is chosen again.

    The neighbourhood functions should be increasing order of 'disturbance', as to first explore the more local neighbours of a solution and only then make more drastic changes.

    :param start: Initial solution
    :param neighbourhoods: List of functions that produce a random neighbour solution given a current solution
    :param local_search: Function that iteratively finds a better solution given some starting solution. This function is passed a starting solution and should return the improved solution and fitness score
    :param objective: Fitness function, returns a score for solution. Iteration is towards a MINIMUM of this score.
    :param iterations_vns: Maximum number of VNS iterations
    :param max_trials_per_neighbourhood: Maximum number of perturbations using some neighbourhood followed by hill climbing before switching to another neighbourhood when no better solution was found
    :param verbose: Log progress
    :param on_iteration_complete: Callback for intermediate results, taking the current best solution and a fitness score
    :param convergence_threshold: Stop when fitness score is above this threshold
    :return: Tuple of the improved solution and its fitness score
    """
    k = 0

    best_fitness = objective(start)
    best = start

    print(f"[VNS] Initial fitness: {best_fitness}")

    trials = max_trials_per_neighbourhood
    i = 1

    iterations_no_improvement = 0

    while i <= iterations_vns and k < len(neighbourhoods):
        print("[VNS] Best score:", best_fitness)

        shaken, delta = next(neighbourhoods[k](best), (None, None))
        if shaken is None:
            print(f"[VNS] Neighbourhood k={k} returned no neighbours, moving to next neighbourhood")
            k += 1
            trials = max_trials_per_neighbourhood
            continue

        improved, new_fitness = local_search(shaken)
        if new_fitness < best_fitness:
            iterations_no_improvement = 0
            (
                print(f"[VNS] Iteration {i} - Found improvement from {best_fitness} to {new_fitness}")
                if verbose
                else None
            )
            best_fitness = new_fitness
            best = improved
            k = 0  # Move back to the first neighbourhood
            trials = max_trials_per_neighbourhood
        else:
            iterations_no_improvement += 1
            if trials == 0:
                k += 1  # Switch to next neighbourhood
                trials = max_trials_per_neighbourhood
                (
                    print(f"[VNS] Iteration {i} - No significant improvement, switching to neighbourhood k={k}")
                    if verbose
                    else None
                )
            else:
                (
                    print(f"[VNS] Iteration {i} - No significant improvement, shaking again (neighbourhood k={k})")
                    if verbose
                    else None
                )
                trials -= 1
        on_iteration_complete(best, best_fitness)
        if convergence_threshold is not None and best_fitness <= convergence_threshold:
            print("[VNS] Convergence threshold reached") if verbose else None
            break
        if max_iterations_without_improvement is not None:
            if iterations_no_improvement >= max_iterations_without_improvement:
                (
                    print(f"[VNS] No improvement in the last {iterations_no_improvement} iterations, stopping.")
                    if verbose
                    else None
                )
                break
        i += 1

    print(f"[VNS] Done. Score: {best_fitness}") if verbose else None
    return best, best_fitness
