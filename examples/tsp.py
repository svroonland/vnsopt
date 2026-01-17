import random
from typing import Tuple, Generator, Any
from vnsopt import vnd, variable_neighbourhood_search, FitnessScore

# Define a simple TSP instance
nr_cities = 30
cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(nr_cities)]


def distance(city1, city2):
    """Calculate Euclidean distance between two cities"""
    return ((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2) ** 0.5


def tour_length(tour):
    """Calculate total length of a tour"""
    length = sum(distance(cities[tour[i]], cities[tour[i + 1]]) for i in range(len(tour) - 1))
    length += distance(cities[tour[-1]], cities[tour[0]])  # Return to start
    return length


# Define neighbourhoods


def swap_neighbourhood(tour):
    """Swap two cities in the tour"""
    for i in range(len(tour)):
        for j in range(i + 1, len(tour)):
            new_tour = tour.copy()
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            yield new_tour, tour_length(new_tour) - tour_length(tour)


def reverse_segment_neighbourhood(tour):
    """Reverse a segment of the tour (2-opt move)"""
    for i in range(len(tour)):
        for j in range(i + 2, len(tour)):
            new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
            yield new_tour, tour_length(new_tour) - tour_length(tour)


def insert_neighbourhood(tour) -> Generator[Tuple[list, FitnessScore], Any, Any]:
    """Remove a city and insert it elsewhere"""
    for i in range(len(tour)):
        for j in range(len(tour)):
            if i != j:
                new_tour = tour.copy()
                city = new_tour.pop(i)
                new_tour.insert(j, city)
                yield new_tour, tour_length(new_tour) - tour_length(tour)


# Define neighbourhood structures in order of increasing size
neighbourhoods = [swap_neighbourhood, reverse_segment_neighbourhood, insert_neighbourhood]

# Initial solution: random tour
initial_solution = list(range(len(cities)))
random.shuffle(initial_solution)

best_tour, fitness = variable_neighbourhood_search(
    start=initial_solution,
    objective=tour_length,
    neighbourhoods=neighbourhoods,
    local_search=lambda s: vnd(neighbourhoods, s, tour_length, verbosity=1),
    verbose=True,
)

print(f"Best tour: {best_tour}")
print(f"VNS tour length: {tour_length(best_tour):.2f}")
