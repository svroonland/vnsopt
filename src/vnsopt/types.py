from typing import Any, Callable, Generator, TypeVar, Tuple

S = TypeVar("S")
type FitnessScore = float
type FitnessFunction[S] = Callable[[S], FitnessScore]
type Neighbourhood[S] = Callable[[S], Generator[Tuple[S, FitnessScore], Any, Any]]  # The delta fitness score
