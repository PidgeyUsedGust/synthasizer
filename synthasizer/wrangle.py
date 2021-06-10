from abc import ABC, abstractmethod
from heapq import heappop, heappush
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Optional, Iterable, Tuple, Type
from .error import ReconstructionError
from .heuristics import Heuristic
from .table import Table
from .transformations import Transformation
from .table import Table


class Program:
    """Transformation program."""

    def __init__(self, transformations: Iterable[Transformation] = []) -> None:
        self.transformations: Tuple[Transformation] = tuple(transformations)

    def __call__(self, table: Table) -> Table:
        """Apply to a table."""
        for transformation in self.transformations:
            table = transformation(table)

    def __hash__(self) -> int:
        return hash(self.transformations)

    def __str__(self) -> str:
        return "\n".join(map(str, self.transformations))

    def extend(self, transformation: Transformation) -> "Program":
        """Extend program with a new transformation."""
        return Program((*self.transformations, transformation))


class Language:
    """Transformation language."""

    def __init__(
        self, transformations: Optional[List[Type[Transformation]]] = None
    ) -> None:
        self._transformations = transformations or list()

    def candidates(self, table: Table) -> List[Transformation]:
        """Get transformations that can be applied."""
        candidates = list()
        for transformation in self._transformations:
            arguments = transformation.arguments(table)
            for argument in arguments:
                candidates.append(transformation(*argument))
        return candidates


@dataclass
class State:
    """State during search."""

    score: float
    program: Program
    table: Table

    @property
    def kind(self) -> str:
        """Get name of the last applied transformation."""
        return self.program.transformations[-1].__class__.__name__

    def __lt__(self, other: "State") -> bool:
        """Score meant to be maximised."""
        return self.score > other.score


class Strategy(ABC):
    """Base search strategy."""

    def __init__(self) -> None:
        self._queue: List[State] = list()
        self.reset()

    def reset(self) -> None:
        self._queue.clear()

    def pop(self) -> State:
        """Get next state to explore."""
        return self._queue.pop()

    @abstractmethod
    def push(self, candidates: List[State]) -> None:
        """Decide which transformations should be explored.

        Args:
            candidates: Ranked list of candidates from bad
                to good such that candidates.pop() yields
                the best candidate.

        Returns:
            A subsequence of candidates to be added to the
            queue.

        """
        pass


class BFS(Strategy):
    """Breadth first search."""

    def push(self, candidates: List[State]) -> None:
        self._queue = candidates + self._queue


class DFS(Strategy):
    """Depth first search."""

    def push(self, candidates: List[State]) -> None:
        self._queue.extend(candidates)


class Beam(Strategy):
    """Beam search."""

    def __init__(self, width: int = 2) -> None:
        super().__init__()
        self._width = width

    def push(self, candidates: List[State]) -> None:
        self._queue = candidates[-self._width :] + self._queue


class VariedBeam(Beam):
    """Beam search that considers different transformations."""

    def __init__(self, width: int = 2, kinds: int = 2) -> None:
        """

        Args:
            width: Number of each unique transformation to keep.
            kinds: Number of unique types of transformations to keep.

        """
        super().__init__()
        self._width = width
        self._kinds = kinds

    def push(self, candidates: List[State]) -> None:
        unique = list()
        seen = defaultdict(int)
        for candidate in candidates[::-1]:
            # check if the last applied transformation
            # is not yet seen too often
            if seen[candidate.kind] < self._kinds:
                unique.append(candidate)
            # update number of times it was seen
            seen[candidate.kind] += 1
            # seen enough unique ones
            if len(unique) == self._width:
                break
        # update queue
        self._queue = unique + self._queue


class Greedy(Strategy):
    """Greedy search."""

    def pop(self) -> State:
        return heappop(self._queue)

    def push(self, candidates: List[State]) -> None:
        for candidate in candidates:
            heappush(self._queue, candidate)


class VariedGreedy(Greedy):
    """Greedy, but only keep a fixed number per kind."""

    def __init__(self, kinds: int = 2) -> None:
        super().__init__()
        self._kinds = kinds

    def push(self, candidates: List[State]) -> None:
        return super().push(candidates)


class Wrangler:
    """Main wrangler class."""

    def __init__(
        self, language: Language, heuristic: Heuristic, strategy: Strategy = None
    ) -> None:
        self._language = language
        self._heuristic = heuristic
        self._strategy = strategy or VariedGreedy(3)

    def learn(self, table: Table) -> List[Program]:
        """Learn and rank wrangling programs."""
        # initialise error
        error = ReconstructionError(table)
        # reset the strategy
        self._strategy.reset()
        self._strategy.push([State(0.0, Program(), table)])
        while True:
            state = self._strategy.pop()
            candidates = self._language.candidates(state.table)
            # evaluate all tables
            states = list()
            for candidate in candidates:
                # print(candidate)
                new_table = candidate(state.table)
                new_program = state.program.extend(candidate)
                new_score = self._heuristic(new_table) - error(new_table)
                states.append(State(new_score, new_program, new_table))
                print(new_score, new_program)
            # update the strategy with ranked new transformations
            self._strategy.push(sorted(states, reverse=True))
            # print(candidates)
            break

    def wrangle(self, table: Table) -> Table:
        """Learn wrangling programs and apply the best one."""
        pass
