from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Type, Dict
from synthasizer.transformation import Transformation, Program
from synthasizer.table import Table


Pattern = List[Type[Transformation]]
"""Pattern of transformations."""


@dataclass
class State:
    """State during search."""

    score: float
    program: Program
    table: Table

    @property
    def kind(self) -> str:
        """Get name of the last applied transformation."""
        if len(self.program.transformations) == 0:
            return ""
        return self.program.transformations[-1].__class__.__name__

    def __lt__(self, other: "State") -> bool:
        """Score meant to be maximised."""
        return self.score > other.score


class Strategy(ABC):
    """Base search strategy."""

    def __init__(self) -> None:
        self._queue: List[State] = list()
        self.reset()

    def __len__(self) -> int:
        return len(self._queue)

    def empty(self) -> bool:
        return len(self._queue) == 0

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
            if seen[candidate.kind] < self._width:
                unique.append(candidate)
            # update number of times it was seen
            seen[candidate.kind] += 1
            # seen enough unique ones
            if len(unique) >= self._width * self._kinds:
                break
        # update queue
        self._queue = unique + self._queue


class Greedy(Strategy):
    """Greedy search."""

    def push(self, candidates: List[State]) -> None:
        self._queue.extend(candidates)
        self._queue.sort()


class VariedGreedy(Greedy):
    """Greedy, but only keep a fixed number of each kind."""

    def __init__(self, kinds: int = 1) -> None:
        super().__init__()
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
        # update queue
        super().push(unique)


class IterativeDeepeningBeam(Beam):
    """Iterative deepening beam.

    Combines beam search with iterative deepening. We'll
    most likely not require a complete search space anyway.

    """

    def __init__(self, width: int, start: int = 4, delta: int = 2) -> None:
        """

        Args:
            width: Width of the beam.
            start: Starting maximal depth. End is controlled
                by the wrangler and not by the strategy.
            delta: Number of levels to increase after previous
                depth is completely explored.

        """
        super().__init__(width=width)
        self._level = start
        self._delta = delta
        self._wait = list()

    def push(self, candidates: List[State]) -> None:
        # exceeds current length, add to wait list
        if len(candidates[0].program) > self._level:
            self._wait.extend(candidates)


class Junction(Strategy):
    """Emphasize on making varied initial decisions.

    Performs depth first search until max depth is
    reached (controlled by the wrangler) and then
    backtracks to the beginning, rather than to
    the previous level.

    """

    def __init__(self, width: int = 3) -> None:
        super().__init__()
        self._width = width

    def push(self, candidates: List[State]) -> None:
        # add best one to the end of the queue
        # to ensure depth first part
        self._queue.append(candidates[-1])
        # adds the rest to the front of the queue,
        # to backtrack once depth has been reached
        self._queue = candidates[:-1] + self._queue


class Astar(Strategy):
    """Use A*.

    The length of the program is the current
    cost, and the score is the heuristic.

    """

    def __init__(self, length: int = 5) -> None:
        super().__init__()
        self._length = float(length)
        self._scores: Dict[Program, float] = dict()

    def push(self, candidates: List[State]) -> None:
        # update scores
        for candidate in candidates:
            self._scores[candidate.program] = self.rate(candidate)
        # add to the queue and reorder it
        self._queue.extend(candidates)
        self._queue.sort(key=lambda s: self._scores[s.program])

    def rate(self, state: State) -> float:
        """Rate a state."""
        return state.score - (len(state.program) / 40)


class Prioritizer(Strategy):
    """Manually prioritize specific transformations."""

    def __init__(self, base: Strategy, patterns: List[Pattern]) -> None:
        super().__init__()
        self._base = base
        self._patterns = list()

    def push(self, candidates: List[State]) -> None:
        # update the base
        self._base.push(candidates)
        # check patterns
        # reorder based on patterns
