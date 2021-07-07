from time import time
from operator import attrgetter
from typing import List, Optional
from multiprocessing import Pool
from .error import ContentReconstructionError, ReconstructionError
from .heuristics import Heuristic
from .table import Table
from .transformation import *
from .table import Table
from .strategy import *


class Wrangler:
    """Main wrangler class."""

    def __init__(
        self,
        language: Language,
        heuristic: Heuristic,
        error: Optional[ReconstructionError] = None,
        strategy: Optional[Strategy] = None,
        max_depth: int = 5,
        max_iterations: int = 0,
        verbose: bool = False,
    ) -> None:
        """

        Args:
            prune_worse: If true, only consider transformations
                that improve a table.

        """
        self._language = language
        self._heuristic = heuristic
        self._error = error or ContentReconstructionError()
        self._strategy = strategy or Astar()
        self._max_depth = max_depth
        self._max_iterations = max_iterations
        self._verbose = verbose

    def learn(self, table: Table) -> List[Program]:
        """Learn and rank wrangling programs."""
        result = list()
        seen = set()
        self._error.initialise(table)
        self._strategy.reset()
        self._strategy.push([State(0.0, Program(), table)])
        iterations = 0
        while not self._strategy.empty():
            state = self._strategy.pop()
            if self._verbose:
                print(repr(state.program), state.score)
            transformations = self._language.candidates(state.table)
            states = list()
            for candidate in transformations:
                new_table = candidate(state.table)
                new_table_hash = hash(new_table)
                if new_table_hash not in seen:
                    new_program = state.program.extend(candidate)
                    new_score = self.rate(new_table)
                    if self._verbose:
                        print(repr(new_program), "->", new_score)
                    states.append(State(new_score, new_program, new_table))
                    seen.add(new_table_hash)
            # get best one
            best = max(states, key=attrgetter("score"), default=state)
            # perfect score, can stop
            if best.score >= 0.9999:
                return [best]
            # found better table, expand
            elif len(state.program.transformations) < self._max_depth:
                self._strategy.push(sorted(states, reverse=True))
            else:
                result.append(state)
            iterations += 1
            if self._max_iterations and iterations >= self._max_iterations:
                break
            # if iterations == 2:
            #     break
            # break
        return result

    def wrangle(self, table: Table) -> Table:
        """Learn wrangling programs and apply the best one."""
        pass

    def rate(self, table: Table) -> float:
        return self._heuristic(table) - self._error(table)
