from operator import attrgetter
from typing import List, Optional, Type
from .error import ContentReconstructionError, ReconstructionError
from .heuristics import Heuristic
from .table import Table
from .transformation import Program, Language
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
        max_depth: Optional[int] = 5,
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

    def learn(self, table: Table) -> List[Program]:
        """Learn and rank wrangling programs."""
        result = list()
        seen = set()
        # initialise error
        self._error.initialise(table)
        # reset the strategy
        self._strategy.reset()
        self._strategy.push([State(0.0, Program(), table)])
        iterations = 0
        while not self._strategy.empty():
            state = self._strategy.pop()
            print(repr(state.program), state.score)
            candidates = self._language.candidates(state.table)
            # evaluate all tables
            states = list()
            for candidate in candidates:
                new_table = candidate(state.table)
                new_program = state.program.extend(candidate)
                print(repr(new_program), end=" -> ")
                new_score = self._heuristic(new_table) - self._error(new_table)
                print(new_score)
                if hash(new_table) not in seen:
                    states.append(State(new_score, new_program, new_table))
                    seen.add(hash(new_table))
            # get best one
            best = max(states, key=attrgetter("score"), default=state)
            # perfect score, can stop
            if best.score == 1:
                return [best]
            # found better table, expand
            elif len(state.program.transformations) < self._max_depth:
                self._strategy.push(sorted(states, reverse=True))
            else:
                result.append(state)
            iterations += 1
            # print(len(self._strategy._queue))
            # if iterations == 10:
            #     break
            # break
        return result

    def wrangle(self, table: Table) -> Table:
        """Learn wrangling programs and apply the best one."""
        pass
