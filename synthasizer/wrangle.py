from heapq import merge
from itertools import islice, takewhile
from tabnanny import verbose
from time import time
from operator import attrgetter
from typing import List, Optional
from .error import ReconstructionError, ThresholdedReconstructionError
from .heuristics import Heuristic
from .table import Table
from .transformation import *
from .table import Table
from .strategy import *


Phase = Tuple[Language, Heuristic]


class Wrangler:
    """Main wrangler class."""

    def __init__(
        self,
        phases: List[Phase],
        error: Optional[ReconstructionError] = None,
        strategy: Optional[Strategy] = None,
        max_depth: int = 5,
        max_pass: int = 0,
        max_time: float = 1,
        verbose: bool = False,
    ) -> None:
        """

        Args:
            max_depth: Maximal number of transformations.
            max_pass:
            max_time: Maximal number of seconds.

        """
        self._phases = phases
        self._error = error or ThresholdedReconstructionError()
        self._strategy = strategy or Astar()
        self._max_depth = max_depth
        self._max_pass = max_pass
        self._max_time = max_time
        self._verbose = verbose

    def learn(self, table: Table, k: int = 3) -> List[Program]:
        """Learn and rank wrangling programs.

        Args:
            k: Number of programs to pass between phases.

        Returns:
            A ranked list of programs.

        """

        # prepare the error, which is shared
        # across all phases
        self._error.initialise(table)

        # initialise with initial table
        current = [State(0.0, 0.0, Program(), table)]
        seen = set()
        result = list()

        for language, heuristic in self._phases:
            if self._verbose:
                print("> Next Phase")
            # prepare the strategy
            self._strategy.reset()
            self._strategy.push(
                [State(heuristic(s.table), 0.0, s.program, s.table) for s in current]
            )
            # keep result for this phase
            # result = list()
            start = time()
            while not self._strategy.empty():
                state = self._strategy.pop()
                if self._verbose:
                    print(" >", repr(state.program), state.score)
                # expand state
                states = list()
                for candidate in language.candidates(state.table):
                    new_table = candidate(state.table)
                    new_table_hash = hash(new_table)
                    if new_table_hash not in seen:
                        new_program = state.program.extend(candidate)
                        new_score = heuristic(new_table)
                        new_error = self._error(new_table)
                        if self._verbose:
                            print("  >", candidate, new_score, new_error)
                        if new_error < 1:
                            states.append(
                                State(new_score, new_error, new_program, new_table)
                            )
                        seen.add(new_table_hash)
                # evaluate new found states
                if len(states) > 0:
                    # sort by score and merge into result
                    ranked = sorted(states, key=attrgetter("score"), reverse=True)
                    result = merge(
                        ranked, result, key=attrgetter("score"), reverse=True
                    )
                    result = list(islice(result, k))
                    # if found perfect, set the result to perfect only
                    if result[0].score == 1:
                        result = list(takewhile(lambda x: x.score == 1.0, result))
                        break
                    # found better table, expand
                    if len(state.program.transformations) < self._max_depth - 1:
                        self._strategy.push(ranked)
                    # reached timeout
                    if time() - start > self._max_time:
                        break
            # reset current for next iteration
            current = result
        # return with best current
        return current
