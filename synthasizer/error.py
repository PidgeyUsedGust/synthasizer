"""Computing reconstruction error."""
from abc import ABC, abstractmethod
import re
from typing import Any, Dict, List, Tuple
from collections import Counter
from .table import Cell, Table


class ReconstructionError(ABC):
    """Base reconstruction error."""

    def __call__(self, table: Table) -> float:
        return self.compute(table)

    @abstractmethod
    def initialise(self, table: Table) -> None:
        self._table = table

    @abstractmethod
    def compute(self, table: Table) -> float:
        pass


class ContentReconstructionError(ReconstructionError):
    """Only use cell content."""

    def initialise(self, table: Table) -> None:
        super().initialise(table)
        self._cells = set(table.cells)

    def compute(self, table: Table) -> float:
        # if a color was removed, punish really hard
        if table.n_colors < self._table.n_colors:
            return 1.0
        # else count the number of cells that was removed
        # return len(self._cells - set(table.cells)) / len(self._cells)
        return max(0, 0.05 * len(self._cells - set(table.cells)))


class MixedReconstructionError(ReconstructionError):
    """Compute reconstruction error."""

    def initialise(self, table: Table):
        super().initialise(table)
        self._cells = Counter(table.cells)
        self._style = Counter()
        self._shape = Counter()
        self._total = sum(self._cells.values())
        # count styles
        for cell, count in self._cells.items():
            self._style[_freeze(cell.style)] += count
        # count shapes
        for cell, count in self._cells.items():
            self._shape[_shape(str(cell))] += count

    def compute(self, table: Table) -> float:
        """Compute score."""
        # if a color was removed, punish really hard
        if table.n_colors < self._table.n_colors:
            return 1.0
        # else count value of removed cells
        removed = self._cells - Counter(table.cells)
        score = sum(self.score(cell) * count for cell, count in removed.items())
        return score / self._total

    def score(self, cell: Cell) -> float:
        """Compute score of a cell."""
        if cell.value is None:
            return 0.0
        score_style = self._style[_freeze(cell.style)] / self._total
        score_shape = self._shape[_shape(str(cell))] / self._total
        return (score_style + score_shape) / 2.0


def _freeze(d: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """Freeze dictionary."""
    return tuple(sorted(d.items()))


def _shape(s: str) -> str:
    """Get shape of string."""
    return re.sub(r"[A-Z]+", "A", re.sub(r"[a-z]+", "a", re.sub(r"[0-9]+", "0", s)))
