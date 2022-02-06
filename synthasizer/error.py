"""Computing reconstruction error."""
from abc import ABC, abstractmethod
import re
from typing import Any, Dict
from collections import Counter
from .table import Table


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


class ThresholdedReconstructionError(ReconstructionError):
    """Compute reconstruction error.

    Reconstructions errors above a threshold are scaled
    to being completely disallowed.

    """

    def __init__(self, threshold: float = 0.02) -> None:
        """"""
        super().__init__()
        self._threshold = threshold

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
        cells = Counter(cell.base for cell in table.cells)
        score = 0
        for cell, count in (self._cells - cells).items():
            # criteria for cells that are free to remove
            if (
                cell.value is None
                or self._style[_freeze(cell.style)] == 1
                or cell.base in cells
            ):
                continue
            # else,
            score += count
        score = score / self._total
        if score > self._threshold:
            return 1.0
        return score


def _freeze(d: Dict[str, Any]) -> str:
    """Freeze dictionary."""
    return str(sorted(d.items()))


def _shape(s: str) -> str:
    """Get shape of string."""
    return re.sub(r"[A-Z]+", "A", re.sub(r"[a-z]+", "a", re.sub(r"[0-9]+", "0", s)))
