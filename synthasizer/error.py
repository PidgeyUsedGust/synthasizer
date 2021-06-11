"""Computing reconstruction error."""
import re
from typing import Any, Dict, List, Tuple
from collections import Counter
from .table import Cell, Table


class ReconstructionError:
    """Compute reconstruction error."""

    def __init__(self, table: Table):
        self._table = table
        self._cells = Counter(table.cells)
        self._style = Counter()
        self._shape = Counter()
        self._total = sum(self._cells.values())
        # count styles
        for cell, count in self._cells.items():
            self._style[freeze(cell.style)] += count
        # count shapes
        for cell, count in self._cells.items():
            self._shape[shape(str(cell))] += count

    def __call__(self, table: Table) -> float:
        return self.compute(table)

    def compute(self, table: Table) -> float:
        """Compute score."""
        # if a color was removed, punish really hard
        if table.n_colors < self._table.n_colors:
            return 1
        # else count value of removed cells
        removed = self._cells - Counter(table.cells)
        score = sum(self.score(cell) * count for cell, count in removed.items())
        return score / self._total

    def score(self, cell: Cell) -> float:
        """Compute score of a cell."""
        if cell.value is None:
            return 0.0
        score_style = self._style[freeze(cell.style)] / self._total
        score_shape = self._shape[shape(str(cell))] / self._total
        return (score_style + score_shape) / 2.0


def freeze(d: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """Freeze dictionary."""
    return tuple(sorted(d.items()))


def shape(s: str) -> str:
    """Get shape of string."""
    return re.sub(r"[A-Z]+", "A", re.sub(r"[a-z]+", "a", re.sub(r"[0-9]+", "0", s)))
