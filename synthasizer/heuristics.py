"""Heuristics.

Heuristics compute a score on the estimated quality
of a table, according to some criterium.

Aside from heuristics on the whole table, they can also be
defined on a single column. A wrapper heuristic computes
their value for the whole table, but it allows the wrangling
algorithm to focus on specfic columns.

All heuristics should return a score in [0, 1] to make
it easy to combine them.

"""
import numpy as np
from typing import Counter, List, Optional
from itertools import combinations
from abc import ABC, abstractmethod
from .similarity import CellSimilarity, CompressedSimilarity
from .table import Table, Cell
from .utilities import nzs, transpose


class Heuristic(ABC):
    """Base heuristic."""

    @abstractmethod
    def __call__(self, table: Table) -> float:
        pass


class ColumnHeuristic(ABC):
    """Compute heuristic for each column."""

    @abstractmethod
    def __call__(self, table: Table) -> np.ndarray:
        pass

    def __repr__(self) -> str:
        return "{}()".format(self.__class__.__name__)


class WeightedHeuristic(Heuristic):
    """Weighted combination of other heuristics."""

    def __init__(
        self, heuristics: List[Heuristic], weights: Optional[List[float]] = None
    ) -> None:
        self._heuristics = heuristics
        # ensure that the weights sum to 1
        if weights is None:
            weights = np.ones(len(heuristics))
        self._weights = np.array(weights) / sum(weights)

    def __call__(self, table: Table) -> float:
        scores = list()
        for i, heuristic in enumerate(self._heuristics):
            scores.append(self._weights[i] * heuristic(table))
        return len(self._heuristics) * np.mean(scores)


class ColorRowHeuristic(Heuristic):
    """Check if row with colors exists."""

    def __call__(self, table: Table) -> float:
        colors = table.color_df
        # get colors in each row
        rows = np.apply_along_axis(nzs, 1, colors.values).tolist()
        rows.extend(map(nzs, transpose(colors.columns)))
        # get all unique colors
        unique = set.union(*rows)
        # no colors
        if len(unique) == 0:
            return 1.0
        # get score
        return max(map(len, rows)) / len(unique)


class EmptyHeuristic(Heuristic):
    """Consider global empty cells, rather than column based."""

    def __call__(self, table: Table) -> float:
        return (table.df != Cell(None)).values.sum() / table.df.size


class AggregatedHeuristic(Heuristic):
    """Turn a ColumnHeuristic into a Heuristic."""

    def __init__(self, heuristic: ColumnHeuristic) -> None:
        super().__init__()
        self._heuristic = heuristic

    def __call__(self, table: Table) -> float:
        return np.mean(self._heuristic(table))

    def __repr__(self) -> str:
        return "AggregatedHeuristic({})".format(repr(self._heuristic))


class ColorColumnHeuristic(ColumnHeuristic):
    """Use colors to compute heuristic.

    Assume each target column to be marked with
    a different color.

    """

    def __call__(self, table: Table) -> float:
        cdf = table.color_df
        columns = [set(cdf.iloc[:, i]) for i in range(table.width)]
        # add colors of header cells
        if table.header:
            for i in range(table.width):
                columns[i].add(cdf.columns[i])
        # remove no colors
        columns = [c - {0} for c in columns]
        # compute score
        return [1.0 / max(1, len(n)) for n in columns]


class EmptyColumnHeuristic(ColumnHeuristic):
    """Use empty values to compute heuristic.

    Tolerance parameter ensures that missing a few values
    doesn't result in superfluous fill operations acting
    like missing data imputers.

    """

    def __init__(self, tolerance: float = 0.99) -> None:
        self._tolerance = tolerance

    def __call__(self, table: Table) -> np.ndarray:
        scores = list()
        for i in range(table.width):
            v = table[i].map(bool).sum() / float(table.height)
            if v > self._tolerance:
                scores.append(1)
            else:
                scores.append(v)
        return scores


class ValueColumnHeuristic(ColumnHeuristic):
    """Use cell similarity for strings.

    Will skip columns with a clean type, such
    as numbers and dates.

    """

    def __init__(self, similarity: Optional[CellSimilarity] = None):
        self._similarity = similarity or CompressedSimilarity()

    def __call__(self, table: Table) -> np.ndarray:
        scores = np.ones(table.width)
        dtypes = table.column_types
        for i in range(table.width):
            # string type, compute average similarity between cells
            if dtypes[i] == "string":
                values = set(cell.value for cell in table[i] if cell)
                if len(values) > 1:
                    simils = [
                        self._similarity(a, b) for a, b in combinations(values, 2)
                    ]
                    scores[i] = np.mean(simils)
                else:
                    scores[i] = 1.0
            # mixed columns
            elif "mixed" in dtypes[i]:
                count = table[i].map(bool).sum()
                if count == 0:
                    scores[i] = 0
                else:
                    types = Counter(cell.dtype for cell in table[i] if cell)
                    # _, n = types.most_common(1)[0]
                    scores[i] = types.most_common(1)[0][1] / count
            # uniform number columns get a perfect score
            else:
                scores[i] = 1.0
        return scores


class TypeColumnHeuristic(ColumnHeuristic):
    """Heuristic based on uniform types.

    Computers uniformity of types for all mixed columns,
    and gives 1 in other columns.

    """

    def __call__(self, table: Table) -> np.ndarray:
        ctypes = table.cell_types.T
        dtypes = table.column_types
        scores = np.zeros(table.width)
        for i in range(table.width):
            if "mixed" in dtypes[i]:
                non_empty = ctypes[i] != "empty"
                _, counts = np.unique(ctypes[i][non_empty], return_counts=True)
                scores[i] = np.max(counts) / np.count_nonzero(non_empty)
            else:
                scores[i] = 1.0
        return scores


# def quality(table: pd.DataFrame, distance: Similarity):
#     """Table quality without colors.

#     Args:

#     Returns:
#         pass

#     """
#     # loop over columns
#     score = 0
#     for c in range(table.n):
#         column = table[c, :]
#         column_score = 0
#         n_distances = 0
#         for (e1, e2) in itertools.combinations(column, 2):
#             if e1.value and e2.value:
#                 column_score += distance(e1.value, e2.value)
#                 n_distances += 1
#         # add empty value normalised column uniformity to score.
#         score += (column_score / max(1, n_distances)) * (
#             0.01 + (column.count("") / len(column))
#         )
#     # average for table
#     return score / table.n


# def quality_colors(table, distance, noise=1):
#     """Table quality with respect to a coloring.

#     Arguments:
#         noise {float} -- Noise tolerance.

#     """
#     # compute noise tolerance
#     # noise = noise * table.m
#     # print(noise)
#     # get best assignments
#     assignments = table.assignments()
#     if not assignments:
#         return 0
#     # list of scores for each assignment
#     scores = list()
#     # go over assignments
#     for assignment in assignments:
#         # initialise score
#         us = list()
#         es = list()
#         for _, (value, x) in enumerate(assignment):
#             # get assigned column for the value
#             column = table[x, :]
#             # compute score
#             u = 0
#             for element in column:
#                 if element.value:
#                     try:
#                         u += distance(element.value, value.value)
#                     except TypeError:
#                         print(element.value, value.value)
#             us.append(u / max(1, len(column) - column.count(Cell(""))))
#             e = float(column.count(Cell(""))) / len(column)
#             if e < (1 - noise):
#                 es.append(0)
#             else:
#                 es.append(e)
#             # print(u / max(1, len(column) - column.count(Cell(''))), e)
#         # score = sum(map(lambda u, e: u * (1 + e), us, es)) / len(us)
#         # print(us, es)
#         score = sum(map(lambda u, e: (u + e), us, es)) / max(1, len(us))
#         scores.append((score, us, es))
#     # select best assignment
#     i = scores.index(min(scores))
#     # assign to table
#     table.column_assignment = assignments[i]
#     table.scores_u = scores[i][1]
#     table.scores_e = scores[i][2]
#     # return score
#     return scores[i][0]  # * max(0.01, (1 - table.assignment_score))


# def quality_mixed(table, distance, noise=1):
#     """Mixed quality.

#     Use assignment for those columns.
#     """
#     # get score for colored columns
#     score_color = quality_colors(table, distance, noise=noise)
#     # get colored columns
#     color_colums = {x for (_, x) in table.assignment}
#     # if no other columns, return
#     if len(color_colums) == table.n:
#         return score_color
#     # get score for other columns
#     score_nocolor = 0
#     for c in range(table.n):
#         if c not in color_colums:
#             column = table[c, :]
#             column_score = 0
#             n_distances = 0
#             for (e1, e2) in itertools.combinations(column, 2):
#                 if e1.value and e2.value:
#                     column_score += distance(e1.value, e2.value)
#                     n_distances += 1
#             # add empty value normalised column uniformity to score.
#             e = float(column.count("")) / len(column)
#             if e < (1 - noise):
#                 e = 0
#             # score_nocolor += ((column_score / max(1, n_distances)) *
#             #                   (1 + (column.count('') / len(column))))
#             score_nocolor += (column_score / max(1, n_distances)) + e
#     # average for table
#     return score_color + score_nocolor
