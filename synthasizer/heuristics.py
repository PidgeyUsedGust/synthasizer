import math
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from .similarity import *
from .table import Table


class Heuristic(ABC):
    """Base heuristic."""

    @abstractmethod
    def __call__(self, table: Table) -> float:
        pass


class ColorRowHeuristic(Heuristic):
    """Check if row with colors exists."""

    def __call__(self, table: Table) -> float:
        rows = table.color_df.apply(frozenset, axis=1).to_list()
        if table.header:
            rows.append(frozenset(table.color_df.columns.map(lambda c: c.color)))
        print(set.union(*rows))


class ColumnHeuristic(ABC):
    """Compute heuristic for each column."""

    @abstractmethod
    def __call__(self, table: Table) -> np.ndarray:
        pass


class ColorColumnHeuristic(ColumnHeuristic):
    """Use colors to compute heuristic."""

    def __call__(self, table: Table) -> float:
        colors = table.color_df
        column = [set(colors.iloc[:, i].unique()) for i in range(table.width)]
        return [float(len(colors) < 3) for colors in column]


class EmptyColumnHeuristic(ColumnHeuristic):
    """Use empty values to compute heuristic."""

    def __call__(self, table: Table) -> np.ndarray:
        return [table[i].map(bool).sum() / table.height for i in range(table.width)]


class ValueColumnHeuristic(ColumnHeuristic):
    """Use cell similarity."""

    def __init__(self, similarity: Optional[CellSimilarity] = None):
        self._similarity = similarity

    def __call__(self, table: Table) -> np.ndarray:
        pass


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


def quality_mixed(table, distance, noise=1):
    """Mixed quality.

    Use assignment for those columns.
    """
    # get score for colored columns
    score_color = quality_colors(table, distance, noise=noise)
    # get colored columns
    color_colums = {x for (_, x) in table.assignment}
    # if no other columns, return
    if len(color_colums) == table.n:
        return score_color
    # get score for other columns
    score_nocolor = 0
    for c in range(table.n):
        if c not in color_colums:
            column = table[c, :]
            column_score = 0
            n_distances = 0
            for (e1, e2) in itertools.combinations(column, 2):
                if e1.value and e2.value:
                    column_score += distance(e1.value, e2.value)
                    n_distances += 1
            # add empty value normalised column uniformity to score.
            e = float(column.count("")) / len(column)
            if e < (1 - noise):
                e = 0
            # score_nocolor += ((column_score / max(1, n_distances)) *
            #                   (1 + (column.count('') / len(column))))
            score_nocolor += (column_score / max(1, n_distances)) + e
    # average for table
    return score_color + score_nocolor

