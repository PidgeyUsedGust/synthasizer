from abc import ABC, abstractmethod, abstractclassmethod
from copy import copy
from typing import List, Tuple, Any, Union
from collections import Counter
import numpy as np
import pandas as pd
from .conditions import Condition, EmptyCondition
from .table import Table, Cell


Label = Union[int, str, Cell]
"""Column label."""


class Transformation(ABC):
    """Generic transformation."""

    @abstractmethod
    def __call__(self, table: Table) -> Table:
        pass

    @abstractclassmethod
    def arguments(self, table: Table) -> List[Tuple[Any]]:
        pass


class Delete(Transformation):
    """Delete all rows that don't specify a condition in a column"""

    conditions = [EmptyCondition]
    """List of conditions."""

    def __init__(self, column: int, condition: Condition):
        self._column = column
        self._condition = condition

    def __call__(self, table: Table):
        table = copy(table)
        table.df = table.df[~table.df.iloc[:, self._column].map(self._condition)]
        return table

    @classmethod
    def arguments(cls, table: Table) -> List[Tuple[int, Condition]]:
        """Columns for which the condition hold in at least one cell."""
        arguments = list()
        for i in range(table.width):
            column = table.df.iloc[:, i]
            for candidate in cls.conditions:
                for condition in candidate.generate(column.tolist()):
                    arguments.append((i, condition))
        return arguments

    def __str__(self):
        return "Delete({}, {})".format(self._column, self._condition)

    def __repr__(self):
        return "Delete({}, {})".format(repr(self._column), repr(self._condition))

    def __hash__(self):
        return hash(("Delete", self._column, str(self._condition)))


class Divide(Transformation):
    """Divide.

    Split on having different values for a property.

    Divides the column in place.

    """

    def __init__(self, column: int, on: str = "dtype"):
        self._column = column
        self._on = on

    def __call__(self, table: Table) -> Table:
        table = copy(table)
        column = table[self._column]
        values = {getattr(c, self._on) for c in column}
        values.discard("empty")
        masks = [
            column.map(lambda x: getattr(x, self._on) == value) for value in values
        ]
        data = pd.concat([column[mask] for mask in masks], axis=1)
        table.df = pd.concat(
            (table[:, : self._column], data, table[:, self._column + 1 :]), axis=1
        ).fillna(Cell(None))
        return table

    @classmethod
    def arguments(cls, table: Table) -> List[Tuple[int, str]]:
        return list(set(cls.arguments_dtype(table) + cls.arguments_style(table)))

    @classmethod
    def arguments_dtype(cls, table: Table) -> List[Tuple[int, str]]:
        arguments = list()
        for i in range(table.width):
            types = set(c.dtype for c in table[i])
            types.discard("empty")
            if len(types) > 1:
                arguments.append((i, "dtype"))
        return arguments

    @classmethod
    def arguments_style(cls, table: Table) -> List[Tuple[int, str]]:
        arguments = list()
        for i in range(table.width):
            column = table[i]
            for s in column[0].style:
                styles = Counter(c.style[s] for c in column)
                if len(styles) > 1:
                    arguments.append((i, s))
        return arguments

    def __str__(self) -> str:
        return "Divide({}, {})".format(self._column, self._on)

    def __repr__(self) -> str:
        return "Divide({}, {})".format(repr(self._column), repr(self._on))


class Header(Transformation):
    """Set header.

    If multiple rows are set to be the header
    and they can be compressed, that is done.

    For example, setting

        A B C
        A     D

    as header becomes

        A B C D

    in order to support fixing headers after dividing.

    """

    def __init__(self, n: int):
        """

        Args:
            n: Number of lines to take as header.

        """
        assert n > 0
        self._n = n

    def __call__(self, table: Table):
        table = copy(table)
        if self._n > 1:
            # first check if need to compress
            headers = table.df.iloc[: self._n].values
            mask = np.vectorize(bool)(headers)
            # can compress because no column has more
            # than one value
            if (mask.sum(axis=0) < 2).all():
                columns = np.full(mask.shape[1], Cell(None))
                for i, row in enumerate(mask):
                    columns[row] = headers[i][row]
            # else make multi index
            else:
                columns = pd.MultiIndex.from_arrays(table.df.iloc[: self._n].values)
        else:
            columns = table.df.iloc[self._n - 1]
        table = copy(table)
        table.df.columns = columns
        table.df = table.df.iloc[self._n + 1 :]
        table.df.reset_index(drop=True, inplace=True)
        return table

    @classmethod
    def arguments(cls, table: Table) -> List[Tuple[int]]:
        """Use dtype and style to determine arguments.

        First try if can be detected using only dtypes.
        If that doesn't work, try to detect using style.

        If the table already has a header, return no arguments.

        """
        if table.header:
            return []
        arguments = cls.arguments_dtype(table)
        if len(arguments) == 0:
            arguments = cls.arguments_style(table)
        return arguments

    @classmethod
    def arguments_dtype(cls, table: Table) -> List[Tuple[int]]:
        """Arguments based on dtype.

        Look for the first row after which a consistent
        dtype is obtained.

        """
        arguments = list()
        df = table.dataframe
        for name in df:
            column = df[name]
            for i in range(len(column) // 2):
                t = pd.api.types.infer_dtype(column.iloc[i + 1 :], skipna=True)
                if "mixed" not in t:
                    if (i,) not in arguments:
                        arguments.append((i,))
                    break
        return arguments

    @classmethod
    def arguments_style(cls, table: Table) -> List[Tuple[int]]:
        """Arguments based on style.

        Look for the first row after which a consistent
        style is obtained.

        """
        arguments = list()
        for i in range(table.width):
            column = table.df.iloc[:, i]
            for k, v in column[0].style.items():
                styles = [c.style[k] == v for c in column]
                for j in range(len(styles) // 2):
                    if len(set(styles[j:])) == 1:
                        if j > 0 and ((j,) not in arguments):
                            arguments.append((j,))
                        break
        return [a for a in arguments if a[0] > 0]

    def __str__(self) -> str:
        return "Header({})".format(self._n)

    def __repr__(self) -> str:
        return "Header({})".format(repr(self._n))


class Fill(Transformation):
    """Forward fill a column."""

    def __init__(self, column: int):
        self._column = column

    def __call__(self, table: Table) -> pd.DataFrame:
        table = copy(table)
        table.df.iloc[:, self._column].replace([Cell(pd.NA)], np.nan, inplace=True)
        table.df.iloc[:, self._column].ffill(inplace=True)
        return table

    @classmethod
    def arguments(cls, table: Table) -> List[Tuple[int]]:
        arguments = list()
        for i in range(len(table.df.columns)):
            if any(not v for v in table.df.iloc[:, i]):
                arguments.append((i,))
        return arguments

    def __str__(self):
        return "ForwardFill({})".format(self._column)

    def __repr__(self) -> str:
        return "ForwardFill({})".format(repr(self._column))

    def __hash__(self):
        return hash(("ForwardFill", self._column))


class Fold(Transformation):
    """Unpivot transformation (wide to long).

    Can only fold tables with headers. Else,
    use a Stack with interval set to 1.

    """

    def __init__(self, column1: int, column2: int):
        self._column1 = column1
        self._column2 = column2

    def __hash__(self):
        return hash(("Fold", self._column1, self._column2))

    def __call__(self, table: Table) -> Table:
        """

        Reuse as much of `pd.melt` as possible.

        """
        table = copy(table)
        columns = table.df.columns.tolist()
        columns_value = columns[self._column1 : self._column2 + 1]
        columns_id = set(columns) - set(columns_value)
        table.df = pd.melt(table.df, value_vars=columns_value, id_vars=columns_id)
        return table

    @classmethod
    def arguments(cls, table: Table) -> List[Tuple[int, int]]:
        """Get fold arguments.

        By default, use only dtype and color information.

        Require that (1) the folded columns share dtype
        and (2) if color is available, all cells require
        the same color as they will end up in the same
        column.

        """
        if not table.header:
            return []
        # else, find the candidates
        arguments = list()
        color = table.color_df
        types = table.column_types
        # initial position
        a = next(i for i, t in enumerate(types) if "mixed" not in t)
        b = a
        # get set of colors
        colors = set(color.iloc[:, a].unique())
        while b < table.width - 1:
            b += 1
            colors.update(color.iloc[:, b].unique())
            # if mixed, look for next clean type
            if "mixed" in types[b]:
                a = next(
                    i + b for i, t in enumerate(types[b + 1 :]) if "mixed" not in t
                )
                b = a
            # some other criterium invalidated
            elif len(colors) > 2 or types[a] != types[b]:
                a = b
                colors = set(color.iloc[:, a].unique())
            # valid argument
            else:
                arguments.append((a, b))
        return arguments

    def __str__(self):
        return "Fold({}, {})".format(self._column1, self._column2)

    def __repr__(self):
        return "Fold({}, {})".format(repr(self._column1), repr(self._column2))


class Stack(Transformation):
    """Stack a range of column."""

    def __init__(self, column1: int, column2: int, interval: int):
        self._column1 = column1
        self._column2 = column2
        self._interval = interval

    def __call__(self, table: Table) -> Table:
        """ """
        return table

    @classmethod
    def arguments(cls, table: Table) -> List[Tuple[int, int]]:
        """Get stack arguments.

        Look for a range (a, b) of columns and an
        interval i such that i divides (b-a) and
        each column that gets stacked has the
        same dtype and style.

        """
        return table

    def __hash__(self):
        return hash(("Stack", self._column1, self._column2, self._interval))

    def __str__(self):
        return "Stack({}, {}, {})".format(self._column1, self._column2, self._interval)

    def __repr__(self):
        return "Stack({}, {}, {})".format(
            repr(self._column1), repr(self._column2), repr(self._interval)
        )


class DivideAndFill(Transformation):
    """Divide and forward fill.

    Split on having different values for a property.

    Divides the column in place.

    """

    def __init__(self, column: int, on: str = "dtype"):
        self._column = column
        self._on = on

    def __call__(self, table: Table) -> Table:
        # use divide
        divided = Divide(self._column, self._on)(table)
        # compute number of divided
        b = self._column
        e = self._column + (divided.width - table.width) + 1
        # forward fill the divided columns
        divided.df.iloc[:, b:e] = (
            divided.df.iloc[:, b:e]
            .replace([Cell(None)], np.nan)
            .ffill()
            .fillna(Cell(None))
        )
        return divided

    @classmethod
    def arguments(cls, table: Table) -> List[Tuple[int, str]]:
        """Arguments exactly the same as divide."""
        return Divide.arguments(table)

    def __hash__(self) -> int:
        return hash(("DivideAndFill", self._column, self._on))

    def __str__(self) -> str:
        return "DivideAndFill({}, {})".format(self._column, self._on)

    def __repr__(self) -> str:
        return "DivideAndFill({}, {})".format(repr(self._column), repr(self._on))
