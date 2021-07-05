from abc import ABC, abstractmethod, abstractclassmethod
from copy import copy
from operator import attrgetter
from typing import Callable, List, Tuple, Any, Union, Optional, Type, Iterable
from collections import Counter
import numpy as np
import pandas as pd
from .conditions import Condition, EmptyCondition, StyleCondition
from .table import Table, Cell
from .utilities import duplicates, nzs, infer_types


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

    conditions = [EmptyCondition, StyleCondition]
    """List of conditions."""

    def __init__(self, column: int, condition: Condition):
        self._column = column
        self._condition = condition

    def __call__(self, table: Table):
        table = copy(table)
        table.df = table.df[~table.df.iloc[:, self._column].map(self._condition)]
        table.df.reset_index(drop=True, inplace=True)
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

    def __init__(self, column: int, on: str = "datatype"):
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
        return list(set(cls.arguments_datatype(table) + cls.arguments_style(table)))

    @classmethod
    def arguments_datatype(cls, table: Table) -> List[Tuple[int, str]]:
        ctypes = table.cell_types.T
        result = list()
        for i in range(table.width):
            types = set(ctypes[i])
            types.discard("empty")
            if len(types) > 1:
                result.append((i, "datatype"))
        return result

    @classmethod
    def arguments_style(cls, table: Table) -> List[Tuple[int, str]]:
        values = (table.df.values != Cell(None)).T
        styles = np.vectorize(cls.extract_style)(table.df.values).T
        result = list()
        for i in range(table.width):
            union = set.union(*styles[i][values[i]])
            inter = set.intersection(*styles[i][values[i]])
            for (k, _) in union - inter:
                result.append((i, k))
        return result

    @classmethod
    def extract_style(cls, cell: Cell) -> set[Tuple[str, Any]]:
        if not cell:
            return set()
        return set(cell.style.items())

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
        table.df = table.df.iloc[self._n :]
        table.df.reset_index(drop=True, inplace=True)
        table.header = True
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
        for i in range(table.width):
            column = table[i]
            column_types = [cell.datatype for cell in column]
            if "mixed" in infer_types(column_types):
                for i in range(1, len(column) // 2):
                    t = infer_types(column_types[i:])
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

    threshold: float = 0.8

    def __init__(self, column: int):
        self._column = column

    def __call__(self, table: Table) -> pd.DataFrame:
        table = copy(table)
        table.df.iloc[:, self._column] = (
            table.df.iloc[:, self._column]
            .replace([Cell(pd.NA)], np.nan)
            .ffill()
            .fillna(Cell(None))
        )

        return table

    @classmethod
    def arguments(cls, table: Table) -> List[Tuple[int]]:
        """Look for columns to fill.

        Following requirements for filling a table.

         * Containing at least a percentage of empty values.
         * Containing all unique elements.

        """
        arguments = list()
        for i in range(table.width):
            column = table[i]
            values = column[column.map(bool)]
            missing = len(values) / len(column)
            if (
                missing > 0
                and missing < cls.threshold
                and not values.duplicated().any()
            ):
                arguments.append((i,))
        return arguments

    def __str__(self):
        return "Fill({})".format(self._column)

    def __repr__(self) -> str:
        return "Fill({})".format(repr(self._column))

    def __hash__(self):
        return hash(("Fill", self._column))


class Fold(Transformation):
    """Unpivot transformation (wide to long).

    Can only fold tables with headers. Else,
    use a Stack with interval set to 1.

    """

    def __init__(self, column1: int, column2: int):
        """

        Args:
            column1, column2: Indices of columns between
                which should be folded.

        """
        self._column1 = column1
        self._column2 = column2

    def __hash__(self):
        return hash(("Fold", self._column1, self._column2))

    def __call__(self, table: Table) -> Table:
        """Custom Fold implementation.

        Supports duplicate columns outside of the
        folding range.

        """
        # slice out block of values to be folded
        block = table.df.iloc[:, self._column1 : self._column2 + 1]
        # past them one after another
        val = pd.DataFrame(block.values.reshape((-1, 1)), columns=[Cell("value")])
        var = pd.DataFrame(
            np.tile(
                table.df.columns[self._column1 : self._column2 + 1].values, table.height
            ),
            columns=[Cell("variable")],
        )
        # before and after pieces
        idx = table.df.index.repeat(self._column2 - self._column1 + 1)
        before = table.df.iloc[:, : self._column1].loc[idx].reset_index(drop=True)
        after = table.df.iloc[:, self._column2 + 1 :].loc[idx].reset_index(drop=True)
        # make new table
        table = copy(table)
        table.df = pd.concat((before, var, val, after), axis=1).reset_index(drop=True)
        return table

    def __call__old(self, table: Table) -> Table:
        """Fold columns.

        Reuse as much of `pd.melt` as possible,
        which requires to add a new column to
        ensure duplicates aren't complained about.

        """
        table = copy(table)
        columns = table.df.columns.tolist()
        columns_value = columns[self._column1 : self._column2 + 1]
        columns_id = [column for column in columns if column not in columns_value]
        # get mapping and inverse
        table.df = pd.melt(
            table.df,
            value_vars=columns_value,
            id_vars=columns_id,
            var_name=[Cell("variable")],
            value_name=Cell("value"),
        )
        return table

    @classmethod
    def get_mapping(
        cls,
        columns: List[Cell],
    ) -> Tuple[Callable[[Cell], Cell], Callable[[Cell], Cell]]:
        """Return two mappings.

        Returns:
            Two cell -> cell functions, one for deduplicating
            and one for restoring.

        """

        duplicated = duplicates(columns)
        m = {id(d): Cell("!{}".format(i)) for i, d in enumerate(duplicated)}
        i = {v: k for k, v in m.items()}

        def mapping(cell: Cell) -> Cell:
            i = id(cell)
            if i in m:
                return m[id(cell)]
            return cell

        def inverse(cell: Cell) -> Cell:
            if cell in i:
                return i[cell]
            return cell

        return mapping, inverse

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
        # initial position
        a, b = 0, 0
        # get the colors
        colors_header = nzs(table.color_df.columns[a])
        colors_table = nzs(table.color_df.iloc[:, a])
        while b < table.width - 1:
            b += 1
            colors_header.update(table.color_df.columns[b])
            colors_table.update(table.color_df.iloc[:, b])
            # check if range became invalid
            if (
                len(colors_table) > 1
                or len(colors_header) > 1
                or len(colors_header & colors_table) > 0
                or table.column_types[a] != table.column_types[b]
                or "mixed" in table.column_types[a]
            ):
                a = b
                colors_header = nzs(table.color_df.columns[a])
                colors_table = nzs(table.color_df.iloc[:, a])
            elif b > a:
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
        """Build new table by stacking manually."""
        tostack = [
            table.df.iloc[:, self._column1 + i : self._column1 + i + self._interval]
            for i in range(0, self._column2 - self._column1, self._interval)
        ]
        n = len(tostack)
        # build parts
        left = pd.concat([table.df.iloc[:, : self._column1]] * n, axis=0)
        middle = pd.concat(tostack)
        right = pd.concat([table.df.iloc[:, self._column2 :]] * n, axis=0)
        # combine
        table = copy(table)
        table.df = pd.concat((left, middle, right), axis=1).reset_index(drop=True)
        return table

    @classmethod
    def arguments(cls, table: Table) -> List[Tuple[int, int, int]]:
        """Get stack arguments.

        Look for a range (a, b) of columns and an
        interval i such that i divides (b-a) and
        each column that gets stacked has the
        same dtype and style.

        If headers are present, they need to be the same.

        """
        if table.header:
            return cls.arguments_header(table)
        return list()

    @classmethod
    def arguments_header(cls, table: Table) -> List[Tuple[int, int, int]]:
        """Arguments if header.

        Look for headers that can be stacked.

        """
        results = list()
        columns = table.df.columns.tolist()
        # loop over number of columns to stack
        for n in range(2, len(columns) // 2):
            # loop over initial positions
            i = 0
            while i < (len(columns) - 2 * n):
                # loop over possible ending positions
                for j in reversed(range(i + 2 * n, len(columns) + 1, n)):
                    header = {tuple(columns[a : a + n]) for a in range(i, j - n + 1, n)}
                    # if header can be stacked, check if can't be stacked further
                    if len(header) == 1:
                        new = next(iter(header))
                        if not _has_pattern(new):
                            results.append((i, j, n))
                        i += j
                i += 1
        return results

    def __hash__(self):
        return hash(("Stack", self._column1, self._column2, self._interval))

    def __str__(self):
        return "Stack({}, {}, {})".format(self._column1, self._column2, self._interval)

    def __repr__(self):
        return "Stack({}, {}, {})".format(
            repr(self._column1), repr(self._column2), repr(self._interval)
        )


class Language:
    """Transformation language."""

    def __init__(
        self, transformations: Optional[List[Type[Transformation]]] = None
    ) -> None:
        self._transformations = transformations or list()

    def candidates(self, table: Table) -> List[Transformation]:
        """Get transformations that can be applied."""
        candidates = list()
        for transformation in self._transformations:
            arguments = transformation.arguments(table)
            for argument in arguments:
                candidates.append(transformation(*argument))
        return candidates


class Program:
    """Transformation program."""

    def __init__(self, transformations: Iterable[Transformation] = []) -> None:
        self.transformations: Tuple[Transformation] = tuple(transformations)

    def __call__(self, table: Table) -> Table:
        """Apply to a table."""
        for transformation in self.transformations:
            table = transformation(table)
        return table

    def __len__(self) -> int:
        return len(self.transformations)

    def __hash__(self) -> int:
        return hash(self.transformations)

    def __str__(self) -> str:
        return "\n".join(map(str, self.transformations))

    def __repr__(self) -> str:
        return "[{}]".format(", ".join(map(str, self.transformations)))

    def __eq__(self, other: "Program") -> bool:
        return self.transformations == other.transformations

    def extend(self, transformation: Transformation) -> "Program":
        """Extend program with a new transformation."""
        return Program((*self.transformations, transformation))

    @property
    def python(self) -> str:
        """Convert to Python code."""
        program = "x"
        for transformation in self.transformations:
            program = "{}({})".format(repr(transformation), program)
        return program


def _has_pattern(l: List[Any]) -> bool:
    """Check if a list consists of a pattern."""
    for i in range(2, len(l) // 2 + 1):
        if len(l) % i == 0:
            p = l[:i]
            n = len(l) // i
            if p * n == l:
                return True
    return False
