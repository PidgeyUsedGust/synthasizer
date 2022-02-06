from abc import ABC, abstractmethod, abstractclassmethod
from ast import arg, arguments
from copy import copy
from itertools import chain
from turtle import width
from typing import List, Set, Tuple, Any, Union, Optional, Type, Iterable
import inspect
from unicodedata import name
from urllib.parse import parse_qs
import numpy as np
import pandas as pd
from pyparsing import col
from .conditions import Condition, EmptyCondition, StyleCondition
from .table import Table, Cell
from .utilities import nzs, infer_types, constants, transpose


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

    def __eq__(self, other: "Transformation") -> bool:
        if self.__class__ != other.__class__:
            return False
        return self.unpack() == other.unpack()

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, *self.unpack()))

    def unpack(self) -> Tuple[Any, ...]:
        """Get arguments."""
        names = inspect.signature(self.__init__).parameters
        values = tuple(getattr(self, "_{}".format(a)) for a in names)
        return values


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

    @classmethod
    def arguments_segmented(cls, table: Table) -> List[Tuple[int, Condition]]:
        arguments = list()
        for i in range(table.width):
            column = table.df.iloc[:, i]
            colors = table.color_df.iloc[:, i]
            colors = colors[colors > 0]
            print(colors)
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

    """Whether to use styles or not."""
    style = True

    def __init__(self, column: int, on: str = "datatype"):
        self._column = column
        self._on = on

    def __call__(self, table: Table) -> Table:
        table = copy(table)
        column = table[self._column]
        values = {c.style[self._on] for c in column}
        values.discard("empty")
        masks = [column.map(lambda x: x.style[self._on] == value) for value in values]
        columns = [pd.Series(column[mask], name=copy(column.name)) for mask in masks]
        data = pd.concat(columns, axis=1)
        table.df = pd.concat(
            (table[:, : self._column], data, table[:, self._column + 1 :]), axis=1
        ).fillna(Cell(None))
        return table

    @classmethod
    def arguments(cls, table: Table) -> List[Tuple[int, str]]:
        arguments = set(cls.arguments_datatype(table))
        if cls.style:
            arguments.update(cls.arguments_style(table))
        return list(arguments)

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
            if len(styles[i][values[i]]) < 1:
                continue
            union = set.union(*styles[i][values[i]], set())
            inter = set.intersection(*styles[i][values[i]])
            for (k, _) in union - inter:
                result.append((i, k))
        return result

    @classmethod
    def arguments_color(cls, table: Table) -> List[Tuple[int, str]]:
        pass

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
                columns = pd.MultiIndex.from_tuples(
                    transpose(table.df.iloc[: self._n].values)
                )
        else:
            columns = table.df.iloc[self._n - 1]
        table = copy(table)
        table.df.columns = columns
        table.df = table.df.iloc[self._n :]
        table.df.reset_index(drop=True, inplace=True)
        table.header = self._n
        return table

    @classmethod
    def arguments(cls, table: Table) -> List[Tuple[int]]:
        """Use dtype and style to determine arguments.

        First try if can be detected using only dtypes.
        If that doesn't work, try to detect using style.

        If the table already has a header, return no arguments.

        """
        if table.header > 0:
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

    threshold: float = 0.5

    def __init__(self, column: int):
        self._column = column

    def __call__(self, table: Table) -> pd.DataFrame:
        table = copy(table)
        value = Cell(pd.NA)
        column = table.df.iloc[:, self._column]
        for i, cell in column.iteritems():
            if not bool(cell):
                column[i] = value
            else:
                value = copy(cell)
                value.base = cell
        return table

    @classmethod
    def arguments(cls, table: Table) -> List[Tuple[int]]:
        """Look for columns to fill.

        Following requirements for filling a table.

         * Containing at least a percentage of empty values.
         * Containing all unique elements.
         * Must be type consistent.

        """
        arguments = list()
        for i in range(table.width):
            column = table[i]
            values = column[column.map(bool)]
            missing = len(values) / len(column)
            if (
                missing > 0
                and missing < cls.threshold
                and "mixed" not in table.column_types[i]
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

    Attributes:
        smart: If set to True, will try to use smarter
            detection of folds in single level headers.

    """

    smart: bool = False

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
        # build column of values
        val = pd.DataFrame(block.values.reshape((-1, 1)), columns=[Cell("value")])
        # build column(s) of variables
        var = pd.DataFrame(
            [
                *np.tile(
                    table.df.columns[self._column1 : self._column2 + 1].values,
                    table.height,
                )
            ],
            columns=[Cell("variable{}".format(i)) for i in range(table.header)],
        )
        # before and after pieces
        idx = table.df.index.repeat(self._column2 - self._column1 + 1)
        before = table.df.iloc[:, : self._column1].loc[idx].reset_index(drop=True)
        after = table.df.iloc[:, self._column2 + 1 :].loc[idx].reset_index(drop=True)
        # make new table
        table = copy(table)
        table.df = pd.concat((before, var, val, after), axis=1)
        # fix the index
        if table.header > 1:
            height = max(len(c) for c in table.df.columns if isinstance(c, tuple))
            values = list()
            for v in table.df.columns:
                if not isinstance(v, tuple):
                    v = (v,)
                values.append((Cell(None),) * (height - len(v)) + v)
            index = pd.MultiIndex.from_tuples(values)
            index = index.droplevel(
                [i for i, l in enumerate(index.levels) if not any(c for c in l)]
            )
            table.df.columns = index
            table.header = table.df.columns.nlevels
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

        if table.header == 0:
            return list()

        # initialise candidates
        candidates = set()

        # single header, look for some combinations of
        # columns that we think can be folded over
        if table.header == 1:
            if table.n_colors > 1:
                candidates.update(cls.columns_color(table))
            elif cls.smart:
                candidates.update(cls.columns_number(table))
                candidates.update(cls.columns_constant(table))

        # multi header
        else:
            if table.n_colors > 1:
                candidates.update(cls.columns_color(table))
            else:
                candidates.update(cls.columns_multi(table))

        # filter candidates with inconsistent types
        candidates = cls.filter_types(candidates, table)

        return list(candidates)

    @staticmethod
    def columns_color(table: Table) -> Set[Tuple[int, int]]:
        """Find candidates based on color."""
        arguments = set()
        # get properties used to compute folds
        properties = list()
        for i in range(table.width):
            properties.append(
                (
                    [nzs(h) for h in table.color_df.columns[i]],
                    nzs(table.color_df.iloc[:, i]),
                    table.column_types[i],
                )
            )
        # look for folds
        a = None
        for b, (new_h, new_t, new_type) in enumerate(properties):
            # folding
            if a is not None:
                old_h, old_t, old_type = properties[a]
                if (
                    any(len(nh | oh) > 1 for (nh, oh) in zip(new_h, old_h))
                    or len(new_t | old_t) > 1
                    or (new_type != "empty" and new_type != old_type)
                ):
                    arguments.add((a, b - 1))
                    a = None
            # not folding
            if a is None:
                if (
                    len(new_t) <= 1
                    and len(set.union(*new_h, new_t)) > 1
                    and "mixed" not in new_type
                ):
                    a = b
        # add final one
        if a is not None:
            arguments.add((a, b))
        return arguments

    @staticmethod
    def filter_overlapping(arguments: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        arguments = sorted(arguments)
        filtered = list()
        for i, a in enumerate(arguments):
            if i + 1 < len(arguments):
                b = arguments[i + 1]
                if b[0] > a[1]:
                    filtered.append(a)
            else:
                filtered.append(a)
        return filtered

    @staticmethod
    def filter_types(
        candidates: Set[Tuple[int, int]], table: Table
    ) -> Set[Tuple[int, int]]:
        """Filter arguments that do not fit the type."""
        arguments = set()
        for a, b in candidates:
            types = set(table.column_types[a : b + 1])
            types.discard("empty")
            if len(types) == 1:
                arguments.add((a, b))
        return arguments

    @staticmethod
    def columns_number(table: Table) -> Set[Tuple[int, int]]:
        """Look for column names that are numbers."""
        result = set()
        a = -1
        for b, cell in enumerate(table.df.columns):
            if cell.datatype == "integer":
                # start new serie
                if a < 0:
                    a = b
            else:
                if a > 0:
                    result.add((a, b - 1))
                    a = -1
        return result

    @staticmethod
    def columns_constant(table: Table) -> Set[Tuple[int, int]]:
        result = set()
        a, c = -1, None
        for b, cell in enumerate(table.df.columns):
            if cell.value in constants:
                # start new serie and end previous one
                # if it is happening
                if constants[cell.value] is not c:
                    if c is not None and a < b - 1:
                        result.add((a, b - 1))
                    a, c = b, constants[cell.value]
            # reset
            else:
                if c is not None and a < b - 1:
                    result.add((a, b - 1))
                a, c = -1, None
        # finished, add final one
        if a > 0:
            result.add((a, table.width - 1))
        return result

    @staticmethod
    def columns_multi(table: Table) -> Set[Tuple[int, int]]:
        columns = [[c.value for c in col] for col in transpose(table.df.columns)]
        x, last = columns[:-1], columns[-1]
        # find patterns in the last row of the index
        patterns = _get_patterns(last)
        # select those for which the upper levels match
        # as well, meaning either one value or also
        # a pattern.
        results = set()
        for (b, e, l) in patterns:
            for row in x:
                sets = [row[i : i + l] for i in range(b, e, l)]
                if (
                    all(len(set(s)) == 1 for s in sets)
                    or all(sets[0] == b for b in sets[1:]) == 1
                ):
                    results.add((b, e - 1))
        return results

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
        middle = pd.DataFrame(np.vstack([t.values for t in tostack]))
        middle.index = left.index
        right = pd.concat([table.df.iloc[:, self._column2 :]] * n, axis=0)
        # build header for middle part by copying
        # any colors and setting the base of all
        # removed cells to the remaining cell.
        header = tostack[0].columns
        for other in tostack[1:]:
            o_col = other.columns
            if isinstance(header, pd.MultiIndex):
                for i, column in enumerate(header):
                    for h, o in zip(header, column):
                        if o.color > 0 and h.color != o.color:
                            h.color = o.color
                        o.base = h
            else:
                for i, cell in enumerate(header):
                    if o_col[i].color > 0 and o_col[i].color != cell.color:
                        cell.color = o_col[i].color
                    o_col[i].base = cell
        middle.columns = header
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
        if table.header > 0:
            return cls.arguments_header(table)
        return list()

    @classmethod
    def arguments_header(cls, table: Table) -> List[Tuple[int, int, int]]:
        """Arguments if header.

        Look for headers that can be stacked.

        """
        results = list()
        # get list of tuples representing headers
        columns = table.df.columns
        if isinstance(columns, pd.MultiIndex):
            columns = [tuple(cell.value for cell in h) for h in columns]
        else:
            columns = [cell.value for cell in columns]
        # look for patterns
        for n in range(2, len(columns) // 2):
            i = 0
            while i < (len(columns) - 2 * n):
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
    """Transformation language.

    The language consists of phases of wrangling, which
    can be initialized with different transformations.

    As a simple example, the Stack and Fold transformations
    typically take place in later phases.

    """

    def __init__(
        self, transformations: Optional[List[Type[Transformation]]] = None
    ) -> None:
        self._transformations: List[Type[Transformation]] = transformations

    def candidates(self, table: Table) -> List[Transformation]:
        """Get transformations that can be applied."""
        candidates = list()
        for transformation in self._transformations:
            arguments = transformation.arguments(table)
            for argument in arguments:
                candidates.append(transformation(*argument))
        return candidates

    # def add_phase(self, transformations: List[Type[Transformation]]) -> None:
    #     """Add a new phase."""
    #     self._transformations.append(transformations)

    # def has_next_phase(self) -> bool:
    #     """Check whether a next phase is available."""
    #     return self._current < len(self._transformations)

    # def next_phase(self) -> None:
    #     """Advance to the next phase."""
    #     self._current += 1

    # @classmethod
    # def default(cls) -> "Language":
    #     language = Language()
    #     language.add_phase([Delete, Divide, Header, Fill])
    #     language.add_phase([Fold, Stack])
    #     return language


class Program:
    """Transformation program."""

    def __init__(self, transformations: Iterable[Transformation] = []) -> None:
        self.transformations: Tuple[Transformation] = tuple(transformations)

    def __call__(self, table: Table) -> Table:
        """Apply to a table."""
        for transformation in self.transformations:
            table = transformation(table)
        return table

    def __add__(self, other) -> "Program":
        return Program(self.transformations + other.transformations)

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


def _get_patterns(l: List[Any]) -> List[Tuple[int, int, int]]:
    """Find patterns in a list.

    Returns:
        A list of (start, end, stride) tuples.

    """
    result = list()
    index = 0
    while index < (len(l) - 4):
        for length in range(2, (len(l) - index) // 2):
            e = l[index : index + length]
            for n in range(1, (len(l) - index) // length):
                if l[index + n * length : index + n * length + length] != e:
                    break
            # if found a pattern, move pointer
            # past its end location
            if n > 1:
                result.append((index, index + n * length + length, length))
                index += (n * length) + length
                break
        # only increase if not found
        index += 1
    return result
