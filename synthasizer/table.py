"""
Table wrapper with support for colorings as they
need to be propagated through the table after
applying transformations.
"""
from copy import copy
import itertools
import pandas as pd
import numpy as np
from typing import List, Any, Optional, Dict, Tuple
from functools import cached_property
from operator import itemgetter
from openpyxl.cell.cell import Cell as PyxlCell
from openpyxl.worksheet.worksheet import Worksheet
from openpyxl.utils import get_column_letter


class Cell:
    """A spreadsheet cell.

    Style is represented as a simple dictionary
    that can be used to set any cell properties.

    Color is an integer defining the type of cell.
    It can also be used to define other criteria,
    such as a segmentation.

    """

    def __init__(self, value: Optional[Any] = None, **kwargs):
        self.value = none(value)
        self.style = dict(kwargs)
        self.color = 0

    def same_style(self, other: "Cell") -> bool:
        """Check if same style.

        Only returns true if exactly the same. Probably
        not very robust in practice.

        """
        if set(self.style) != set(other.style):
            return False
        return all(self.style[k] == v for k, v in other.style.items())

    @property
    def dtype(self) -> str:
        return pd.api.types.infer_dtype([self.value])

    @property
    def is_colored(self) -> bool:
        return self.color > 0

    def __getattr__(self, name: str):
        if name in self.__dict__["style"]:
            return self.__dict__["style"][name]
        else:
            raise AttributeError

    def __eq__(self, other):
        return isinstance(other, Cell) and (self.value == other.value)

    def __ne__(self, other):
        return self.value != other.value

    def __lt__(self, other):
        return self.value < other.value

    def __gt__(self, other):
        return self.value > other.value

    def __bool__(self):
        return not pd.isna(self.value)

    def __copy__(self):
        cell = Cell(self.value)
        cell.style = copy(self.style)
        cell.color = self.color
        return cell

    def __hash__(self):
        return hash((self.value,))

    def __str__(self):
        if self.value is None:
            return ""
        return str(self.value)

    def __repr__(self):
        return "Cell({})".format(repr(self.value))

    @classmethod
    def from_openpyxl(cls, cell: PyxlCell) -> "Cell":
        # extract styles from cell
        style = dict(
            bold=cell.font.bold,
            italic=cell.font.italic,
            underline=cell.font.underline,
            size=cell.font.size,
            fill=cell.fill.fgColor.value,
        )
        # sometimes font color is not set
        if cell.font.color is not None:
            style["color"] = cell.font.color.value
        # build cell
        new_cell = cls(cell.value, **style)
        return new_cell


class Table:
    """Table wrapper."""

    def __init__(self, df: Optional[pd.DataFrame] = None):
        """Initialise table.

        Args:
            df: A pandas dataframe.

        """
        self.df = pd.DataFrame()
        self.current_color = 1
        if df is not None:
            data = [[Cell(v) for v in row] for _, row in df.iterrows()]
            columns = [Cell(v) for v in df.columns]
            self.df = pd.DataFrame(data, columns=columns)

    def color(self, x: int, y: int):
        if not self.df.iloc[y, x].is_colored:
            self.df.iloc[y, x].color = self.current_color
            self.current_color += 1

    @property
    def color_df(self) -> pd.DataFrame:
        """Colors as dataframe."""
        return self.df.applymap(lambda cell: cell.color)

    @property
    def color_dict(self) -> Dict[Tuple[int, int], int]:
        """Colors as a dictionary."""
        indices = zip(*np.where(self.color_df > 0))
        return {(y, x): self.df.iloc[y, x].color for (y, x) in indices}

    @cached_property
    def column_types(self) -> List[str]:
        """List of the type of each column."""
        return [pd.api.types.infer_dtype(self.dataframe[c]) for c in self.dataframe]

    @cached_property
    def dataframe(self) -> pd.DataFrame:
        """Unwrapped dataframe."""
        return self.df.applymap(lambda cell: cell.value).convert_dtypes()

    @property
    def cells(self) -> List[Cell]:
        return list(self.df.values.ravel("K"))

    @property
    def height(self) -> int:
        return len(self.df.index)

    @property
    def width(self) -> int:
        return len(self.df.columns)

    @property
    def header(self) -> bool:
        return not isinstance(self.df.columns, pd.RangeIndex)

    @classmethod
    def from_csv(cls, file: str, header: Optional[int] = None):
        """Load from CSV."""
        return Table(pd.read_csv(file, header=header))

    @classmethod
    def from_openpyxl(cls, data: List[List[PyxlCell]]):
        """Load from openpyxl cells."""
        cells = [[Cell.from_openpyxl(c) for c in row] for row in data]
        table = Table()
        table.df = pd.DataFrame(cells)
        return table

    def __getitem__(self, i):
        """Implement slicing.

        By default, forward everything to iloc, except
        for single integers, which are considered
        to be columns.

        """
        if isinstance(i, int):
            return self.df.iloc[:, i]
        return self.df.iloc[i]

    def __copy__(self):
        new = Table()
        new.df = self.df.applymap(copy)
        return new

    def __str__(self):
        return str(self.df)

    def __repr__(self):
        return str(self.df)


na_values = {
    "",
    "-1.#IND",
    "1.#QNAN",
    "1.#IND",
    "-1.#QNAN",
    "#N/A",
    "N/A",
    "NA",
    "#NA",
    "NULL",
    "NaN",
    "-NaN",
    "nan",
    "-nan",
    "<NA>",
}
"""Default NA values from pandas."""


def none(value: Any):
    """Check if value is nan."""
    if (value in na_values) or pd.isna(value):
        return None
    return value


def detect(sheet: Worksheet) -> List[Table]:
    return [Table.from_openpyxl(sheet[i]) for i in detect_ranges(sheet)]


def detect_ranges(sheet: Worksheet) -> List[str]:
    """Extract tables from a worksheet."""
    # initialise mask
    mask = np.zeros((sheet.max_row, sheet.max_column), dtype=int)
    for row in sheet.rows:
        for cell in row:
            if cell.value is not None:
                mask[cell.row - 1, cell.column - 1] = 1.0
    # look for tables
    tables = list()
    while np.count_nonzero(mask) > 0:
        # find nonzero cell
        point = next(zip(*np.where(mask > 0)))
        # detect connected region
        x1, y1, x2, y2 = detect_from(mask, point)
        # add to results
        tables.append("{}:{}".format(to_excel((y1, x1)), to_excel((y2, x2))))
        # zero out the discovered region
        mask[x1 : x2 + 1, y1 : y2 + 1] = 0
    return tables


def detect_from(table: np.ndarray, index: Tuple[int, int]) -> str:
    """Detect table from a given index."""
    queue = [index]
    cells = set()
    while len(queue) > 0:
        a, b = queue.pop()
        for dx, dy in itertools.product((-1, 0, 1), repeat=2):
            if a + dx >= 0 and b + dy >= 0:
                try:
                    if table[a + dx, b + dy] != 0:
                        if (a + dx, b + dy) not in cells:
                            queue.append((a + dx, b + dy))
                            cells.add(((a + dx, b + dy)))
                except IndexError:
                    pass
    return (
        min(cells, key=itemgetter(0))[0],
        min(cells, key=itemgetter(1))[1],
        max(cells, key=itemgetter(0))[0],
        max(cells, key=itemgetter(1))[1],
    )


def to_excel(point: Tuple[int, int]) -> str:
    """Make into Excel coordinate.

    Args:
        point: A one-indexed (column, row) tuple.

    """
    return get_column_letter(point[0] + 1) + str(point[1] + 1)
