import pandas as pd
import numpy as np
from typing import List, Any, Optional, Dict, Tuple
from functools import cached_property


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

    def __hash__(self):
        return hash((self.value,))

    def __str__(self):
        return str(self.value)

    @classmethod
    def from_openpyxl(cls, cell: "PyxlCell") -> "Cell":
        new_cell = cls(
            cell.value,
            bold=cell.font.bold,
            italic=cell.font.italic,
            underline=cell.font.underline,
            size=cell.font.size,
            color=cell.font.color.value,
            fill=cell.fill.fgColor.value,
        )
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
        """Get colors as dataframe."""
        return self.df.applymap(lambda cell: cell.color)

    @cached_property
    def color_dict(self) -> Dict[Tuple[int, int], int]:
        """Get colors as a dictionary."""
        indices = zip(*np.where(self.color_df > 0))
        return {(y, x): self.df.iloc[y, x].color for (y, x) in indices}

    @cached_property
    def column_types(self) -> List[str]:
        return [pd.api.types.infer_dtype(self.dataframe[c]) for c in self.dataframe]

    @cached_property
    def dataframe(self) -> pd.DataFrame:
        return self.df.applymap(lambda cell: cell.value).convert_dtypes()

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
    def from_openpyxl(cls, data: List[List["PyxlCell"]]):
        """Load from openpyxl cells."""
        from openpyxl.cell.cell import Cell as PyxlCell

        cells = [[Cell.from_openpyxl(c) for c in row] for row in data]
        table = Table()
        table.df = pd.DataFrame(cells)
        return table

    @classmethod
    def from_spreadsheet(cls, file: str):
        """Load from spreadsheet.
        
        Args:
            file: Filename.
            tables: Ranges of tables.

        """
        pass

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
        new.df = self.df.copy()
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
}
"""Default NA values from pandas."""


def none(value: Any):
    """Check if value is nan."""
    if (value in na_values) or pd.isna(value):
        return None
    return value
