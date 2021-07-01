import openpyxl
from collections import Counter
from synthasizer.table import Table, detect
from synthasizer.transformation import *


def get_icecream() -> Table:
    return detect(openpyxl.load_workbook("data/icecream.xlsx")["icecream"])[0]


def get_nurse() -> Table:
    return detect(openpyxl.load_workbook("data/nurse.xlsx")["Sheet1"])[0]


def test_detect():
    tables = detect(openpyxl.load_workbook("data/icecream.xlsx")["icecream"])
    assert len(tables) == 2
    assert tables[0].width == 7
    assert tables[0].height == 10
    assert tables[1].width == 2
    assert tables[1].height == 5


def test_color():
    table = get_icecream()
    color = Header(1)(table).color(3, 3)
    assert table.df.iloc[3, 3].color == 0
    assert color.df.iloc[3, 3].color == 1


def test_color_df():
    table = get_icecream()
    header1 = Header(1)(table)
    header3 = Header(3)(table)
    assert table.color_df.sum().sum() == 0
    assert all(c == (0,) for c in header1.color_df.columns)
    assert all(c == (0, 0, 0) for c in header3.color_df.columns)


def test_cells():
    table = get_icecream()
    header1 = Header(1)(table)
    header3 = Header(3)(table)
    assert Counter(table.cells) == Counter(header1.cells)
    assert Counter(table.cells) == Counter(header3.cells)
    nurse = get_nurse()
    after = Delete(3, EmptyCondition())(Fill(1)(Stack(1, 29, 4)(Header(1)(nurse))))
    print(after.header)
    print(repr(after.df.columns[2]))
    print(after.cells)


def test_copy():
    table = get_icecream()
    color = table.color_all([(0, 1), (1, 1), (2, 1), (2, 0), (6, 1)])
    assert color.n_colors == 5
    assert table.n_colors == 0
    assert table.dataframe.equals(color.dataframe)


def test_style():
    table = get_icecream()
    cell = table.df.iloc[1, 0]
    assert cell.style["bold"] == True
    assert cell.style["random"] == None


def test_types():
    table = get_icecream()
    assert "string" in table.column_types
    assert "mixed-integer" in table.column_types
    header = Header(1)(table)
    assert "mixed-integer" not in header.column_types
    assert "integer" in header.column_types
    header2 = Header(2)(table)
    assert header.column_types == header2.column_types


if __name__ == "__main__":
    test_detect()
    test_copy()
    test_style()
    test_types()
    test_color()
    test_color_df()
    test_cells()
