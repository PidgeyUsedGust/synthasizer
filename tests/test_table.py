from email import header
import openpyxl
from collections import Counter
from synthasizer import table
from synthasizer.table import Table, detect
from synthasizer.transformation import *


def test_detect():
    tables = detect(openpyxl.load_workbook("data/icecream.xlsx")["icecream"])
    assert len(tables) == 2
    assert tables[0].width == 7
    assert tables[0].height == 10
    assert tables[1].width == 2
    assert tables[1].height == 5


def test_color(icecream):
    color = Header(1)(icecream).color(3, 3)
    assert icecream.df.iloc[3, 3].color == 0
    assert color.df.iloc[3, 3].color == 1


def test_color_df(icecream):
    header1 = Header(1)(icecream)
    header3 = Header(3)(icecream)
    assert icecream.color_df.sum().sum() == 0
    assert all(c == (0,) for c in header1.color_df.columns)
    assert all(c == (0, 0, 0) for c in header3.color_df.columns)


def test_cells(icecream, nurse):
    header1 = Header(1)(icecream)
    header3 = Header(3)(icecream)
    assert Counter(icecream.cells) == Counter(header1.cells)
    assert Counter(icecream.cells) == Counter(header3.cells)
    # after = Delete(3, EmptyCondition())(Fill(1)(Stack(1, 29, 4)(Header(2)(nurse))))
    # print(after)


def test_copy(icecream):
    color = icecream.color_all([(0, 1), (1, 1), (2, 1), (2, 0), (6, 1)])
    assert color.n_colors == 5
    assert icecream.n_colors == 0
    assert icecream.dataframe.equals(color.dataframe)


def test_style(icecream):
    cell = icecream.df.iloc[1, 0]
    assert cell.style["bold"] == True
    assert cell.style["random"] == None


def test_types(icecream, nurse):
    assert "string" in icecream.column_types
    assert "mixed-integer-string" in icecream.column_types
    header = Header(1)(icecream)
    assert "mixed-integer" not in header.column_types
    assert "integer" in header.column_types
    header2 = Header(2)(icecream)
    assert header.column_types == header2.column_types


def test_hash(icecream):
    icecream = Header(1)(icecream)

    d1 = Delete(2, EmptyCondition())(icecream)
    d2 = Delete(5, EmptyCondition())(icecream)
    assert hash(d1) == hash(d2)

    f1 = Fill(0)(icecream)
    f2 = Fill(0)(icecream)
    assert hash(f1) == hash(f2)

    f3 = Fold(2, 5)(icecream)
    f4 = Fold(2, 5)(icecream)
    assert hash(f3) == hash(f4)

    d3 = Delete(3, StyleCondition("bold", True))(icecream)
    d4 = Header(1)(icecream)
    assert hash(d3) != hash(d4)


# if __name__ == "__main__":
#     test_detect()
#     test_copy()
#     test_style()
#     test_types()
#     test_color()
#     test_color_df()
#     test_cells()
#     test_hash()
