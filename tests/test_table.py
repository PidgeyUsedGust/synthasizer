import openpyxl
from copy import copy
from openpyxl.worksheet.worksheet import Worksheet
from synthasizer.table import Table, detect


def get_icecream() -> Table:
    return detect(openpyxl.load_workbook("data/icecream.xlsx")["icecream"])[0]


def test_detect():
    tables = detect(openpyxl.load_workbook("data/icecream.xlsx")["icecream"])
    assert len(tables) == 2
    assert tables[0].width == 7
    assert tables[0].height == 10
    assert tables[1].width == 2
    assert tables[1].height == 5


def test_copy():
    table = get_icecream()
    color = copy(table)
    color.color(0, 1)
    color.color(1, 1)
    color.color(2, 1)
    color.color(2, 0)
    color.color(6, 1)
    assert len(color.color_dict) == 5
    assert len(table.color_dict) == 0
    assert table.dataframe.equals(color.dataframe)


if __name__ == "__main__":
    test_detect()
    test_copy()
