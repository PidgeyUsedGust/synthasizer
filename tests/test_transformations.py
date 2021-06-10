from tests.test_table import test_copy
from synthasizer.table import detect
import openpyxl
from synthasizer.transformations import *


def test_header():
    wb = openpyxl.load_workbook("data/icecream.xlsx")
    icecream = detect(wb["icecream"])[0]
    icecream_header = Header(1)(icecream)
    header = detect(wb["header"])[0]
    header_header = Header(2)(header)
    assert icecream_header.dataframe.equals(header_header.dataframe)
    assert isinstance(header_header.df.columns, pd.Index)


def test_header_arguments():
    wb = openpyxl.load_workbook("data/icecream.xlsx")
    # test regular
    icecream = detect(wb["icecream"])[0]
    icecream_arguments = Header.arguments(icecream)
    assert len(icecream_arguments) == 1
    assert (0,) in icecream_arguments
    # test
    header = detect(wb["header"])[0]
    header_arguments = Header.arguments(header)
    assert len(header_arguments) == 2
    assert (0,) in header_arguments
    assert (1,) in header_arguments


if __name__ == "__main__":
    test_header()
    test_header_arguments()
