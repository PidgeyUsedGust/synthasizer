import openpyxl
from synthasizer.wrangle import State
from synthasizer.table import detect
from synthasizer.transformation import *
from test_table import get_icecream, get_nurse


def all_cells(df: pd.DataFrame) -> bool:
    data = df.applymap(lambda x: isinstance(x, Cell)).all().all()
    columns = all(isinstance(c, Cell) for c in df.columns.values.ravel("K"))
    return data and columns


def test_header():
    wb = openpyxl.load_workbook("data/icecream.xlsx")
    icecream = detect(wb["icecream"])[0]
    icecream_header = Header(1)(icecream)
    header = detect(wb["header"])[0]
    header_header = Header(2)(header)
    assert icecream_header.header == True
    assert icecream_header.dataframe.equals(header_header.dataframe)
    assert isinstance(header_header.df.columns, pd.Index)
    assert all_cells(icecream_header.df)


def test_header_arguments():
    wb = openpyxl.load_workbook("data/icecream.xlsx")
    # test regular
    icecream = detect(wb["icecream"])[0]
    icecream_arguments = Header.arguments(icecream)
    assert len(icecream_arguments) == 1
    assert (1,) in icecream_arguments
    # test
    header = detect(wb["header"])[0]
    header_arguments = Header.arguments(header)
    assert len(header_arguments) == 2
    assert (1,) in header_arguments
    assert (2,) in header_arguments


# def test_divide():
#     nurse = get_nurse()
#     inter =  Divide(18, dtype)(Delete(2, EmptyCondition)(nurse))


def test_divide_arguments():
    nurse = get_nurse()
    inter = Delete(2, EmptyCondition())(nurse)
    # print(Divide.arguments(inter))
    # print(inter)
    # print(Divide())


def test_fill():
    # load nurse and check for empty cells
    nurse = get_nurse()
    assert nurse.df.iloc[2, 1] == Cell(None)
    assert nurse.df.iloc[0, 1] == Cell(None)
    assert nurse.df.iloc[1, 1] == Cell("Mon")
    # fill and check
    filled = Fill(1)(nurse)
    assert filled.df.iloc[2, 1] == Cell("Mon")
    assert filled.df.iloc[0, 1] == Cell(None)
    assert all_cells(filled.df)


def test_stack_arguments():
    nurse = Header(1)(get_nurse())
    assert Stack.arguments(nurse) == [(1, 29, 4)]


def test_stack():
    nurse = Header(1)(get_nurse())
    stack = Stack(1, 29, 4)
    stacked = stack(nurse)
    assert stacked.width == (nurse.width - (29 - 1) + 4)
    assert stacked.height == ((29 - 1) // 4) * nurse.height
    assert stacked.header == True
    assert all_cells(stacked.df)


def test_fold_nurse():
    nurse = get_nurse()
    nurse = Stack(1, 29, 4)(Header(1)(nurse))
    after = Fold(2, 4)(nurse)
    assert after.width == nurse.width - 1
    assert after.height == nurse.height * (5 - 2)
    assert after.header == True
    assert all_cells(after.df)


def test_fold_arguments_icecream():
    table = get_icecream()
    table = table.color_all([(0, 1), (1, 1), (2, 1), (2, 0), (6, 1)])
    header = Header(1)(table)
    arguments = Fold.arguments(header)
    assert (2, 5) in arguments


def test_fold_arguments_nurse():
    nurse = get_nurse().color_all([(0, 2), (1, 1), (2, 0), (2, 2)])
    inter = Delete(3, EmptyCondition())(Fill(1)(Stack(1, 29, 4)(Header(1)(nurse))))
    arguments = Fold.arguments(inter)
    assert (2, 4) in arguments
    assert (1, 4) not in arguments
    assert Fold.arguments(nurse) == []


def test_delete_arguments():
    nurse = get_nurse()
    header = Header(1)(nurse)
    # print(nurse)
    # print(Delete.arguments(nurse))
    # print(Delete.arguments(nurse))


if __name__ == "__main__":
    test_header()
    test_header_arguments()
    test_fill()
    test_stack()
    test_fold_nurse()
    test_fold_arguments_icecream()
    test_fold_arguments_nurse()
    test_delete_arguments()
    test_divide_arguments()
