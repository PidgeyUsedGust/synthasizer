from collections import defaultdict
from numpy import stack
from synthasizer.conditions import DatatypeCondition
from synthasizer.transformation import *
from tests.conftest import nba


def all_cells(table: Table) -> bool:
    return all(isinstance(c, Cell) for c in table.cells)


def test_header(nurse):
    nurse = nurse.color_all([(0, 2), (1, 1), (2, 0), (2, 2)])
    nurse_header = Header(2)(nurse)
    assert nurse_header.header > 0
    assert isinstance(nurse_header.df.columns, pd.Index)
    assert all_cells(nurse_header)
    assert nurse_header.n_colors == nurse.n_colors


def test_header_arguments(icecream, nurse, nurse2):
    icecream_arguments = Header.arguments(icecream)
    assert len(icecream_arguments) == 1
    assert (1,) in icecream_arguments
    # multi level with one option
    nurse_arguments = Header.arguments(nurse)
    assert len(nurse_arguments) == 1
    assert (2,) in nurse_arguments
    # other nurse
    nurse2_arguments = Header.arguments(nurse2)
    assert len(nurse2_arguments) == 1
    assert (1,) in nurse2_arguments
    # icecream
    assert Header.arguments(Fill(0)(icecream)) == [(1,)]


def test_divide(car, nba):
    after = Divide(0, StyleCondition("bold", True))(car)
    assert set(a.value for a in after[0]) == {"Audi", "BMW", None}
    after = Divide(1, DatatypeCondition("string"))(Header(1)(nba))
    assert after.column_types[1] == "string"
    assert after.column_types[2] == "integer"


def test_divide_deeltijdswerk(deeltijdswerk):
    divided = Divide(1, StyleCondition("bold", True))(deeltijdswerk)
    assert divided.width == deeltijdswerk.width + 1


def test_divide_arguments(car, nba):
    arguments = Divide.arguments(car)
    assert all(c == 0 for c, _ in arguments)
    after = Divide(0, StyleCondition("bold", True))(car)
    assert Divide.arguments(after) == set()


def test_divide_arguments_color(car):
    car = car.color_all([(1, 1), (1, 2)])
    assert (1, StyleCondition("bold", True)) in Divide.arguments(car)


def test_fill(icecream, car):
    assert icecream.df.iloc[1, 0].value == "Banana"
    assert not icecream.df.iloc[2, 0]
    # assert icecream.df.iloc[1, 0].base is None
    # assert icecream.df.iloc[2, 0].base is None
    filled = Fill(0)(icecream)
    assert filled.df.iloc[1, 0].value == "Banana"
    assert filled.df.iloc[2, 0].value == "Banana"
    assert filled.df.iloc[2, 0].base is not None
    # assert filled.df.iloc[1, 0].base is None
    car = Divide(0, PatternCondition("word"))(car)
    car_filled = Fill(0)(car)
    assert all(bool(c) for c in car_filled[0])


def test_fill_arguments(nba, car):
    nba = Header(1)(nba)
    assert Fill.arguments(nba) == [(0,)]
    car = Divide(0, PatternCondition("words"))(car)
    assert Fill.arguments(car) == [(0,)]


def test_stack_arguments(nurse2):
    nurse2 = Header(1)(nurse2)
    assert Stack.arguments(nurse2) == [(1, 29, 4)]


def test_stack(nurse2):
    nurse = Header(1)(nurse2)
    stacked = Stack(1, 29, 4)(nurse)
    assert stacked.width == (nurse.width - (29 - 1) + 4)
    assert stacked.height == ((29 - 1) // 4) * nurse.height
    assert stacked.header == True
    assert all_cells(stacked)
    # get removed cells and test whether each value
    # points to the same base cell and that this
    # cell is still in the table
    removed = set(nurse2.cells) - set(stacked.cells)
    base = defaultdict(set)
    for cell in removed:
        base[cell.value].add(cell.base)
        assert cell.base in stacked.cells
    assert all(len(v) == 1 for v in base.values())


def test_fold_single(icecream):
    folded = Fold(2, 4)(Header(1)(icecream))
    assert all_cells(folded)
    # print(folded)


def test_fold_multi(nurse):
    nurse = Header(2)(nurse)
    fold1 = Fold(1, 3)(nurse)
    fold2 = Fold(4, 6)(fold1)
    assert all_cells(fold1)
    assert all_cells(fold2)
    # both folds map three columns to three columns
    assert nurse.width == fold2.width
    # fold three columns, thus scale height two
    # times by three
    assert fold2.height == nurse.height * 3 * 3
    fold3 = Fold(1, 21)(nurse)


def test_fold_arguments_single(icecream, icecreamyear):

    icecream = Header(1)(icecream)
    icecreamyear = Header(1)(icecreamyear)

    # first try with dumb fold,
    # which does nothing.
    Fold.smart = False
    assert Fold.arguments(icecream) == []
    assert Fold.arguments(icecreamyear) == []

    # then try for smart folding
    Fold.smart = True
    assert Fold.arguments(icecream) == [(2, 4)]
    assert Fold.arguments(icecreamyear) == [(2, 4)]


def test_fold_arguments_single_color(icecream):
    table = icecream.color_all([(2, 1), (2, 0), (6, 1)])
    header = Header(1)(table)
    arguments = Fold.arguments(header)
    assert (2, 5) in arguments


def test_fold_arguments_multi(deeltijdswerk, nurse):
    deeltijdswerk = Header(2)(deeltijdswerk)
    nurse = Header(2)(nurse)
    assert Fold.arguments(deeltijdswerk) == [(1, 12)]
    assert Fold.arguments(nurse) == [(1, 21)]


def test_fold_arguments_multi_color(deeltijdswerk, part_e):
    # color everything
    assert Fold.arguments(
        Header(2)(deeltijdswerk.color_all([(0, 0), (1, 0), (1, 1), (1, 2)]))
    ) == [(1, 12)]
    # part of header and part of table
    assert Fold.arguments(Header(2)(deeltijdswerk.color_all([(1, 1), (1, 2)]))) == [
        (1, 12)
    ]
    # only color header
    assert Fold.arguments(Header(2)(deeltijdswerk.color_all([(1, 0), (1, 1)]))) == [
        (1, 12)
    ]
    # early stop
    assert Fold.arguments(
        Header(2)(deeltijdswerk.color_all([(1, 0), (1, 1), (4, 1)]))
    ) == [(1, 3)]
    # empty
    assert Fold.arguments(Header(2)(part_e.color_all([(1, 0), (1, 1), (1, 2)]))) == [
        (1, 10)
    ]
