from errno import EEXIST
from black import err
import openpyxl
from synthasizer.error import ThresholdedReconstructionError
from synthasizer.transformation import (
    Divide,
    Header,
    Program,
    Fill,
    Delete,
    EmptyCondition,
    Stack,
)


def test_error_duplicate(icecream):
    # fill and then remove one of the filled values
    error = ThresholdedReconstructionError()
    error.initialise(icecream)
    after = Program([Fill(0), Delete(1, EmptyCondition())])(icecream)
    assert error.compute(after) == 0.0


def test_error_color(nurse):
    nurse = nurse.color_all([(0, 2), (1, 1), (2, 0), (2, 2)])
    after = Header(2)(nurse)
    error = ThresholdedReconstructionError()
    error.initialise(nurse)
    assert error(after) == 0
    assert after.n_colors == nurse.n_colors


def test_error_stack(nurse2):
    error = ThresholdedReconstructionError()
    error.initialise(nurse2)
    after = Stack(1, 29, 4)(Header(1)(nurse2))
    assert error(after) == 0


def test_error_fill(nurse2, car):
    error = ThresholdedReconstructionError()
    error.initialise(nurse2)
    after = Delete(0, EmptyCondition())(Fill(1)(Stack(1, 29, 4)(Header(1)(nurse2))))

    error.initialise(car)
    after = Fill(0)(Divide(0, "datatype")(car))
    assert error(after) == 0
    after = Delete(1, EmptyCondition())(after)
    assert error(after) == 0
