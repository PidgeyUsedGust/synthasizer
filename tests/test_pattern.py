from typing import Dict

from numpy.lib.index_tricks import diag_indices
from synthasizer import pattern
from synthasizer.transformation import *
from synthasizer.pattern import Element, Pattern, Variable, unpack


def test_arguments():

    delete_args = (2, EmptyCondition())
    delete = Delete(*delete_args)
    assert unpack(delete) == delete_args

    divide_args = (0, "datatype")
    divide = Divide(*divide_args)
    assert unpack(divide) == divide_args

    header_args = (8,)
    header = Header(*header_args)
    assert unpack(header) == header_args

    fill_args = (2,)
    fill = Fill(*fill_args)
    assert unpack(fill) == fill_args

    fold = Fold(2, 4)
    assert unpack(fold) == (2, 4)

    stack_args = (1, 2, 3)
    stack = Stack(*stack_args)
    assert unpack(stack) == stack_args


def test_element():

    e = Element(Fold, (2, None))
    assert e.matches(Fold(2, 5))
    assert not e.matches(Fold(3, 5))

    e = Element(Stack, (1, None, 3))
    assert e.matches(Stack(1, 10, 3))
    assert not e.matches(Fold(1, 2))

    f = Delete(0, EmptyCondition())
    assert Element.from_transformation(f).matches(f)

    e = Element(Stack)
    assert e.matches(Stack(1, 10, 3))


def test_pattern_match():

    program = [Header(1), Stack(1, 29, 4), Fill(1)]
    pattern = Pattern([Element(Header, (None,)), Element(Stack, (None, None, None))])
    assert pattern.match(program)


def test_pattern_variable():
    program = [Divide(1, "datatype"), Fill(1)]
    x = Variable()
    pattern = Pattern(
        [
            Element(Divide, (x, None)),
            Element(Fill, (x,)),
        ]
    )
    assert pattern.match(program)


if __name__ == "__main__":
    test_arguments()
    test_element()
    test_pattern_match()
    test_pattern_variable()
