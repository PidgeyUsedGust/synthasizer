import pytest
from synthasizer import transformation
from synthasizer.heuristics import *
from synthasizer.settings import Optimizations
from synthasizer.transformation import *
from synthasizer.wrangle import *


@pytest.fixture
def wrangler():
    phase_one = (
        Language([Delete, Divide, Header, Fill, Stack]),
        WeightedHeuristic(
            [
                EmptyColumnHeuristic(),
                EmptyRowHeuristic(),
                TypeHeuristic(),
                ColorRowHeuristic(False),
            ],
            weights=[0.1, 0.1, 1, 1],
        ),
    )
    phase_two = (
        Language([Fold, Stack]),
        WeightedHeuristic(
            [
                TypeHeuristic(),
                ColorRowHeuristic(True),
                WidthHeuristic(),
            ],
            weights=[1, 1, 0.5],
        ),
    )
    return Wrangler([phase_one, phase_two], max_depth=5, max_time=10, verbose=True)


def test_nurse(nurse, wrangler):
    programs = wrangler.learn(nurse)
    assert programs[0].program == Program([Header(2), Fold(1, 21)])


def test_nurse_color(nurse, wrangler):
    programs = wrangler.learn(nurse.color_all([(0, 2), (2, 2), (2, 0), (2, 1)]))
    assert programs[0].program == Program([Header(2), Fold(1, 21)])


def test_nurse_weird(nurse2, wrangler):
    programs = wrangler.learn(nurse2)
    program = Program(
        [Header(1), Stack(1, 29, 4), Fill(1), Delete(0, EmptyCondition())]
    )
    assert programs[0].program == program


def test_icecream(icecream, wrangler):
    programs = wrangler.learn(icecream)
    transformations = programs[0].program.transformations
    assert transformations[0] == Header(1)
    assert Delete(1, EmptyCondition()) in transformations
    assert Fill(0) in transformations


def test_icecream_color(icecream, wrangler):
    icecream = icecream.color_all([(2, 1), (2, 0), (6, 1)])
    programs = wrangler.learn(icecream)
    transformations = programs[0].program.transformations
    assert transformations[0] == Header(1)
    assert Delete(1, EmptyCondition()) in transformations
    assert Fill(0) in transformations
    assert Fold(2, 5) in transformations


def test_deeltijdswerk(deeltijdswerk, wrangler):
    programs = wrangler.learn(deeltijdswerk)
    assert programs[0].program == Program([Header(2), Fold(1, 12)])


def test_nba(nba, wrangler):
    # nba = nba.color_all()
    programs = wrangler.learn(nba)
    program = Program(
        [
            Divide(1, "datatype"),
            Header(1),
            Fill(3),
            Delete(4, EmptyCondition()),
            Fill(2),
        ]
    )
    # assert programs[0].program == program


def test_nba_color(nba, wrangler):
    nba = nba.color_all([(1, 1), (1, 2)])
    programs = wrangler.learn(nba)
    program = Program(
        [
            Divide(1, "datatype"),
            Header(1),
            Fill(3),
            Delete(4, EmptyCondition()),
            Fill(2),
        ]
    )
    assert programs[0].program == program


def test_part(part, wrangler):
    part = part.color_all([(1, 0), (1, 1), (1, 2)])
    programs = wrangler.learn(part)
    assert programs[0].program == Program([Header(2), Fold(1, 8)])


def test_part_extended(part_e, wrangler):
    Optimizations.disable_style()
    part_e = part_e.color_all([(1, 0), (1, 1), (1, 2)])
    programs = wrangler.learn(part_e)
    assert programs[0].program == Program([Header(2), Fold(1, 10)])


def test_car(car, wrangler):
    programs = wrangler.learn(car)
    assert programs[0].program == Program(
        [Divide(0, "datatype"), Fill(0), Delete(1, EmptyCondition())]
    )
