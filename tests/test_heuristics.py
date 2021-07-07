from synthasizer import table
from synthasizer import heuristics
from synthasizer.error import ContentReconstructionError
from synthasizer.conditions import EmptyCondition, StyleCondition
from tests.test_table import get_nurse
from synthasizer.wrangle import Program
from synthasizer.transformation import Delete, Stack, Fill, Header, Fold, Divide
from synthasizer.heuristics import (
    ColorRowHeuristic,
    EmptyHeuristic,
    AggregatedHeuristic,
    TypeColumnHeuristic,
    ValueColumnHeuristic,
    WeightedHeuristic,
)
from test_table import get_icecream, get_nba


def test_colorrow():
    table = get_icecream().color_all([(0, 1), (1, 1), (2, 1), (2, 0)])
    header = Header(1)(table)
    after = Fold(2, 5)(header)
    heuristic = ColorRowHeuristic()
    assert heuristic(table) == 0.75
    assert heuristic(header) == 0.75
    assert heuristic(after) == 1.0


def test_typecolumn():
    table = get_icecream().color_all([(0, 1), (1, 1), (2, 1), (2, 0)])
    heuristic = TypeColumnHeuristic()
    # print(table)
    # print(heuristic(table))


def test_empty():
    table = get_nurse()
    # print(EmptyHeuristic()(table))


def test_nurse():
    program = Program(
        [Header(1), Stack(1, 29, 4), Fill(1), Delete(3, EmptyCondition())]
    )
    nurse = get_nurse()
    after = program(nurse)
    heuristic = WeightedHeuristic(
        [
            EmptyHeuristic(),
            AggregatedHeuristic(ValueColumnHeuristic()),
            ColorRowHeuristic(),
        ]
    )
    assert heuristic(nurse) < 0.9
    assert heuristic(after) == 1.0


def test_nurse_color():
    nurse = get_nurse().color_all([(0, 2), (1, 1), (2, 0), (2, 2)])
    program = Program(
        [Header(1), Stack(1, 29, 4), Fill(1), Delete(3, EmptyCondition())]
    )
    inter = program(nurse)
    heuristic = WeightedHeuristic(
        [
            EmptyHeuristic(),
            AggregatedHeuristic(TypeColumnHeuristic()),
            ColorRowHeuristic(),
        ],
        weights=[1.0, 1.0, 0.1],
    )
    # print(heuristic(nurse))
    # print(heuristic(inter))


def test_full_nba():
    nba = get_nba()

    error = ContentReconstructionError()
    error.initialise(nba)

    program = Program(
        [
            Header(1),
            Divide(1, "bold"),
            Fill(0),
            Fill(1),
            Fill(2),
            Delete(4, EmptyCondition()),
        ]
    )

    heuristic = WeightedHeuristic(
        [
            EmptyHeuristic(),
            AggregatedHeuristic(TypeColumnHeuristic()),
            ColorRowHeuristic(),
        ],
        weights=[1.0, 1.0, 0.1],
    )
    after = program(nba)
    print(heuristic(after))


if __name__ == "__main__":
    test_empty()
    test_colorrow()
    test_typecolumn()
    # test_nurse()
    test_nurse_color()
    test_full_nba()
