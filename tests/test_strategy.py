from numpy import exp
from synthasizer.wrangle import Junction, State, Program, VariedBeam
from synthasizer.transformations import *


def test_variedbeam():
    varied = VariedBeam(kinds=3, width=1)
    varied.push(
        [
            State(0.9, Program([Fill(1)]), None),
            State(0.9, Program([Fill(2)]), None),
            State(0.8, Program([Fold(0, 1)]), None),
        ]
    )
    assert [state.kind for state in varied._queue] == ["Fold", "Fill"]


def test_junction():
    expanded = list()
    junction = Junction(3)
    junction.push([0, 1, 2])
    expanded.append(junction.pop())
    junction.push([20, 21, 23])
    expanded.append(junction.pop())
    junction.push([230, 231, 232])
    expanded.append(junction.pop())
    expanded.append(junction.pop())
    junction.push([10, 11, 12])
    expanded.append(junction.pop())
    junction.push([120, 121, 122])
    expanded.append(junction.pop())
    assert expanded == [2, 23, 232, 1, 12, 122]
    assert junction.pop() == 0


if __name__ == "__main__":
    # test_variedbeam()
    test_junction()
