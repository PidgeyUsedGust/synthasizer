import time
from synthasizer.wrangle import Wrangler, Language
from synthasizer.transformation import Delete, Divide, Header, Fill, Fold, Stack
from synthasizer.heuristics import (
    TypeColumnHeuristic,
    WeightedHeuristic,
    EmptyHeuristic,
    AggregatedHeuristic,
    ValueColumnHeuristic,
    ColorRowHeuristic,
)
from test_table import get_nurse


def test_nurse():
    nurse = get_nurse()
    # pick a set of transformations
    language = Language([Delete, Divide, Header, Fill, Fold, Stack])
    # initialise heuristic
    heuristic = WeightedHeuristic(
        [
            EmptyHeuristic(),
            AggregatedHeuristic(TypeColumnHeuristic()),
            ColorRowHeuristic(),
        ],
        weights=[1.0, 1.0, 0.1],
    )
    # initialise wrangler
    wrangler = Wrangler(language, heuristic)
    programs = wrangler.learn(nurse)
    print(programs)


def test_nurse_colored():
    nurse = get_nurse().color_all([(0, 2), (1, 1), (2, 0), (2, 2)])
    # print(nurse.color_df)
    # pick a set of transformations
    language = Language([Delete, Divide, Header, Fill, Fold, Stack])
    # initialise heuristic
    heuristic = WeightedHeuristic(
        [
            EmptyHeuristic(),
            AggregatedHeuristic(TypeColumnHeuristic()),
            ColorRowHeuristic(),
        ],
        weights=[1.0, 1.0, 0.1],
    )
    # initialise wrangler
    wrangler = Wrangler(language, heuristic)
    start = time.time()
    programs = wrangler.learn(nurse)
    end = time.time()


if __name__ == "__main__":
    test_nurse()
