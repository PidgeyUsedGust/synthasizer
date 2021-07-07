from synthasizer.strategy import Astar
from tests.test_table import get_icecream, get_nba, get_nurse
import time
from synthasizer.wrangle import Wrangler, Language
from synthasizer.transformation import (
    Delete,
    Divide,
    Header,
    Fill,
    Fold,
    Stack,
)
from synthasizer.heuristics import (
    TypeColumnHeuristic,
    WeightedHeuristic,
    EmptyHeuristic,
    AggregatedHeuristic,
    ColorRowHeuristic,
)


def test_icecream():
    ice = get_icecream()
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
    wrangler = Wrangler(language, heuristic, verbose=False)
    programs = wrangler.learn(ice)
    print(programs)


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
    wrangler = Wrangler(language, heuristic, verbose=False)
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


def test_nba():
    nba = get_nba()
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
    wrangler = Wrangler(
        language, heuristic, max_depth=6, strategy=Astar(), verbose=False
    )
    programs = wrangler.learn(nba)
    print(programs)

    # # pick a set of transformations
    # language = Language([Delete, Divide, Header, Fill, Fold, Stack])

    # # initialise wrangler
    # wrangler = Wrangler(
    #     language,
    #     heuristic,
    #     max_depth=6,
    #     strategy=Astar(),
    #     max_iterations=1,
    #     verbose=True,
    # )
    # programs = wrangler.learn(nba)
    # print(programs)


if __name__ == "__main__":
    test_icecream()
    test_nurse()
    test_nba()
