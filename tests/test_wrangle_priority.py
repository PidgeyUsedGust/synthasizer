from synthasizer.pattern import Element, Pattern
from synthasizer.strategy import Astar, Prioritizer
from synthasizer.transformation import *
from synthasizer.heuristics import *
from synthasizer.wrangle import Wrangler
from test_table import get_nurse


def test_nurse_header():
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
    strategy = Prioritizer(Astar(), [Pattern([Element(Header)])])
    # initialise wrangler
    wrangler = Wrangler(language, heuristic, strategy=strategy)
    programs = wrangler.learn(nurse)
    print(programs)


if __name__ == "__main__":
    test_nurse_header()
