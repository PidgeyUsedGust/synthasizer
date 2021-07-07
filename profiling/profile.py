import time
import openpyxl
from pyinstrument import Profiler
from synthasizer.table import detect, Table
from synthasizer.wrangle import Wrangler, Language
from synthasizer.transformation import Delete, Divide, Header, Fill, Fold, Stack
from synthasizer.heuristics import (
    TypeColumnHeuristic,
    WeightedHeuristic,
    EmptyHeuristic,
    AggregatedHeuristic,
    ColorRowHeuristic,
)
from synthasizer.strategy import Astar


def get_nurse() -> Table:
    return detect(openpyxl.load_workbook("data/nurse.xlsx")["Sheet1"])[0]


def get_nba() -> Table:
    return detect(openpyxl.load_workbook("data/nba.xlsx")["Sheet1"])[0]


def profile_nurse_colored():
    nurse = get_nurse().color_all([(0, 2), (1, 1), (2, 0), (2, 2)])
    language = Language([Delete, Divide, Header, Fill, Fold, Stack])
    heuristic = WeightedHeuristic(
        [
            EmptyHeuristic(),
            AggregatedHeuristic(TypeColumnHeuristic()),
            ColorRowHeuristic(),
        ],
        weights=[1.0, 1.0, 0.1],
    )
    wrangler = Wrangler(language, heuristic)
    profiler = Profiler()
    profiler.start()
    programs = wrangler.learn(nurse)
    profiler.stop()
    profiler.open_in_browser()


def profile_nba():
    nba = get_nba()
    # pick a set of transformations
    language = Language([Delete, Divide, Header, Fill, Stack])
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
    wrangler = Wrangler(language, heuristic, max_depth=6, strategy=Astar())
    profiler = Profiler()
    profiler.start()
    programs = wrangler.learn(nba)
    profiler.stop()
    profiler.open_in_browser()
    print(programs)


if __name__ == "__main__":
    profile_nurse_colored()
    profile_nba()
