from synthasizer.conditions import PatternCondition


def test_pattern_candidates(car):

    print(PatternCondition.generate(car[0][[0, 4]]))
    print(PatternCondition.generate(car[0][[1, 2, 3, 5]]))


def test_pattern(car):
    pattern = PatternCondition("words")
    matches = [pattern(cell) for cell in car[0]]
    print(matches)
