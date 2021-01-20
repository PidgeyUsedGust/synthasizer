import pandas as pd
import numpy as np
import synthasizer
from pathlib import Path


def test_delete():
    from synthasizer.transformations import Delete

    df = pd.read_csv(Path(__file__).parent / ".." / "data" / "sales.csv")
    arguments = Delete.arguments(df)
    argument = arguments[1]
    delete = Delete(*argument)
    df2 = delete(df)


def test_ffill():
    from synthasizer.transformations import ForwardFill

    df = pd.read_csv(Path(__file__).parent / ".." / "data" / "sales.csv")
    arguments = ForwardFill.arguments(df)

    ff = ForwardFill(*arguments[0])
    df2 = ff(df)


def test_fold():
    from synthasizer.transformations import Fold

    df = pd.read_csv(Path(__file__).parent / ".." / "data" / "sales.csv")
    f = Fold(2, 4)
    df2 = f(df)

    print(df)
    print(Fold.arguments(df))

    assert df2.shape == (27, 6)


if __name__ == "__main__":

    test_delete()
    test_ffill()
    test_fold()
