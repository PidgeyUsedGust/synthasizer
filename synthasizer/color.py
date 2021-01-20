import pandas as pd
from typing import Tuple


def wrap_color(value: str, color: int) -> str:
    return "{:02d}{}".format(color, value)


def unwrap_color(value: str) -> Tuple[str, int]:
    return value[3:], int(value[:2])


class ColoredDataFrame:
    """A pandas DataFrame that supports cells being colored."""

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._colored = dict()

    def __getitem__(self, key):
        value = self._df[key]
        if isinstance(value, pd.Series):
            pass


    def __setitem__(self, key, value):
        self._df[key] = value
    
    def __str__(self):
        return str(self._df)

    def get_color(self, key) -> str:
        return unwrap_color(self[key])[1]
    
    def set_color(self, key, color):
        self[key] = wrap_color(key, color)