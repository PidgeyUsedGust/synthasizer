"""Conditions that cells can satisfy."""
from collections import Counter
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod, abstractclassmethod
from pandas import Series, isna
from .table import Cell


class Condition(ABC):
    """Condition."""

    @abstractmethod
    def __call__(self, value: Cell):
        pass

    @abstractclassmethod
    def generate(cls, values: List[Cell]):
        pass


class EmptyCondition(Condition):
    """Check whether cell is empty."""

    def __call__(self, value: Cell):
        return isna(value.value)

    def __str__(self):
        return "EmptyCondition"
    
    def __repr__(self) -> str:
        return "EmptyCondition()"

    @classmethod
    def generate(cls, values: List[Cell]):
        if any(isna(v.value) for v in values):
            return [cls()]
        return []


class StyleCondition(Condition):
    """Check whether cell matches style."""

    def __init__(self, key: str, value: Any):
        self._key = key
        self._value = value

    def __call__(self, value: Cell):
        if self._key not in value.style:
            return False
        else:
            return value.style[self._key] == self._value

    @classmethod
    def generate(cls, values: List[Cell]) -> List[Tuple[str, Any]]:
        # generate candidates
        candidates = Counter()
        for cell in values:
            for key, value in cell.style.items():
                candidates[(key, value)] += 1
        # generate properties
        styles = list()
        for property, count in candidates.items():
            if count < len(values):
                styles.append(property)
        return styles

    def __repr__(self) -> str:
        return "StyleCondition({}, {})".format(repr(self._key), repr(self._value))