"""Conditions that cells can satisfy."""
from email.mime import base
import re
from collections import Counter
from typing import List, Any, Set, Tuple
from abc import ABC, abstractmethod, abstractclassmethod
from pandas import isna
from .table import Cell


class Condition(ABC):
    """Condition."""

    @abstractmethod
    def __call__(self, value: Cell):
        pass

    @abstractclassmethod
    def generate(cls, values: List[Cell]):
        pass

    def __hash__(self) -> int:
        return hash(repr(self))


class EmptyCondition(Condition):
    """Check whether cell is empty."""

    def __call__(self, value: Cell) -> bool:
        return isna(value.value)

    def __eq__(self, o: object) -> bool:
        return isinstance(o, EmptyCondition)

    def __str__(self):
        return "EmptyCondition"

    def __repr__(self) -> str:
        return "EmptyCondition()"

    @classmethod
    def generate(cls, values: List[Cell]) -> Set[Condition]:
        na = len(values) - sum(map(bool, values))
        if na > 0 and na < len(values):
            return {cls()}
        return {}

    def __hash__(self) -> int:
        return hash(("EmptyCondition",))


class StyleCondition(Condition):
    """Check whether cell matches style."""

    def __init__(self, key: str, value: Any):
        self._key = key
        self._value = value

    def __call__(self, value: Cell) -> bool:
        if self._key not in value.style:
            return False
        else:
            return value.style[self._key] == self._value

    @classmethod
    def generate(cls, values: List[Cell]) -> Set["StyleCondition"]:
        candidates = Counter()
        total = 0
        for cell in values:
            if cell:
                for key, value in cell.style.items():
                    candidates[(key, value)] += 1
                total += 1
        return {StyleCondition(*c) for c in candidates}

    def __repr__(self) -> str:
        return "StyleCondition({}, {})".format(repr(self._key), repr(self._value))

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, StyleCondition)
            and self._key == o._key
            and self._value == o._value
        )

    def __hash__(self) -> int:
        return hash((self._key, self._value))


class DatatypeCondition(Condition):
    """Check whether cell matches style."""

    def __init__(self, value: str):
        self._value = value

    def __call__(self, value: Cell) -> bool:
        return value.datatype == self._value

    @classmethod
    def generate(cls, values: List[Cell]) -> Set["DatatypeCondition"]:
        return {DatatypeCondition(value.datatype) for value in values}

    def __repr__(self) -> str:
        return "DatatypeCondition({})".format(repr(self._value))

    def __eq__(self, o: object) -> bool:
        return isinstance(o, DatatypeCondition) and self._value == o._value

    def __hash__(self) -> int:
        return hash(repr(self))


class PatternCondition(Condition):
    """Check whether a cell matches a pattern."""

    default = {
        "word": r"^[a-zA-Z]+$",
        "word_lower": r"^[a-z]+$",
        "word_upper": r"^[A-Z]+$",
        "words": r"^[a-zA-Z]+( [a-zA-Z]+)*$",
        "words_lower": r"^[a-z]+( [a-z]+)*$",
        "words_upper": r"^[A-Z]+( [A-Z]+)*$",
        "number": r"^\d+$",
    }

    def __init__(self, pattern: str) -> None:
        self._pattern = pattern

    def __call__(self, value: Cell) -> bool:
        if self._pattern in self.default:
            return re.match(self.default[self._pattern], str(value.value)) is not None
        else:
            return re.match(self._pattern, str(value.value)) is not None

    @classmethod
    def generate(cls, values: List[Cell]) -> Set["PatternCondition"]:
        strings = {str(value.value) for value in values}
        results = list()
        for name, pattern in cls.default.items():
            if all(re.match(pattern, s) is not None for s in strings):
                results.append(PatternCondition(name))
        return results
        # print(values)
        # queue = [("", list({str(value.value) for value in values}))]
        # patterns = list()
        # while len(queue):
        #     pattern, todo = queue.pop()
        #     for element in cls.base:
        #         matches = [re.match("{}+".format(element), s) for s in todo]
        #         matches = [m.group(0) for m in matches if m]
        #         # if matches for each element
        #         if len(matches) == len(todo):
        #             left = [todo[i][len(m) :] for i, m in enumerate(matches)]
        #             pttn = pattern + element
        #             if all(len(l) == 0 for l in left):
        #                 patterns.append(pattern + "element")

    def __repr__(self) -> str:
        return "PatternCondition({})".format(repr(self._pattern))

    def __eq__(self, o: object) -> bool:
        return isinstance(o, PatternCondition) and self._pattern == o._pattern

    def __hash__(self) -> int:
        return hash(("Pattern", self._pattern))
