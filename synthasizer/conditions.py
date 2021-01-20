"""Conditions that cells can satisfy."""
from typing import List, Dict, Any
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

    @classmethod
    def generate(cls, values: List[Cell], **kwargs):
        if any(isna(v.value) for v in values):
            return [cls()]
        return []


# class StyleCondition(Condition):
#     """Check whether cell matches style."""

#     def __init__(self, key: str, value: Any):
#         self._key = key
#         self._value = value

#     def __call__(self, value: Cell):
#         pass

#     @classmethod
#     def generate(cls, values: List[Cell]):
#         if any(v.value is None for v in values):
#             return [cls()]
#         return []