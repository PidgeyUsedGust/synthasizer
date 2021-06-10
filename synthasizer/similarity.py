import re
import string
import textdistance
import itertools
from abc import ABC, abstractmethod
from typing import Callable, Optional
from .table import Cell

StringSimilarity = Callable[[str, str], float]


class CellSimilarity(ABC):
    """Generic similarity between cells."""

    @abstractmethod
    def __call__(self, a: Cell, b: Cell) -> float:
        pass


class SyntacticCellSimilarity(CellSimilarity):
    """Wrapper around string similarity."""

    def __init__(self, similarity: StringSimilarity):
        self.similarity = similarity

    def __call__(self, a: Cell, b: Cell) -> float:
        return self.similarity(str(a.value), str(b.value))


class CompressedSimilarity(StringSimilarity):
    """Compress string."""

    classes = {
        "A": string.ascii_uppercase,
        "a": string.ascii_lowercase,
        "0": string.digits,
    }

    def __init__(self, base: Optional[StringSimilarity] = None):
        self._base = base or textdistance.needleman_wunsch.normalized_similarity
        self._map = {ch: cl for cl, chs in self.classes.items() for ch in chs}

    def __call__(self, a: str, b: str) -> float:
        ca = _compress(self.classify(a.strip()))
        cb = _compress(self.classify(b.strip()))
        return self._base(ca, cb)

    def classify(self, text: str) -> str:
        """Map string to character classes."""
        return [self._map[c] if c in self._map else c for c in text]


class PatternSimilarity(CompressedSimilarity):
    """Find patterns after compression."""

    def __call__(self, a: str, b: str) -> float:
        ca = _pattern(_compress(self.classify(a.strip())))
        cb = _pattern(_compress(self.classify(b.strip())))
        return self._base(ca, cb)


def _compress(text: str) -> str:
    return "".join(x for x, _ in itertools.groupby(text))


def _pattern(text: str) -> str:
    t = re.sub(r"\s+", "", text)
    n = len(t)
    for i in range(1, n // 2 + 1):
        length, rest = divmod(n, i)
        if rest == 0:
            pattern = t[:i]
            if pattern * length == t:
                return pattern
    return text


def _dict_to_function(d, missing=1, equal=0):
    """Wrap a dictionary as a function.

    Utility function to create `sim_func` from custom dictionaries.

    """

    def _unwrap(a, b):
        if (a, b) in d:
            return d[(a, b)]
        elif (b, a) in d:
            return d[(b, a)]
        elif a == b:
            return equal
        return missing

    return _unwrap
