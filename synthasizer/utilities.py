from typing import Any, Iterable, List, Set, TypeVar
import typing

T = TypeVar("T")


class nzs(set):
    """Nonzero set."""

    def __init__(self, s=()) -> None:
        super().__init__(set(s) - {0})

    def add(self, element: Any) -> None:
        """Add element to the set."""
        if element != 0:
            super().add(element)

    def update(self, s: Iterable[Any]) -> None:
        """Add many elements to the set."""
        return super().update(set(s) - {0})


def transpose(l: List[List[T]]) -> List[List[T]]:
    """Transpose list of lists."""
    return list(map(list, zip(*l)))


def duplicates(l: Iterable[T]) -> List[T]:
    """Get duplicated elements.

    Returns:
        All duplicate elements from `l`.

    """
    seen = dict()
    duplicates = list()
    for x in l:
        if x in seen:
            duplicates.append(x)
        seen[x] = 1
    return duplicates


def infer_types(types: Iterable[str]) -> str:
    """Infer type from a list of individual types.

    Args:
        types: List of individual types.

    """
    unique = sorted(set(types) - {"empty"})
    # only one type
    if len(unique) == 0:
        return "empty"
    if len(unique) == 1:
        return unique[0]
    return "mixed-{}".format("-".join(unique))


def nothing():
    return None
