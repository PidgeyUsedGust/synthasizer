"""Support for patterns."""
import inspect
from typing import List, Type
from dataclasses import dataclass
from synthasizer.transformation import *


def unpack(transformation: Transformation) -> Tuple[Any, ...]:
    """Get arguments used to initialise transformation.

    Returns:
        A tuple `t` such that `Transformation(*t) == transformation`.

    """
    names = inspect.signature(transformation.__init__).parameters
    values = tuple(getattr(transformation, "_{}".format(a)) for a in names)
    return values


def n_parameters(transformation: Type[Transformation]) -> int:
    """Number of arguments of a transformation."""
    return len(inspect.signature(transformation.__init__).parameters) - 1


class Variable:
    """Variable argument."""

    def __init__(self) -> None:
        self.value: Any = None

    def __eq__(self, o: Any) -> bool:
        if self.value == o:
            return True
        else:
            if self.value is None:
                self.value = o
                return True
            else:
                self.value = None
                return False


@dataclass
class Element:
    """One element of a pattern."""

    def __init__(
        self, transformation: Type[Transformation], arguments: Tuple[Any, ...] = None
    ) -> None:
        self.transformation = transformation
        self.arguments = arguments
        if self.arguments is None:
            self.arguments = tuple(None for _ in range(n_parameters(transformation)))

    def matches(self, transformation: Transformation) -> bool:
        """Test if the arguments match."""
        if transformation.__class__ != self.transformation:
            return False
        arguments = unpack(transformation)
        if len(arguments) != len(self.arguments):
            return False
        for i, argument in enumerate(self.arguments):
            if argument is not None and arguments[i] != argument:
                return False
        return True

    @classmethod
    def from_transformation(cls, transformation: Transformation) -> "Element":
        """Create from transformation."""
        return Element(transformation.__class__, unpack(transformation))


class Pattern:
    """A pattern of transformations."""

    def __init__(self, elements: List[Element], beginning: bool = False) -> None:
        """

        Args;
            beginning: If True, only match this pattern
                at the beginning of a program.

        """
        self._elements = elements
        self._beginning = beginning

    def match(self, program: List[Transformation]):
        """Test if this pattern matches a list of transformations."""
        if self._beginning:
            candidates = program[: len(program)]
        else:
            candidates = [
                program[i : i + len(self._elements)]
                for i in range(len(program) - len(self._elements) + 1)
            ]
        return any(self.match_end(candidate) for candidate in candidates)

    def match_end(self, program: List[Transformation]) -> bool:
        """Matches end of the program."""
        if len(program) < len(self._elements) or (
            self._beginning and len(program) != len(self._arguments)
        ):
            return False
        for i, element in enumerate(self._elements[::-1]):
            if not element.matches(program[-(i + 1)]):
                return False
        return True
