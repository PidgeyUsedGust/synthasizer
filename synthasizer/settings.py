from synthasizer.conditions import EmptyCondition, StyleCondition
from synthasizer.transformation import Divide, Delete, Fold


class Optimizations:
    """Some optimisations."""

    @staticmethod
    def disable_style() -> None:
        Divide.style = False
        Delete.conditions = [EmptyCondition]

    @staticmethod
    def enable_style() -> None:
        Divide.style = True
        Delete.conditions = [EmptyCondition, StyleCondition]

    @staticmethod
    def disable_smart() -> None:
        Fold.smart = False

    def enable_smart() -> None:
        Fold.smart = True
