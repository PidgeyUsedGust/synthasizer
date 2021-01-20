"""Computing reconstruction error."""


class ReconstructionError:
    """Compute reconstruction error."""

    def __init__(self):
        self._removed = set()

    def __call__(self, before: Table, after: Table) -> float:
        if before.width == after.width and before.height == after.height:
            return 0
