"""One explicit matrix convention: A[receiver, source] means source -> receiver."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Topology:
    """A binary mask or stack; manual rooted-connectivity checks occur at inference."""
    receiver_source: np.ndarray

    def __post_init__(self):
        a = np.array(self.receiver_source, copy=True)
        if a.ndim not in (2, 3) or a.shape[-2] != a.shape[-1]:
            raise ValueError("A topology must have shape (M,M) or (count,M,M)")
        if (not np.all(np.isfinite(a)) or not np.all((a == 0) | (a == 1))
                or np.any(np.diagonal(a, axis1=-2, axis2=-1))):
            raise ValueError("Topology matrices must be binary with a zero diagonal")
        a = a.astype(bool); a.setflags(write=False)
        object.__setattr__(self, "receiver_source", a)

    @classmethod
    def from_receiver_source(cls, matrix):
        """Use the same indexing as the numerical coupling matrices."""
        return cls(matrix)

    @classmethod
    def from_source_receiver(cls, matrix):
        """Convert matrices from the existing inspector's input convention."""
        return cls(np.swapaxes(matrix, -1, -2))

    def as_source_receiver(self):
        """Return a copy suitable for the existing inspector/inference backend."""
        return np.swapaxes(self.receiver_source, -1, -2).copy()
