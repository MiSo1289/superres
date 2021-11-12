from typing import Protocol, TypeVar

import numpy as np
import numpy.typing as npt

T = TypeVar('T', bound=np.generic)


class Operator(Protocol):
    def __call__(self, array: npt.NDArray[T]) -> npt.NDArray[T]:
        ...
