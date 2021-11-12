from __future__ import annotations

from typing import TypeVar

import cupy as cp
import numpy as np
import numpy.typing as npt

from superres.operators.device import PartitionableDeviceOperator

T = TypeVar('T', bound=np.generic)


class ToDevice:
    def __init__(self, op: PartitionableDeviceOperator) -> None:
        self.op = op

    def __call__(self, array: npt.NDArray[T]) -> npt.NDArray[T]:
        return self.op(cp.array(array)).get()

    def partition_axis(self, shape: tuple[int, ...]) -> int | None:
        return self.op.partition_axis(shape)
