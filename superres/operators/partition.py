from __future__ import annotations

from typing import Protocol, TypeVar

import numpy as np
import numpy.typing as npt

from superres.operators.operator import Operator

T = TypeVar('T', bound=np.generic)


class Partitionable(Protocol):
    def partition_axis(self, shape: tuple[int, ...]) -> int | None:
        ...


class PartitionableOperator(Partitionable, Operator, Protocol):
    pass


class Partitioned:
    def __init__(self, op: PartitionableOperator, partitions: int) -> None:
        self.op = op
        self.partitions = partitions

    def __call__(self, array: npt.NDArray[T]) -> npt.NDArray[T]:
        if (part_axis := self.op.partition_axis(array.shape)) is None:
            raise RuntimeError("Cannot partition input")

        array_parts = np.array_split(array, self.partitions, axis=part_axis)
        result_parts: list[npt.NDArray[T]] = []

        for array_part in array_parts:
            result_parts.append(self.op(array_part))

        return np.concatenate(result_parts, axis=part_axis)
