from typing import Any

import numpy as np
import torchvision.transforms.functional as torchvision_fn

import superres.operators.device as dev_ops
from superres.operators.partition import Partitioned
from superres.operators.to_device import ToDevice
from superres.operators.operator import Operator


def partitioned_to_device_op_factory(device_op_factory: Any) -> Any:
    def op_factory(partitions: int = 1, **kwargs: Any) -> Operator:
        return Partitioned(
            ToDevice(device_op_factory(**kwargs)),
            partitions=partitions,
        )

    return op_factory


downsample = partitioned_to_device_op_factory(dev_ops.downsample)
bsp_interpolate = partitioned_to_device_op_factory(dev_ops.bsp_interpolate)
gaussian_blur = partitioned_to_device_op_factory(dev_ops.gaussian_blur)
rotate = partitioned_to_device_op_factory(dev_ops.rotate)


class RotateAxes:
    def __init__(self, left: bool = True) -> None:
        self.left = left

    def __call__(self, array: np.ndarray) -> np.ndarray:
        if self.left:
            new_axes = tuple(range(1, len(array.shape))) + (0,)
        else:
            new_axes = (2,) + tuple(range(len(array.shape) - 1))

        return array.transpose(new_axes)


def left_rotate_axes() -> Operator:
    return RotateAxes(left=True)


def right_rotate_axes() -> Operator:
    return RotateAxes(left=False)
