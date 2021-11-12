from __future__ import annotations

from math import degrees, log, sqrt
from typing import Protocol, TypeVar

import cupy as cp
import cupyx.scipy as cusp
import cupyx.scipy.ndimage
import numpy as np

from superres.operators.partition import Partitionable

F = TypeVar("F", bound=np.floating)


class DeviceOperator(Protocol):
    def __call__(self, array: cp.ndarray) -> cp.ndarray:
        ...


class PartitionableDeviceOperator(Partitionable, DeviceOperator, Protocol):
    pass


class BspInterpolate:
    def __init__(self, axis: int, factor: float) -> None:
        self.axis = axis
        self.factor = factor

    def __call__(self, array: cp.ndarray) -> cp.ndarray:
        zoom = len(array.shape) * [1.0]
        zoom[self.axis] = self.factor

        return cusp.ndimage.zoom(cp.array(array), zoom=zoom)

    def partition_axis(self, shape: tuple[int, ...]) -> int:
        return (self.axis + 1) % len(shape)


class GaussianBlur:
    @staticmethod
    def fwhm_to_stdev(fwhm: float) -> float:
        return fwhm / sqrt(8 * log(2))

    def __init__(self, axis: int,
                 stdev: float | None = None,
                 fwhm: float | None = None) -> None:
        self.axis = axis

        if stdev is not None:
            self.stdev = stdev
        elif fwhm is not None:
            self.stdev = self.fwhm_to_stdev(fwhm)
        else:
            self.stdev = 1.0

    def __call__(self, array: cp.ndarray) -> cp.ndarray:
        return cusp.ndimage.gaussian_filter1d(
            array,
            sigma=self.stdev,
            axis=self.axis,
        )

    def partition_axis(self, shape: tuple[int, ...]) -> int:
        return (self.axis + 1) % len(shape)


class Rotate:
    def __init__(self, axes: tuple[int, int], theta: float):
        self.axes = axes
        self.theta = theta

    def __call__(self, array: cp.ndarray) -> cp.ndarray:
        return cusp.ndimage.rotate(
            array,
            axes=self.axes,
            angle=degrees(self.theta),
            mode="reflect",
        )

    def partition_axis(self, shape: tuple[int, ...]) -> int | None:
        for i in range(len(shape)):
            if i not in self.axes:
                return i
        return None


def downsample(axis: int, factor: float) -> PartitionableDeviceOperator:
    return BspInterpolate(axis=axis, factor=1.0 / factor)


def bsp_interpolate(axis: int,
                    factor: float) -> PartitionableDeviceOperator:
    return BspInterpolate(axis=axis, factor=factor)


def gaussian_blur(axis: int, stdev: float | None = None,
                  fwhm: float | None = None) -> PartitionableDeviceOperator:
    return GaussianBlur(axis=axis, stdev=stdev, fwhm=fwhm)


def rotate(axes: tuple[int, int],
           theta: float) -> PartitionableDeviceOperator:
    return Rotate(axes=axes, theta=theta)

