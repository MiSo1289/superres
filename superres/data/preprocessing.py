from __future__ import annotations

from math import sqrt, log
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import scipy as sp
import scipy.ndimage

T = TypeVar("T", bound=np.floating)


class UpSample:
    def __init__(self, dim: int, factor: float) -> None:
        self.dim = dim
        self.factor = factor

    def __call__(self, array: npt.NDArray[T]) -> npt.NDArray[T]:
        zoom = len(array.shape) * [1.0]
        zoom[self.dim] = self.factor

        return sp.ndimage.zoom(array, zoom=zoom)


class DownSample:
    def __init__(self, dim: int, factor: float) -> None:
        self.dim = dim
        self.factor = factor

    def __call__(self, array: npt.NDArray[T]) -> npt.NDArray[T]:
        zoom = len(array.shape) * [1.0]
        zoom[self.dim] = 1.0 / self.factor

        return sp.ndimage.zoom(array, zoom=zoom)


def gaussian_stdev_to_fwhm(stdev: float) -> float:
    return stdev * sqrt(8 * log(2))


def gaussian_fwhm_to_stdev(fwhm: float) -> float:
    return fwhm / sqrt(8 * log(2))


class GaussianBlur:
    def __init__(self, dim: int, stdev: float | None,
                 fwhm: float | None) -> None:
        self.dim = dim

        if stdev is not None:
            self.stdev = stdev
        elif fwhm is not None:
            self.stdev = gaussian_fwhm_to_stdev(fwhm)
        else:
            self.stdev = 1.0

    def __call__(self, array: npt.NDArray[T]) -> npt.NDArray[T]:
        return sp.ndimage.gaussian_filter1d(
            array, sigma=self.stdev, dim=self.dim)
