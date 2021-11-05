from __future__ import annotations

from math import sqrt, log
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import scipy.interpolate as sci
import scipy.signal as scs
import scipy.ndimage as scndi

T = TypeVar("T", bound=np.floating)


class BspInterpolate:
    def __init__(self, dim: int, num_samples: int) -> None:
        self.dim = dim
        self.num_samples = num_samples

    def __call__(self, array: npt.NDArray[T]) -> npt.NDArray[T]:
        num_src_samples = array.shape[self.dim]

        if self.num_samples == num_src_samples:
            # Nothing to do
            return array

        result = np.zeros(
            array.shape[:self.dim]
            + (self.num_samples,)
            + array.shape[self.dim + 1:],
            dtype=array.dtype,
        )

        src_samples = np.linspace(0, 1, num_src_samples)
        samples = np.linspace(0, 1, self.num_samples)

        for index in np.ndindex(
                array.shape[:self.dim] + array.shape[self.dim + 1:]):
            index_slice = index[:self.dim] + (slice(None),) + index[self.dim:]

            src_values = array[index_slice]

            spline = sci.splrep(src_samples, src_values)
            interpolated_values = sci.splev(samples, spline)

            result[index_slice] = interpolated_values

        return result


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
        sigma = [0.0] * len(array.shape)
        sigma[self.dim] = self.stdev
        return scndi.gaussian_filter1d(array, sigma=self.stdev, dim=self.dim)


class DownSample:
    def __init__(self, dim: int, factor: int):
        pass
