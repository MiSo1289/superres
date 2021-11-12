from __future__ import annotations

from typing import Iterator

import numpy as np
from torch.utils.data import IterableDataset


class SlicesDataset(IterableDataset[tuple[np.ndarray, np.ndarray]]):
    def __init__(self, dataset: IterableDataset[tuple[np.ndarray, np.ndarray]],
                 sliced_axis: int = 0) -> None:
        self.dataset = dataset
        self.sliced_axis = sliced_axis

    def __getitem__(self, index) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for image, target in self.dataset:
            assert self.sliced_axis < len(image.shape) == len(target.shape)
            assert image.shape[self.sliced_axis] \
                   == target.shape[self.sliced_axis]

            for i in range(image.shape[self.sliced_axis]):
                indices: list[int | slice] = len(image.shape) * \
                                             [slice(None, None, None)]
                indices[self.sliced_axis] = i
                index_tuple = tuple(indices)

                yield image[index_tuple], target[index_tuple]


def slices(image: np.ndarray, axis: int) -> Iterator[np.ndarray]:
    for i in range(image.shape[axis]):
        indices: list[int | slice] = len(image.shape) * \
                                     [slice(None, None, None)]
        indices[axis] = i
        index_tuple = tuple(indices)

        yield image[index_tuple]
