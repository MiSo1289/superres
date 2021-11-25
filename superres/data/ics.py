from __future__ import annotations

import os
from typing import Callable, Iterator, TypeVar, Union

import numpy as np
from pyics import read_ics, write_ics
from torch.utils.data import IterableDataset

from superres.utils.op_indicator import op_indicator

Path = Union[str, bytes, os.PathLike[str], os.PathLike[bytes]]

T = TypeVar("T")
U = TypeVar("U")


def verbose_read_ics(image_path: Path) -> np.ndimage:
    with op_indicator(f"Reading image '{image_path}'"):
        return read_ics(image_path)


def verbose_write_ics(image_path: Path, image: np.ndimage) -> None:
    with op_indicator(f"Saving image '{image_path}'"):
        write_ics(image_path, image)


def save_dataset_as_ics(dataset: IterableDataset[tuple[np.ndarray, str]],
                        out_folder: Path):
    for image, name in dataset:
        verbose_write_ics(os.path.join(out_folder, f"{name}.ics"), image)


class IcsDataset(IterableDataset[tuple[T, U]]):
    def __init__(self,
                 ics_files: list[Path],
                 transform: Callable[[np.ndarray], T] | None = None,
                 target_transform: Callable[[str], U] | None = None) -> None:
        self.ics_files = ics_files
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.ics_files)

    def __getitem__(self, idx: int) -> tuple[T, U]:
        image_path = self.ics_files[idx]

        image: np.ndarray | T = verbose_read_ics(image_path)
        target: str | U = os.path.splitext(os.path.basename(image_path))[0]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    def __iter__(self) -> Iterator[tuple[T, U]]:
        for i in range(len(self)):
            yield self[i]


class LoadIcsImagesFromDir:
    def __init__(self, img_dir: Path) -> None:
        self.img_dir = img_dir

    def __call__(self, name: str) -> np.ndarray:
        image_path = os.path.join(self.img_dir, f"{name}.ics")
        return verbose_read_ics(image_path)
