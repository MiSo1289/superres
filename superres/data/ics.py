from __future__ import annotations

import os
from typing import Callable, Sized, TypeVar, Union

import numpy as np
from pyics import read_ics, write_ics
from torch.utils.data import Dataset

Path = Union[str, bytes, os.PathLike[str], os.PathLike[bytes]]

T = TypeVar("T")
U = TypeVar("U")


class IcsDataset(Dataset[tuple[T, U]]):
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
        image: np.ndarray | T = read_ics(image_path)
        target: str | U = os.path.splitext(os.path.basename(image_path))[0]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target


def save_dataset_as_ics(dataset: Dataset[tuple[np.ndarray, str]] + Sized,
                        out_folder: Path):
    for i in range(len(dataset)):
        image, label = dataset[i]
        write_ics(os.path.join(out_folder, f"{label}.ics"), image)
