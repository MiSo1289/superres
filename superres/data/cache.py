from __future__ import annotations

from typing import Iterator, TypeVar

import numpy as np
from torch.utils.data import IterableDataset, Dataset

T = TypeVar("T")


class CacheDataset(IterableDataset[T]):
    def __init__(self, dataset: IterableDataset[T]) -> None:
        self.dataset = dataset
        self.cache: list[T] = []
        self.current: Iterator[T] | None = iter(self.dataset)

    def __getitem__(self, index) -> T:
        while self.current and index >= len(self.cache):
            try:
                assert self.current is not None
                self.cache.append(next(self.current))
            except StopIteration:
                self.current = None

        return self.cache[index]

    def __iter__(self) -> Iterator[T]:
        yield from self.cache

        while self.current:
            try:
                assert self.current is not None
                new_entry = next(self.current)
                self.cache.append(new_entry)

                yield new_entry
            except StopIteration:
                self.current = None


class PreloadDataset(Dataset[T]):
    def __init__(self, dataset: IterableDataset[T]) -> None:
        self.data: list[T] = [*dataset]

    def __getitem__(self, index) -> T:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)
