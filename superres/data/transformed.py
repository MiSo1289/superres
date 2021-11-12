from __future__ import annotations

from typing import Callable, Iterator, TypeVar, Any

import torch
from torch.utils.data import IterableDataset

T = TypeVar("T")
U = TypeVar("U")
T2 = TypeVar("T2")
U2 = TypeVar("U2")


# Dataset transformation that can use same RNG state for both items and
# targets, because PyTorch is very well designed and there is no easier way
# to do this https://github.com/pytorch/vision/issues/9
class TransformedDataset(IterableDataset[tuple[T, U]]):
    def __init__(self, source: IterableDataset[tuple[T2, U2]],
                 transform: Callable[[T2], T],
                 target_transform: Callable[[U2], U] | None = None,
                 same_rng: bool | None = None) -> None:
        self.source = source
        self.transform = transform

        if target_transform:
            self.target_transform = target_transform
        else:
            self.target_transform = transform

        if same_rng is not None:
            self.same_rng = same_rng
        else:
            self.same_rng = self.target_transform is None

    def __getitem__(self, index: int) -> tuple[T, U]:
        item, target = self.source[index]
        return self._transform(item, target)

    def __iter__(self) -> Iterator[tuple[T, U]]:
        for item, target in self.source:
            yield self._transform(item, target)

    def _transform(self, item: Any, target: Any) -> tuple[T, U]:
        rng_state: torch.Tensor | None = None
        if self.same_rng:
            rng_state = torch.get_rng_state()
        transformed_item = self.transform(item)

        if self.same_rng:
            assert rng_state is not None
            torch.set_rng_state(rng_state)
        transformed_target = self.target_transform(target)

        return transformed_item, transformed_target
