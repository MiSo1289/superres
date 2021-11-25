from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, \
    RandomVerticalFlip, Resize, ToTensor

import superres.nn.model.edsr as edsr
from superres.data.slices import SlicesDataset
from superres.data.transformed import TransformedDataset

T = TypeVar("T", bound=np.generic)


# TODO early stopping
def train_edsr(images: IterableDataset[tuple[npt.NDArray[T], npt.NDArray[T]]],
               patch_size: int = 32,
               infer_axis: int = 0,
               max_color_level: int = 255,
               device: str = "cuda:0",
               epochs: int = 5,
               batch_size: int = 32,
               learning_rate: float = 1e-6) -> nn.Module:
    transform = Compose([
        lambda im: np.stack([im.astype(np.float32)], axis=-1),
        ToTensor(),
        RandomCrop(size=(patch_size, patch_size)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
    ])

    train_data = TransformedDataset(
        SlicesDataset(images, sliced_axis=infer_axis),
        transform=transform,
        target_transform=transform,
        same_rng=True,
    )

    model = edsr.make_model(edsr.Args(
        n_colors=1,
        max_color_level=max_color_level,
    )).to(device=device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=4e-3,
        amsgrad=False,
    )

    dataloader = DataLoader(train_data, batch_size=batch_size)

    epoch_avg_losses: list[float] = []
    best_epoch: int | None = None
    best_loss: float | None = None
    best_model_state: dict[str, Any] | None = None

    print(f"epochs={epochs} batch_size={batch_size}")
    print(f"learning_rate={learning_rate}")

    for epoch in range(epochs):
        epoch_sum_loss = 0.0
        num_batches = 0

        for batch, (image, target) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(image.to(device=device))
            loss = loss_fn(pred, target.to(device=device))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"epoch={epoch} batch={batch} loss={loss}")

            epoch_sum_loss += float(loss)
            num_batches += 1

        epoch_avg_loss = epoch_sum_loss / num_batches
        print(f"epoch_avg_loss={epoch_avg_loss}")

        epoch_avg_losses.append(epoch_avg_loss)

        if len(epoch_avg_losses) >= 2:
            epoch_loss_delta = epoch_avg_losses[-1] - epoch_avg_losses[-2]
            print(f"epoch_loss_delta={epoch_loss_delta}")

        if best_loss is None or epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            best_epoch = epoch
            best_model_state = model.state_dict()

    print("Finished training")
    print(f"epoch_avg_losses={epoch_avg_losses}")

    print(f"Using best model state from epoch={best_epoch} "
          f"with loss={best_loss}")
    model.load_state_dict(best_model_state)

    return model
