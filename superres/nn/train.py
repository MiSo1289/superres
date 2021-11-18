from typing import TypeVar

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


def train_edsr(images: IterableDataset[tuple[npt.NDArray[T], npt.NDArray[T]]],
               scale: int = 2,
               patch_size: int = 32,
               infer_axis: int = 0,
               rgb_range: int = 255,
               device: str = "cuda:0",
               epochs: int = 5,
               batch_size: int = 32,
               learning_rate: float = 1e-6) -> nn.Module:
    # first_image = images[0][0]
    # zero_level = np.min(first_image)
    # std = np.std(first_image)
    # max_val = np.max(first_image)


    train_data = TransformedDataset(
        SlicesDataset(images, sliced_axis=infer_axis),
        transform=Compose([
            lambda im: np.stack(3 * [im.astype(np.float32)], axis=-1),
            ToTensor(),
            RandomCrop(size=patch_size),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            Resize(size=(patch_size // scale, patch_size // scale))
        ]),
        target_transform=Compose([
            lambda im: np.stack(3 * [im.astype(np.float32)], axis=-1),
            ToTensor(),
            RandomCrop(size=(patch_size, patch_size)),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
        ]),
        same_rng=True,
    )

    model = edsr.make_model(edsr.Args(
        n_colors=3,
        scale=(scale, scale),
        rgb_range=rgb_range,
    )).to(device=device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=4e-3,
        # weight_decay=0.0,
        amsgrad=False,
    )

    dataloader = DataLoader(train_data, batch_size=batch_size)

    epoch_avg_losses: list[float] = []

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

            epoch_sum_loss += loss.item()
            num_batches += 1

        epoch_avg_loss = epoch_sum_loss / num_batches
        print(f"epoch_avg_loss={epoch_avg_loss}")

        epoch_avg_losses.append(epoch_avg_loss)

        if len(epoch_avg_losses) >= 2:
            epoch_loss_delta = epoch_avg_losses[-1] - epoch_avg_losses[-2]
            print(f"epoch_loss_delta={epoch_loss_delta}")

    print("Finished training")
    print(f"epoch_avg_losses={epoch_avg_losses}")

    return model
