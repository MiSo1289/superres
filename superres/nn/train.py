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
               patch_size: int = 32, train_axis: int = 2,
               infer_axis: int = 0,
               rgb_range: int = 255) -> nn.Module:
    first_image = images[0][0]
    # zero_level = np.min(first_image)
    # std = np.std(first_image)
    # max_val = np.max(first_image)


    train_data = TransformedDataset(
        SlicesDataset(images, sliced_axis=infer_axis),
        transform=Compose([
            lambda im: np.stack(3 * [im.astype(np.float32)], axis=-1),
            ToTensor(),
            #Normalize(mean=zero_level, std=std),
            RandomCrop(size=patch_size),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            Resize(size=(patch_size // scale, patch_size // scale))
        ]),
        target_transform=Compose([
            lambda im: np.stack(3 * [im.astype(np.float32)], axis=-1),
            ToTensor(),
            #Normalize(mean=zero_level, std=std),
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
    )).to("cuda")

    learning_rate = 1e-6
    batch_size = 32
    epochs = 1

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    dataloader = DataLoader(train_data, batch_size=batch_size)
    for epoch in range(epochs):
        for batch, (x, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(x.to("cuda"))
            loss = loss_fn(pred, y.to("cuda"))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"epoch: {epoch} batch: {batch} loss: {loss}")

    return model
