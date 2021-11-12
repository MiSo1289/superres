from __future__ import annotations

import numpy as np
from napari import view_image
from torch.utils.data import IterableDataset


def view_dataset(dataset: IterableDataset[tuple[np.ndarray, str]]) -> None:
    for image, label in dataset:
        viewer = view_image(image, title=label)
        viewer.show(block=True)
