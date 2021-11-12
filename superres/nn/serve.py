from typing import TypeVar

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms.functional import to_tensor

import superres.nn.model.edsr as edsr
from superres.data.slices import slices

T = TypeVar("T", bound=np.generic)


def load_edsr(model_path: str, scale: int = 2,
              rgb_range: int = 255) -> nn.Module:
    model = edsr.make_model(edsr.Args(
        n_colors=3,
        scale=(scale, scale),
        rgb_range=rgb_range,
    )).to("cuda")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def infer(model: nn.Module, image: np.ndarray, scale: int = 2,
          train_axis: int = 2) -> np.ndarray:
    inferred_slices = []
    for sl in slices(image, axis=train_axis):
        slice_tensor = to_tensor(
            np.stack(3 * [sl.astype(np.float32)], axis=-1))
        slice_tensor = torch.stack([slice_tensor]).to("cuda")

        inferred_slices.append(
            model(slice_tensor)[0, 0, :, 0:-1:scale].detach().cpu().numpy())
    return np.stack(inferred_slices)
