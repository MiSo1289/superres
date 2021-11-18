from typing import TypeVar

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms.functional import to_tensor

import superres.operators.ops as ops
import superres.nn.model.edsr as edsr
from superres.data.slices import slices

T = TypeVar("T", bound=np.generic)


def load_edsr(model_path: str, scale: int = 2,
              rgb_range: int = 255, device: str = "cuda:0") -> nn.Module:
    model = edsr.make_model(edsr.Args(
        n_colors=3,
        scale=(scale, scale),
        rgb_range=rgb_range,
    )).to(device=device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def infer(model: nn.Module, image: np.ndarray, scale: int = 2,
          infer_axis: int = 0, train_axis: int = 2,
          device: str = "cuda:0") -> np.ndarray:
    # The model expects lower-res input on the infer axis, but our data
    # is BSP interpolated
    image_downsampled = ops.downsample(axis=infer_axis, factor=scale)(image)

    inferred_slices = []
    third_axis = (set(range(len(image.shape))) - {infer_axis, train_axis}).pop()
    for sl in slices(image_downsampled, axis=third_axis):
        slice_tensor = to_tensor(
            np.stack(3 * [sl.astype(np.float32)], axis=-1))
        slice_tensor = torch.stack([slice_tensor]).to(device=device)

        inference = model(slice_tensor)[0, 0, :, :].detach().cpu().numpy()
        # The model increases resolution on both axes, so downsample
        # the non-inferred axis
        train_axis_slice = train_axis
        if train_axis_slice > third_axis:
            # Adjust for removal of the third axis
            train_axis_slice -= 1
        inference = ops.downsample(axis=train_axis_slice, factor=scale)(inference)

        inferred_slices.append(inference)
    return np.stack(inferred_slices, axis=third_axis)
