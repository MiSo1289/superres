import math
from typing import TypeVar

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms.functional import to_tensor

import superres.nn.model.edsr as edsr
from superres.data.slices import slices
from superres.utils.op_indicator import op_indicator

T = TypeVar("T", bound=np.generic)


def load_edsr(model_path: str,
              max_color_level: int = 255, device: str = "cuda:0") -> nn.Module:
    model = edsr.make_model(edsr.Args(
        n_colors=1,
        max_color_level=max_color_level,
    )).to(device=device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def infer(model: nn.Module, image: np.ndarray, chunk_size: int = 1,
          infer_axis: int = 0, train_axis: int = 2,
          device: str = "cuda:0") -> np.ndarray:
    # The model expects lower-res input on the infer axis, but our data
    # is BSP interpolated
    # image_downsampled = ops.downsample(axis=infer_axis, factor=scale)(image)
    #
    # inferred_slices = []
    # third_axis = (set(range(len(image.shape))) - {infer_axis, train_axis}).pop()
    # for sl in slices(image_downsampled, axis=third_axis):
    #     slice_tensor = to_tensor(
    #         np.stack(3 * [sl.astype(np.float32)], axis=-1))
    #     slice_tensor = torch.stack([slice_tensor]).to(device=device)
    #
    #     inference = model(slice_tensor)[0, 0, :, :].detach().cpu().numpy()
    #     # The model increases resolution on both axes, so downsample
    #     # the non-inferred axis
    #     train_axis_slice = train_axis
    #     if train_axis_slice > third_axis:
    #         # Adjust for removal of the third axis
    #         train_axis_slice -= 1
    #     inference = ops.downsample(axis=train_axis_slice, factor=scale)(inference)
    #
    #     inferred_slices.append(inference)
    # return np.stack(inferred_slices, axis=third_axis)

    # inferred_slices = []
    # third_axis = (set(range(len(image.shape))) - {infer_axis, train_axis}).pop()
    # for sl in slices(image, axis=third_axis):
    #     slice_tensor = to_tensor(
    #         np.stack(1 * [sl.astype(np.float32)], axis=-1))
    #     slice_tensor = torch.stack([slice_tensor]).to(device=device)
    #
    #     inference = model(slice_tensor)[0, 0, :, :].detach().cpu().numpy()
    #
    #     inferred_slices.append(inference)
    # return np.stack(inferred_slices, axis=third_axis)

    with op_indicator(f"Running inference"):
        image = np.require(image, dtype=np.float32)

        third_axis = (set(range(len(image.shape))) - {infer_axis,
                                                      train_axis}).pop()

        train_lr_axis = train_axis
        train_hr_axis = third_axis
        infer_lr_axis = infer_axis
        infer_hr_axis = train_axis
        # Check if order of low-res - high-res axis is different than in training
        transpose_axes = (train_lr_axis > train_hr_axis) != \
                         (infer_lr_axis > infer_hr_axis)

        num_partitions = int(math.ceil(image.shape[third_axis] / chunk_size))

        inferred_chunks: list[np.ndarray] = []
        for chunk in np.array_split(image, num_partitions, axis=third_axis):
            slice_tensors = []
            for sl in slices(chunk, axis=third_axis):
                if transpose_axes:
                    sl = sl.transpose()
                slice_tensors.append(to_tensor(np.stack([sl], axis=-1)))

            chunk_tensor = torch.stack(slice_tensors).to(device)
            chunk_inference = model(chunk_tensor)[:, 0, :,
                              :].detach().cpu().numpy()
            del chunk_tensor

            inferred_chunk_slices: list[np.ndarray] = []
            for sl in slices(chunk_inference, axis=0):
                if transpose_axes:
                    sl = sl.transpose()
                inferred_chunk_slices.append(sl)
            inferred_chunks.append(
                np.stack(inferred_chunk_slices, axis=third_axis))

        inference = np.concatenate(inferred_chunks, axis=third_axis)
        return np.require(inference, requirements='C')
