from __future__ import annotations

import os
from argparse import ArgumentParser
from math import pi
from typing import Callable, Iterator

import numpy as np
import torch
from napari import view_image
from pyics import write_ics
from torch.utils.data import ChainDataset, IterableDataset
from torchvision.transforms import Compose

import superres.operators.ops as ops
from superres.data.cache import CacheDataset
from superres.data.ics import IcsDataset, LoadIcsImagesFromDir
from superres.data.view import view_dataset
from superres.nn.infer import infer, load_edsr
from superres.nn.train import train_edsr

NamedImage = tuple[np.ndarray, str]
NamedImageDataset = IterableDataset[NamedImage]
NamedImageDatasetFactory = Callable[[...], NamedImageDataset]


def downsample_ics_dataset(images: list[str], axis: int, factor: float,
                           partitions: int,
                           **kwargs: object) -> NamedImageDataset:
    return IcsDataset(
        images,
        transform=ops.downsample(
            axis=axis,
            factor=factor,
            partitions=partitions,
        ),
    )


def bsp_interpolate_ics_dataset(images: list[str], axis: int, factor: float,
                                partitions: int,
                                **kwargs: object) -> NamedImageDataset:
    return IcsDataset(
        images,
        transform=ops.bsp_interpolate(
            axis=axis,
            factor=factor,
            partitions=partitions,
        ),
    )


def rotate_ics_dataset(images: list[str], axis_1: int, axis_2: int,
                       num_rotations: int, partitions: int,
                       **kwargs: object) -> NamedImageDataset:
    def rotations() -> Iterator[NamedImageDataset]:
        yield IcsDataset(images)

        for i in range(1, num_rotations):
            yield IcsDataset(
                images,
                transform=ops.rotate(
                    axes=(axis_1, axis_2),
                    theta=(2 * i * pi) / num_rotations,
                    partitions=partitions,
                ),
                target_transform=lambda name: f"{name}-rot-{i}",
            )

    return ChainDataset(rotations())


def gaussian_blur_ics_dataset(images: list[str], axis: int, partitions: int,
                              fwhm: float | None = None,
                              stdev: float | None = None,
                              **kwargs: object) -> NamedImageDataset:
    return IcsDataset(
        images,
        transform=ops.gaussian_blur(
            axis=axis,
            partitions=partitions,
            fwhm=fwhm,
            stdev=stdev,
        ),
    )


def create_aliasing_on_ics_dataset(images: list[str], axis: int, factor: float,
                                   partitions: int,
                                   **kwargs: object) -> NamedImageDataset:
    return IcsDataset(
        images,
        transform=Compose([
            ops.downsample(axis=axis, factor=factor, partitions=partitions),
            ops.bsp_interpolate(axis=axis, factor=factor,
                                partitions=partitions),
        ]),
    )


def create_dataset_command_parser(
        command_parser: ArgumentParser,
        dataset_factory: NamedImageDatasetFactory) -> ArgumentParser:
    def create_dataset_command(display: bool,
                               output_folder: str | None,
                               **kwargs: object) -> None:
        dataset = dataset_factory(**kwargs)

        for image, label in dataset:
            if output_folder:
                os.makedirs(output_folder, exist_ok=True)
                out_path = os.path.join(output_folder, f"{label}.ics")
                print(f"Saving image {out_path}")
                write_ics(out_path, image)
            if display or not output_folder:
                viewer = view_image(image, title=label)
                viewer.show(block=True)

    command_parser.set_defaults(command=create_dataset_command)
    command_parser.add_argument(
        "images",
        type=str,
        nargs='+',
        help="Input images",
    )
    command_parser.add_argument(
        "-d", "--display",
        action="store_true",
        help="Display results",
    )
    command_parser.add_argument(
        "-o", "--output-folder",
        type=str,
        help="Folder to output transformed images",
    )

    return command_parser


def view_command(images: list[str], **kwargs: object) -> None:
    dataset = IcsDataset(images)
    view_dataset(dataset)


def train_command(images: list[str], target_folder: str, out_model: str,
                  scale: int, infer_axis: int, rgb_range: int,
                  in_memory: bool, device: str, epochs: int, batch_size: int,
                  patch_size: int, learning_rate: float,
                  **kwargs: object) -> None:
    if len(images) == 1:
        # Avoid needlessly re-reading the image from disk
        in_memory = True

    train_dataset: IterableDataset[tuple[np.ndarray, np.ndarray]] = IcsDataset(
        images,
        target_transform=LoadIcsImagesFromDir(target_folder),
    )

    if in_memory:
        print("Keeping all training images in memory when training")
        train_dataset = CacheDataset(train_dataset)

    model = train_edsr(
        train_dataset,
        scale=scale,
        infer_axis=infer_axis,
        rgb_range=rgb_range,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patch_size=patch_size,
    )

    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    torch.save(model.state_dict(), out_model)


def inferred_dataset(model: str, images: list[str], scale: int, rgb_range: int,
                     infer_axis: int, train_axis: int, device: str,
                     **kwargs: object) -> NamedImageDataset:
    model = load_edsr(
        model_path=model,
        scale=scale,
        rgb_range=rgb_range,
        device=device,
    )

    return IcsDataset(
        images,
        transform=lambda image: infer(
            model=model, image=image, scale=scale, infer_axis=infer_axis,
            train_axis=train_axis, device=device),
    )


def main() -> None:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    downsample_parser = create_dataset_command_parser(
        subparsers.add_parser(
            "downsample",
            help="Downsample images on an axis",
        ),
        dataset_factory=downsample_ics_dataset,
    )
    downsample_parser.add_argument(
        "-a", "--axis",
        type=int,
        default=0,
        help="Axis to downsample",
    )
    downsample_parser.add_argument(
        "-x", "--factor",
        type=float,
        default=2.0,
        help="Downsample factor",
    )
    downsample_parser.add_argument(
        "-p", "--partitions",
        type=float,
        default=1,
        help="Partition the input into N parts so they fit into GPU memory",
    )

    bsp_interpolate_parser = create_dataset_command_parser(
        subparsers.add_parser(
            "bsp-interpolate",
            help="B-spline interpolate images on an axis",
        ),
        dataset_factory=bsp_interpolate_ics_dataset,
    )
    bsp_interpolate_parser.add_argument(
        "-a", "--axis",
        type=int,
        default=0,
        help="Axis to interpolate",
    )
    bsp_interpolate_parser.add_argument(
        "-x", "--factor",
        type=float,
        default=2.0,
        help="Interpolation factor",
    )
    bsp_interpolate_parser.add_argument(
        "-p", "--partitions",
        type=float,
        default=1,
        help="Partition the input into N parts so they fit into GPU memory",
    )

    rotate_parser = create_dataset_command_parser(
        subparsers.add_parser(
            "rotate",
            help="Generate N rotations of images",
        ),
        dataset_factory=rotate_ics_dataset,
    )
    rotate_parser.add_argument(
        "--axis-1",
        type=int,
        default=0,
        help="First axis of the rotated plane",
    )
    rotate_parser.add_argument(
        "--axis-2",
        type=int,
        default=1,
        help="Second axis of the rotated plane",
    )
    rotate_parser.add_argument(
        "-n", "--num-rotations",
        type=int,
        default=8,
        help="Number of rotations to generate per image",
    )
    rotate_parser.add_argument(
        "-p", "--partitions",
        type=float,
        default=1,
        help="Partition the input into N parts so they fit into GPU memory",
    )

    gaussian_blur_parser = create_dataset_command_parser(
        subparsers.add_parser(
            "gaussian-blur",
            help="Apply gaussian blur on an axis",
        ),
        dataset_factory=gaussian_blur_ics_dataset,
    )
    gaussian_blur_parser.add_argument(
        "-a", "--axis",
        type=int,
        default=0,
        help="Blur axis",
    )
    gaussian_blur_parser.add_argument(
        "--fwhm",
        type=float,
        nargs='?',
        help="Full-width at half-maximum parameter",
    )
    gaussian_blur_parser.add_argument(
        "--stdev",
        type=float,
        nargs='?',
        help="Sigma / standard deviation parameter",
    )
    gaussian_blur_parser.add_argument(
        "-p", "--partitions",
        type=float,
        default=1,
        help="Partition the input into N parts so they fit into GPU memory",
    )

    create_aliasing_parser = create_dataset_command_parser(
        subparsers.add_parser(
            "create-aliasing",
            help="Create aliasing on an axis by downsampling and upsampling",
        ),
        dataset_factory=create_aliasing_on_ics_dataset,
    )
    create_aliasing_parser.add_argument(
        "-a", "--axis",
        type=int,
        default=0,
        help="Axis to create aliasing on",
    )
    create_aliasing_parser.add_argument(
        "-x", "--factor",
        type=float,
        default=2.0,
        help="Factor of downsample / upsample operations",
    )
    create_aliasing_parser.add_argument(
        "-p", "--partitions",
        type=float,
        default=1,
        help="Partition the input into N parts so they fit into GPU memory",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train an EDSR network",
    )
    train_parser.set_defaults(command=train_command)
    train_parser.add_argument(
        "images",
        type=str,
        nargs='+',
        help="Input images",
    )
    train_parser.add_argument(
        "-t", "--target-folder",
        type=str,
        help="Folder with target images",
    )
    train_parser.add_argument(
        "-o", "--out-model",
        type=str,
        help="Where to store trained model",
    )
    train_parser.add_argument(
        "-s", "--scale",
        type=int,
        default=2,
        help="Super-resolution scale",
    )
    train_parser.add_argument(
        "--infer-axis",
        type=int,
        default=0,
        help="Inference axis",
    )
    train_parser.add_argument(
        "--rgb-range",
        type=int,
        default=255,
        help="Max color value",
    )
    train_parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Keep all images in memory while training",
    )
    train_parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to train on",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of passes over the whole dataset",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Size of mini-batch",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
        help="Learning rate",
    )
    train_parser.add_argument(
        "--patch-size",
        type=int,
        default=32,
        help="Size of cropped regions to train on",
    )

    infer_parser = create_dataset_command_parser(
        subparsers.add_parser(
            "infer",
            help="Infer with an EDSR network",
        ),
        dataset_factory=inferred_dataset,
    )
    infer_parser.add_argument(
        "-m", "--model",
        type=str,
        help="Trained model",
    )
    infer_parser.add_argument(
        "-s", "--scale",
        type=int,
        default=2,
        help="Super-resolution scale",
    )
    infer_parser.add_argument(
        "--train-axis",
        type=int,
        default=2,
        help="Train axis",
    )
    infer_parser.add_argument(
        "--infer-axis",
        type=int,
        default=0,
        help="Inference axis",
    )
    infer_parser.add_argument(
        "--rgb-range",
        type=int,
        default=255,
        help="Max color value",
    )
    infer_parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on",
    )

    view_parser = subparsers.add_parser(
        "view",
        help="View ICS images",
    )
    view_parser.set_defaults(command=view_command)
    view_parser.add_argument(
        "images",
        type=str,
        nargs='+',
        help="Images to view",
    )

    args = parser.parse_args()
    args.command(**vars(args))
