import os

from napari import view_image

from superres.dataset.ics import IcsDataset, save_dataset_as_ics
from superres.dataset.preprocessing import BspInterpolate


def view_ics(in_file: str) -> None:
    ds = IcsDataset([in_file])

    viewer = view_image(ds[0][0])
    viewer.show(block=True)


def preprocess(in_file: str, out_folder: str) -> None:
    print(f"Preprocessing file '{in_file}'")
    print(f"Output folder '{out_folder}'")

    os.makedirs(out_folder, exist_ok=True)

    ds = IcsDataset(
        [in_file],
        transform=BspInterpolate(dim=0, num_samples=256),
    )

    save_dataset_as_ics(ds, out_folder)
