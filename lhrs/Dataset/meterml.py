from pathlib import Path
from typing import Callable, Union

import geopandas as gpd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}


def convert_folder(folder_name):
    return folder_name.split("/")[1]


class METERMLDataset(Dataset):
    CLASS_NAME = [
        "Other",
        "concentrated animal feeding operations",
        "landfills",
        "coal mines",
        "natural gas processing plants",
        "refineries and petroleum terminals",
        "wastewater treatment plants",
    ]

    class_dict = {
        "Negative": 0,
        "CAFOs": 1,
        "Landfills": 2,
        "Mines": 3,
        "ProcPlants": 4,
        "R&Ts": 5,
        "WWTPs": 6,
    }

    def __init__(
        self, root: Union[str, Path], split: str, mode: str, transform: Callable = None
    ) -> None:
        assert split.lower() in ["train", "test", "val"]
        assert mode.lower() in [
            "naip_rgb",
            "s2_rgb",
        ], "%s is not implemented currently." % (mode.lower())
        super().__init__()
        if isinstance(root, str):
            root = Path(root)
        self.root = root
        self.split = split
        self.img_dir = self.root / (split + "_images")

        geojson = gpd.read_file(self.root / (split + ".geojson"))
        geojson["Image_Folder"] = geojson["Image_Folder"].apply(convert_folder)

        self.image_folder = geojson["Image_Folder"].values
        self.idx = geojson["idx"].values
        del geojson

        self.transform = transform
        self.mode = mode.lower()

        assert (
            self.image_folder.size == self.idx.size
        ), "The length of label is not match with those of images folder"

    def __len__(self) -> int:
        return self.image_folder.size

    def __getitem__(self, index):
        image_folder = self.image_folder[index]
        label = self.idx[index]

        if self.mode == "naip_rgb":
            img = Image.open(self.img_dir / image_folder / "naip.png")
            img = img.convert("RGB")
        elif self.mode == "s2_rgb":
            img = np.load(self.img_dir / image_folder / "sentinel-2-10m.npy")
            img = img[:, :, :3]
            img = img.astype(np.float32) / 10000

        if self.transform is not None:
            img = self.transform(img)

        return img, label
