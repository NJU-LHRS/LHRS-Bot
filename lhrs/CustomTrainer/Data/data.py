import os
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

from .transform import *


class BaseDataset(Dataset):
    def __init__(
        self,
        size: Union[Tuple, List],
        data_root: Optional[Union[str, Path]],
        label_root: Optional[Union[str, Path]] = None,
        mode: str = "pretrain",
        txt_file: bool = False,
        txt_file_dir: Optional[Union[str, Path]] = None,
        custom_transform: Callable = None,
    ):
        super(BaseDataset, self).__init__()
        assert mode.lower() in ["pretrain", "val", "train", "test"]
        if txt_file:
            assert txt_file_dir is not None
        if mode.lower() in ["train", "val"]:
            assert label_root is not None

        self.data_root = Path(data_root)
        self.mode = mode.lower()
        self.size = size
        if label_root is not None and self.mode in ["train", "val"]:
            self.label_root = Path(label_root)
        if txt_file:
            self.txt_file_dir = Path(txt_file_dir)
            txt_file_name = mode + ".txt"
            self.txt_file_dir = self.txt_file_dir / txt_file_name

        self.img_list = []
        if txt_file:
            f = open(self.txt_file_dir, "r")
            for line in f.readlines():
                self.img_list.append(line.strip("\n") + ".png")
            f.close()
        else:
            self.img_list = os.listdir(self.data_root)

        if custom_transform is not None:
            self.transform = custom_transform
        elif self.mode in ["pretrain"]:
            self.transform = get_pretrain_transform_BYOL(size)
        elif self.mode in ["train"]:
            self.transform = get_train_transform(size)
        else:
            self.transform = get_test_transform(size)

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, idx: int) -> Dict:
        img_name = self.img_list[idx]
        img = Image.open(self.data_root / img_name)
        img = np.asarray(img)
        if self.mode in ["train", "val"]:
            label = Image.open(self.label_root / img_name)
            label = np.asarray(label)

        if self.mode == "pretrain":
            view1 = self.transform[0](image=img)["image"].type(torch.float32)
            view2 = self.transform[1](image=img)["image"].type(torch.float32)
            img_dict = dict(view1=view1, view2=view2)
        elif self.mode in ["train", "val"]:
            ts = self.transform(image=img, mask=label)
            img = ts["image"]
            label = ts["mask"]
            img_dict = dict(img=img, label=label)
        else:
            ts = self.transform(image=img)
            img = ts["image"]
            img_dict = dict(img=img)

        return img_dict


class BaseMaskDataset(BaseDataset):
    def __init__(
        self,
        grid_size: int = 7,
        input_size: Union[List, Tuple] = [224, 224],
        crop_size: Union[List, Tuple] = [224, 224],
        crop_num: int = 2,
        **kwargs,
    ) -> None:
        super(BaseMaskDataset, self).__init__(**kwargs)
        assert self.mode == "pretrain", "BaseMaskDataset Only Support to mask dataset"
        self.input_size = input_size
        self.crop_size = crop_size
        self.crop_num = crop_num
        self.grid_size = grid_size
        self.transform = get_pretrain_transform(self.crop_size, type="image")
        self.interpolate = get_pretrain_transform(self.crop_size, type="mask")

    def __getitem__(self, idx: int) -> Dict:
        img_name = self.img_list[idx]
        img = Image.open(self.data_root / img_name).resize(self.input_size)
        img = np.asarray(img)
        images = []
        masks = []
        for _ in range(self.crop_num):
            crop = img.copy()
            mask = np.arange(self.grid_size * self.grid_size, dtype=np.uint8).reshape(
                self.grid_size, self.grid_size
            )
            mask = self.interpolate(image=mask)["image"]
            transformed = self.transform(image=crop, mask=mask)
            crop, mask = (
                transformed["image"].type(torch.float32),
                transformed["mask"].float(),
            )
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(1), size=56, mode="nearest"
            ).squeeze()
            images.append(crop), masks.append(mask)

        return dict(views=images, masks=masks)


class Potsdam(BaseDataset):
    CLASSES = [
        "Impervious surfaces",
        "Building",
        "Low vegetation",
        "Tree",
        "Car",
        "background",
    ]

    PALETTE = [
        [255, 255, 255],
        [0, 0, 255],
        [0, 255, 255],
        [0, 255, 0],
        [255, 255, 0],
        [255, 0, 0],
    ]

    def __init__(
        self,
        size: Union[Tuple, List],
        data_dir: str,
        mode: str,
        custom_transform: Callable = None,
    ):
        data_dir = Path(data_dir)
        super(Potsdam, self).__init__(
            size=size,
            data_root=data_dir / "Image",
            label_root=data_dir / "Label",
            mode=mode,
            txt_file=True,
            txt_file_dir=data_dir,
            custom_transform=custom_transform,
        )


class PotsdamMask(BaseMaskDataset):
    def __init__(
        self,
        size: Union[Tuple, List],
        data_dir: str,
        mode: str,
        custom_transform: Callable = None,
    ):
        data_dir = Path(data_dir)
        super(PotsdamMask, self).__init__(
            size=size,
            data_root=data_dir / "Image",
            label_root=data_dir / "Label",
            mode=mode,
            txt_file=True,
            txt_file_dir=data_dir,
            custom_transform=custom_transform,
        )


def LoveDA(
    size: Union[Tuple, List],
    root_dir: str,
    mode: str,
    custom_transform: Callable = None,
) -> Optional[ConcatDataset]:
    root_dir = Path(root_dir)
    dataset = None
    if mode == "pretrain":
        split = ["Train", "Val", "Test"]
    elif mode == "train":
        split = ["train"]
    elif mode == "val":
        split = ["val"]
    else:
        split = ["test"]

    for name in ["Urban", "Rural"]:
        for s in split:
            dir = root_dir / s / name
            sub_dataset = BaseMaskDataset(
                size=size,
                data_root=dir / "images",
                label_root=dir / "annfiles",
                mode=mode,
                custom_transform=custom_transform,
            )
            if dataset is None:
                dataset = sub_dataset
            else:
                dataset += sub_dataset

    return dataset
