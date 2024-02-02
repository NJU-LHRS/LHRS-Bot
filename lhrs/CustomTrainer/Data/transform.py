from typing import List

import albumentations as A
import cv2
from albumentations.pytorch.transforms import ToTensorV2


def get_pretrain_transform(size, type) -> A.Compose:
    assert type in ["image", "mask"]
    if type == "image":
        transform = A.Compose(
            [
                A.RandomResizedCrop(size[0], size[1], scale=(0.08, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
                A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                A.GaussianBlur(blur_limit=(3, 7), sigma_limit=[0.1, 2.0], p=0.5),
                A.GaussNoise(p=0.6),
                A.Solarize(p=0.2),
                A.ToGray(p=0.2),
                A.Normalize(
                    mean=[0.33797, 0.3605, 0.3348], std=[0.1359, 0.1352, 0.1407]
                ),
                ToTensorV2(),
            ]
        )
    else:
        transform = A.Resize(*size, interpolation=cv2.INTER_NEAREST_EXACT)

    return transform


def get_pretrain_transform_BYOL(size) -> List[A.Compose]:
    transform1 = A.Compose(
        [
            A.RandomResizedCrop(size[0], size[1], scale=(0.2, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(0.4, 0.4, 0.2, 0.1, p=0.8),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=[0.1, 2.0], p=1.0),
            A.GaussNoise(p=0.6),
            A.Solarize(p=0.0),
            A.ToGray(p=0.2),
            A.Normalize(mean=[0.33797, 0.3605, 0.3348], std=[0.1359, 0.1352, 0.1407]),
            ToTensorV2(),
        ]
    )

    transform2 = A.Compose(
        [
            A.RandomResizedCrop(size[0], size[1], scale=(0.2, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(0.4, 0.4, 0.2, 0.1, p=0.8),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=[0.1, 2.0], p=0.1),
            A.GaussNoise(p=0.6),
            A.Solarize(p=0.2),
            A.ToGray(p=0.2),
            A.Normalize(mean=[0.33797, 0.3605, 0.3348], std=[0.1359, 0.1352, 0.1407]),
            ToTensorV2(),
        ]
    )

    transforms = [transform1, transform2]
    return transforms


def get_train_transform(size) -> A.Compose:
    transform = A.Compose(
        [
            A.RandomResizedCrop(size[0], size[1], scale=(0.2, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GaussNoise(p=0.5),
            A.GaussianBlur((3, 3), (1.5, 1.5), p=0.3),
            A.Normalize(mean=[0.33797, 0.3605, 0.3348], std=[0.1359, 0.1352, 0.1407]),
            ToTensorV2(),
        ]
    )

    return transform


def get_test_transform(size) -> A.Compose:
    transform = A.Compose(
        [
            A.Resize(size[0], size[1]),
            A.Normalize(mean=[0.33797, 0.3605, 0.3348], std=[0.1359, 0.1352, 0.1407]),
            ToTensorV2(),
        ]
    )

    return transform
