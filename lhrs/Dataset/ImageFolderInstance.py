import torch
from torchvision.datasets import ImageFolder
from transformers import BaseImageProcessor


class ImageFolderInstance(ImageFolder):
    def __init__(self, return_index: bool = True, **kwargs):
        super(ImageFolderInstance, self).__init__(**kwargs)
        self.return_index = return_index

    def __getitem__(self, index):
        img, target = super(ImageFolderInstance, self).__getitem__(index)
        if isinstance(self.transform, BaseImageProcessor):
            img = torch.from_numpy(img.pixel_values[0])
        if self.return_index:
            return img, target, index
        else:
            return img, target
