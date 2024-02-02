import torch
from torchvision.datasets import ImageFolder
from transformers import CLIPImageProcessor

CLASS_NAME_MAP = {
    "AID": [
        "Airport",
        "BareLand",
        "BaseballField",
        "Beach",
        "Bridge",
        "Center",
        "Church",
        "Commercial",
        "DenseResidential",
        "Desert",
        "Farmland",
        "Forest",
        "Industrial",
        "Meadow",
        "MediumResidential",
        "Mountain",
        "Park",
        "Parking",
        "Playground",
        "Pond",
        "Port",
        "RailwayStation",
        "Resort",
        "River",
        "School",
        "SparseResidential",
        "Square",
        "Stadium",
        "StorageTanks",
        "Viaduct",
    ]
}


class ImageFolderInstance(ImageFolder):
    def __init__(self, dataset_name: str, return_index: bool = True, **kwargs):
        assert (
            dataset_name in CLASS_NAME_MAP.keys()
        ), "dataset name must be in {}".format(CLASS_NAME_MAP.keys())
        super(ImageFolderInstance, self).__init__(**kwargs)
        self.CLASS_NAME = CLASS_NAME_MAP[dataset_name]
        self.return_index = return_index

    def __getitem__(self, index):
        img, target = super(ImageFolderInstance, self).__getitem__(index)
        if isinstance(self.transform, CLIPImageProcessor):
            img = torch.from_numpy(img.pixel_values[0])
        if self.return_index:
            return img, target, index
        else:
            return img, target
