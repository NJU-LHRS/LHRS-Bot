from pathlib import Path
from typing import Callable, Union

from PIL import Image
from torch.utils.data import Dataset


class MillionAidEval(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Callable = None,
        return_idx: bool = False,
    ):
        """

        Parameters
        ----------
        img_file_name : image dir under img root that contain image
        """
        super(MillionAidEval, self).__init__()
        assert split in ["train", "test"], "data split must be train or test"
        if isinstance(root, str):
            root = Path(root)

        self.root = root
        self.split = split
        self.imgs = []
        self.cat_id = []
        self.transform = transform
        self.return_idx = return_idx

        with open(self.root / (self.split + ".txt"), "r") as f:
            for line in f.readlines():
                img_name, idx = line.split(" ")
                idx = int(idx.replace("\n", ""))
                self.imgs.append(img_name)
                self.cat_id.append(idx)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        file, cat_id = self.imgs[item], self.cat_id[item]
        img = Image.open(file)

        if self.transform is not None:
            img = self.transform(img)

        if self.return_idx:
            return img, cat_id, item
        else:
            return img, cat_id
