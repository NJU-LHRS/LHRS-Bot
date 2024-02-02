from typing import Sequence

import numpy as np
import torch
import torchvision
from PIL import Image


def to_numpy(data, ToCHW=True):
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, Image.Image):
        data = np.array(data, dtype=np.uint8)
        if data.ndim < 3:
            data = np.expand_dims(data, axis=-1)
        if ToCHW:
            data = np.rollaxis(data, 2)  # HWC to CHW
        return data
    elif isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    else:
        raise TypeError(f"Type {type(data)} cannot be converted to numpy.")


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).type(torch.float32)
    elif isinstance(data, Image.Image):
        return torchvision.transforms.functional.to_tensor(data)
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f"Type {type(data)} cannot be converted to tensor."
            "Supported types are: `numpy.ndarray`, `torch.Tensor`, "
            "`Sequence`, `int` and `float`"
        )
