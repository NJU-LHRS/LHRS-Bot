from typing import Dict, Union

import ml_collections
import torch
import torch.nn as nn
from transformers import BatchEncoding


class BaseModal(nn.Module):
    def __init__(self, config: ml_collections.ConfigDict):
        """
        Base class for all modal encoder

        Parameters
        ----------
        config : configuration of the all modal encoder
        """
        super(BaseModal, self).__init__()

    def encode(self, x: torch.Tensor):
        """
        Encode the input tensor

        Parameters
        ----------
        x : input tensor for the current modal
        -------

        """
        raise NotImplementedError

    def get_modal_input(self, x: Dict[str, Union[str, torch.Tensor]]) -> torch.Tensor:
        """
        Get the input for the each modal encoder

        Parameters
        ----------
        x : whole input dict
        -------
        """
        raise NotImplementedError

    def forward(
        self,
        x: Dict[str, Union[str, torch.Tensor, BatchEncoding]],
        image_embedding: torch.Tensor = None,
        **kwargs,
    ):
        modal_input = self.get_modal_input(x)
        if hasattr(self, "decode"):
            return self.decode(
                **self.encode(modal_input), image_embedding=image_embedding, **kwargs
            )

        return self.encode(modal_input)
