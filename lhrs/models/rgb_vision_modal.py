import math
from typing import Callable, Dict, Tuple, Union

import ml_collections
import torch
from torch.nn import functional as F
from transformers.models.siglip.modeling_siglip import SiglipVisionModel

from .base_modal import BaseModal

type_dict = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class VisionModal(BaseModal):
    EMBEDDING_DIM = dict(
        vit_large=1152,
    )

    def __init__(self, config: ml_collections.ConfigDict):
        super(VisionModal, self).__init__(config)

        self.arch = config.rgb_vision.arch.lower()
        assert self.arch in [
            "vit_base",
            "vit_large",
        ], "rgb vision arch should be one of swin_base, swin_large, vit_base, vit_large"

        kwargs = {}
        if config.dtype == "int8":
            kwargs["load_in_8bit"] = True
        elif config.dtype == "int4":
            kwargs["load_in_4bit"] = True
        else:
            kwargs["torch_dtype"] = type_dict[config.dtype]

        self.embedding_dim = self.EMBEDDING_DIM[self.arch]

        self.encoder = SiglipVisionModel.from_pretrained(config.rgb_vision.vit_name)
        self.encoder.vision_model.encoder.layers = self.encoder.vision_model.encoder.layers[:-1]
        if getattr(config, "use_checkpoint", False):
            self.encoder.gradient_checkpointing_enable()

        if self.arch.startswith("vit"):
            self.extract_stage = [
                self.encoder.vision_model.config.num_hidden_layers // 3,
                self.encoder.vision_model.config.num_hidden_layers // 3 * 2,
                self.encoder.vision_model.config.num_hidden_layers - 1,
            ]

    def get_config(self):
        return self.encoder.vision_model.config

    def encode(self, x: torch.Tensor):
        if self.arch.startswith("vit"):
            outputs = self.encoder(
                x,
                return_dict=True,
                output_hidden_states=True,
            )
            if hasattr(self, "extract_stage"):
                image_embeds = []
                for _, stage in enumerate(self.extract_stage):
                    current_hidden_states = outputs.hidden_states[stage]
                    image_embeds.append(current_hidden_states)
                image_embeds = torch.cat(image_embeds, dim=1)
                return image_embeds
        else:
            raise NotImplementedError(f"arch {self.arch} not implemented")

    def get_modal_input(self, x: Dict[str, Union[str, torch.Tensor]]) -> torch.Tensor:
        img = x["rgb"]
        return img
