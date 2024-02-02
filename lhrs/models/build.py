from typing import Tuple

import ml_collections
import torch.nn as nn

from .UniBind import UniBind


def build_vlm_model(
    config: ml_collections.ConfigDict, activate_modal: Tuple[str, str] = ("rgb", "text")
) -> nn.Module:
    model = UniBind(activate_modal, config)
    return model


def build_model(
    config: ml_collections.ConfigDict,
    activate_modal: Tuple[str, str] = ("rgb", "text"),
) -> nn.Module:
    model = build_vlm_model(config, activate_modal=activate_modal)

    return model
