import logging

import ml_collections
import torch.nn as nn
from timm.optim.optim_factory import create_optimizer_v2

logger = logging.getLogger("train")


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword.split["."][-1] in name:
            isin = True
    return isin


def get_pretrain_param_groups(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    has_decay_name = []
    no_decay_name = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or (name.split(".")[-1] in skip_list)
            or check_keywords_in_name(name, skip_keywords)
        ):
            no_decay.append(param)
            no_decay_name.append(name)
        else:
            has_decay.append(param)
            has_decay_name.append(name)
    return [{"params": has_decay}, {"params": no_decay, "weight_decay": 0.0}]


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or (name.split(".")[-1] in skip_list)
            or check_keywords_in_name(name, skip_keywords)
        ):
            no_decay.append(param)
            logger.info(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{"params": has_decay}, {"params": no_decay, "weight_decay": 0.0}]


def get_param_group(model: nn.Module, is_pretrain: bool = True):
    skip = {}
    skip_keywords = {}
    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()
    if hasattr(model, "no_weight_decay_keywords"):
        skip_keywords = model.no_weight_decay_keywords()

    if is_pretrain:
        parameters = get_pretrain_param_groups(model, skip, skip_keywords)
    else:
        parameters = set_weight_decay(model, skip, skip_keywords)
    return parameters


def build_optimizer(
    model: nn.Module, config: ml_collections.ConfigDict, is_pretrain: bool = True
):
    parameters = get_param_group(model, is_pretrain)
    return create_optimizer_v2(
        parameters,
        opt=config.optimizer,
        lr=config.lr,
        weight_decay=config.wd,
        filter_bias_and_bn=False,
    )
