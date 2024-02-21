import json
import logging
import os
from difflib import SequenceMatcher
from typing import Dict, List

import ml_collections.config_dict
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from lhrs.CustomTrainer import init_distributed
from lhrs.CustomTrainer.utils import ConfigArgumentParser, setup_logger, str2bool
from lhrs.Dataset.build_loader import build_zero_shot_loader
from lhrs.Dataset.conversation import default_conversation
from lhrs.models import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    build_model,
    tokenizer_image_token,
)
from lhrs.utils import type_dict
from sklearn.metrics import balanced_accuracy_score, classification_report
from tqdm import tqdm

logger = logging.getLogger("train")


CLS_TEMPLATE = [lambda c: f"[CLS] Choose the best categories describe the image from: {c}"]


def find_index_of_max_similar_substring(given_string, string_list):
    max_similarity = 0
    max_index = -1

    for i, string in enumerate(string_list):
        similarity = (
            SequenceMatcher(None, given_string, string)
            .find_longest_match(0, len(given_string), 0, len(string))
            .size
        )
        if similarity > max_similarity:
            max_similarity = similarity
            max_index = i

    return max_index


def classname_2_idx(preds: List[str], classes_to_idx: Dict[str, int]):
    results = []
    classes = list(classes_to_idx.keys())
    for pred in preds:
        pred = pred.strip()
        if pred in classes:
            results.append(classes_to_idx[pred])
        else:
            index = find_index_of_max_similar_substring(pred, classes)
            results.append(classes_to_idx[classes[index]])
    return results


def parse_option():
    parser = ConfigArgumentParser()
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    # basic
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument("--workers", type=int, default=8, help="workers of dataloader")
    parser.add_argument("--model-path", type=str, default=None, help="pretrained checkpoint path")
    parser.add_argument("--enable-amp", type=str2bool, default=False, help="mixed precision")
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--seed", type=int, default=322, help="random seed")
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )
    parser.add_argument("--gpus", type=int, default=0, help="gpus ID")
    parser.add_argument(
        "--inf_sampler",
        type=str2bool,
        default=False,
        help="Use Infinite loader if ture, else default datalodaer (Usually, inf_sampler for iterbased training)",
    )

    # wandb
    parser.add_argument("--wandb", type=str2bool, default=False, help="wandb logger")
    parser.add_argument("--entity", type=str, default="pumpkinn", help="wandb entity")
    parser.add_argument("--project", type=str, default="MaskIndexNet", help="wandb project")

    # HardWare
    parser.add_argument(
        "--accelerator",
        default="cpu",
        type=str,
        choices=["cpu", "gpu", "mps"],
        help="accelerator",
    )
    parser.add_argument("--local_rank", type=int, help="local rank")

    config = parser.parse_args(wandb=True)
    config = ml_collections.config_dict.ConfigDict(config)

    return config


def main(config: ml_collections.ConfigDict):
    logger.info(f"Creating model")
    model = build_model(config, activate_modal=("rgb", "text"))
    dtype = type_dict[config.dtype]
    model.to(dtype)

    data_loader_train = build_zero_shot_loader(config, mode="zero_shot_cls")

    if config.model_path is not None:
        logger.info(f"Loading pretrained checkpoint from {config.model_path}")
        if getattr(model, "custom_load_state_dict", False):
            msg = model.custom_load_state_dict(config.model_path)
        else:
            ckpt = torch.load(config.model_path, map_location="cpu")
            msg = model.load_state_dict(ckpt["model"], strict=False)
        if msg is not None:
            logger.info(f"After loading, missing keys: {msg.missing_keys}, unexpected keys: {msg.unexpected_keys}")
            logger.info(str(model))

    if config.accelerator == "gpu":
        if config.is_distribute:
            device = torch.device(getattr(config, "local_rank", 0))
        elif (
            "CUDA_VISABLE_DEVICE" in os.environ.keys() and len(os.environ["CUDA_VISABLE_DEVICES"].split(",")) == 1
        ):
            device = torch.device("cuda:" + os.environ["CUDA_VISABLE_DEVICES"])
        else:
            device = torch.device("cuda")
    else:
        device = torch.device(config.accelerator)
    model.to(device)
    model.eval()

    if hasattr(data_loader_train.dataset, "classes"):
        all_classes = data_loader_train.dataset.classes
    else:
        all_classes = data_loader_train.dataset.CLASS_NAME
    all_classes = [i.lower().replace("_", " ") for i in all_classes]
    classes_2_idx = {classname: idx for idx, classname in enumerate(all_classes)}
    inp = CLS_TEMPLATE[0](all_classes)

    conv = default_conversation.copy()
    roles = conv.roles

    if config.tune_im_start:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inp
    else:
        inp = DEFAULT_IMAGE_TOKEN + "\n" + inp

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt, model.text.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(device)
    )
    input_ids = input_ids.repeat(config.batch_size, 1)
    model.eval()
    with torch.no_grad():
        preds = []
        trues = []
        for image, target in tqdm(data_loader_train, unit_scale=config.batch_size, desc="Evaluating"):
            image = image.to(dtype).to(device)
            if input_ids.shape[0] != image.shape[0]:
                # last iter
                input_ids = input_ids[: image.shape[0]]

            with torch.autocast(
                device_type="cuda" if config.accelerator == "gpu" else "cpu",
                enabled=config.enable_amp,
                dtype=dtype,
            ):
                output_ids = model.generate(
                    input_ids=input_ids,
                    images=image,
                    do_sample=False,
                    num_beams=1,
                    temperature=1.0,
                    top_p=1.0,
                    max_new_tokens=20 if config.eval.dataset != "METERML" else 30,
                )

            outputs = model.text.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            preds += outputs
            trues.append(target.cpu())

    preds = classname_2_idx(preds, classes_2_idx)
    trues = torch.cat(trues)
    mean_per_class_recall = balanced_accuracy_score(trues, preds)
    logger.info(classification_report(trues, preds, digits=3, target_names=all_classes))
    logger.info(mean_per_class_recall)


if __name__ == "__main__":
    config = parse_option()

    config.rank, config.local_rank, config.world_size = init_distributed()
    config.is_distribute = config.world_size > 1
    config.adjust_norm = False
    print(config)

    setup_logger("train", output=config.output, rank=config.rank)
    os.makedirs(config.output, exist_ok=True)

    if config.is_distribute:
        seed = config.seed + dist.get_rank()
    else:
        seed = config.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    if config.rank == 0:
        path = os.path.join(config.output, "config.json")
        with open(path, "w") as f:
            configDict = dict(config.to_dict())
            json.dump(configDict, f, indent=4)
        logger.info(f"Full config saved to {path}")
        logger.info(config)

    if config.wandb and config.rank == 0:
        wandb.init(config=config.to_dict(), entity=config.entity, project=config.project)
        config = wandb.config

    main(config)
