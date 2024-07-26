import json
import logging
import os
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import List

import ml_collections.config_dict
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from lhrs.CustomTrainer import init_distributed
from lhrs.CustomTrainer.utils import ConfigArgumentParser, setup_logger, str2bool
from lhrs.CustomTrainer.utils.distribute import (
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
)
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
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor

logger = logging.getLogger("train")


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def save_result(result, result_dir, filename, remove_duplicate=""):
    result_file = os.path.join(result_dir, "%s_rank%d.json" % (filename, get_rank()))
    final_result_file = os.path.join(result_dir, "%s.json" % filename)

    json.dump(result, open(result_file, "w"))

    if is_distributed():
        dist.barrier()

    if is_main_process():
        # combine results from all processes
        result = []

        for rank in range(get_world_size()):
            result_file = os.path.join(result_dir, "%s_rank%d.json" % (filename, rank))
            res = json.load(open(result_file, "r"))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new

        json.dump(result, open(final_result_file, "w"))
        logger.info("result file saved to %s" % final_result_file)

    return final_result_file


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
    parser.add_argument("--data-target", type=str, help="path to dataset annotation file ")
    parser.add_argument(
        "--data-type",
        type=str,
        choices=["LR", "HR"],
        help="VQA dataset type",
        default="HR",
    )
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
    vis_transform = CLIPImageProcessor.from_pretrained(config.rgb_vision.vit_name)
    tokenizer = model.text.tokenizer
    dtype = type_dict[config.dtype]
    model.to(dtype)

    # load model
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
        device = torch.device("cuda")
    else:
        device = torch.device(config.accelerator)
    model.to(device)
    model.eval()

    # load data
    image_root = Path(config.data_path)
    qa_data = json.load(open(config.data_target, "r"))

    question_answer = qa_data["data"]
    qtype = qa_data["qtype"]
    id_2_type = {}
    for key, _ in qtype.items():
        type_id = key.split(" ")[0]
        type_name = key.split(" ")[1]
        id_2_type[type_id] = type_name
    choice_list = ["A.", "B.", "C.", "D.", "E.", "F.", "G.", "H.", "I.", "J.", "K.", "L.", "M.", "N.", "O.", "P."]
    gathed_result = defaultdict(list)
    with torch.no_grad():
        for data_dict in tqdm(question_answer, desc="Evaluating"):
            filename = data_dict["filename"]
            image_path = image_root / filename
            assert image_path.exists(), f"Image not found: {image_path}"

            image = Image.open(image_path).convert("RGB")
            image_tensor = vis_transform(image, return_tensors="pt")["pixel_values"].to(device).to(dtype)

            qa_pairs = data_dict["qa_pairs"]
            for qa_pair in qa_pairs:
                question = qa_pair["question"]
                choices = qa_pair["choices"]
                answer = qa_pair["answer"]
                types = qa_pair["type"]

                if config.tune_im_start:
                    question = (
                        DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
                    )
                else:
                    question = DEFAULT_IMAGE_TOKEN + "\n" + question

                inp = (
                    question + "\nChoices: " + choices + " Answer from the given choices with A., B., C., D., etc."
                )
                choices_logits_list = []
                dummy_conv = default_conversation.copy()
                dummy_conv.append_message(dummy_conv.roles[0], inp)
                dummy_conv.append_message(dummy_conv.roles[1], None)
                dummy_input = (
                    tokenizer_image_token(
                        dummy_conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    )
                    .unsqueeze(0)
                    .to(device)
                )
                with torch.autocast(
                    device_type="cuda" if config.accelerator == "gpu" else "cpu",
                    enabled=config.enable_amp,
                    dtype=dtype,
                ):
                    output_ids = model.generate(
                        dummy_input,
                        images=image_tensor,
                        do_sample=False,
                        num_beams=1,
                        temperature=1.0,
                        top_p=1.0,
                        max_new_tokens=10,
                    )

                    outputs = model.text.tokenizer.batch_decode(output_ids, skip_special_tokens=False)
                    outputs = outputs[0].split("<|eot_id|>")[0]
                    output = outputs[0].strip()

                output = output.lower()
                answer = answer.lower()
                output = normalize_answer(output)
                answer = normalize_answer(answer)
                if output == answer:
                    correct = 1
                else:
                    correct = 0

                gathed_result["total"].append(correct)
                for type_id in types:
                    type_name = id_2_type[type_id]
                    gathed_result[type_name].append(correct)

    for key, value in gathed_result.items():
        if key == "total":
            continue
        acc = np.mean(value)
        # percentage with 2 floating point
        acc = round(acc * 100, 2)
        logger.info(f"Type: {key}, accuracy: {acc}%")

    acc = np.mean(gathed_result["total"])
    acc = round(acc * 100, 2)
    logger.info(f"Total accuracy: {acc}%")


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
