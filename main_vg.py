import json
import logging
import os
import re

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
from lhrs.Dataset import DataCollatorForVGSupervisedDataset, VGEvalDataset
from lhrs.Dataset.conversation import default_conversation
from lhrs.models import build_model
from lhrs.utils import type_dict
from tqdm import tqdm
from transformers import CLIPImageProcessor

logger = logging.getLogger("train")


def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)

    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(
        0, intersection_y2 - intersection_y1 + 1
    )

    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x4 - x3 + 1) * (y4 - y3 + 1)

    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area

    return iou


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

    vis_transform = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    dataset = VGEvalDataset(
        root=config.data_path,
        target=config.data_target,
        transform=vis_transform,
        tokenizer=model.text.tokenizer,
    )
    logger.info(f"Data Length: {len(dataset)}")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=config.workers,
        pin_memory=True,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=DataCollatorForVGSupervisedDataset(model.text.tokenizer),
    )

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
            "CUDA_VISABLE_DEVICES" in os.environ.keys() and len(os.environ["CUDA_VISABLE_DEVICES"].split(",")) == 1
        ):
            device = torch.device("cuda:" + os.environ["CUDA_VISABLE_DEVICES"])
        else:
            device = torch.device("cuda")
    else:
        device = torch.device(config.accelerator)
    model.to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for image, input_ids, targets, file_name, attention_mask in tqdm(
            data_loader, unit_scale=config.batch_size, desc="Evaluating"
        ):
            image = image.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

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
                    num_beams=1,
                    attention_mask=attention_mask,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    max_new_tokens=100,
                )

            outputs = model.text.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            outputs = [output.strip() for output in outputs]
            for pred, target, name in zip(outputs, targets, file_name):
                preds.append(dict(pred=pred, target=target, filename=name))

    save_result(preds, config.output, "eval_save_file", "filename")
    if is_main_process():
        pattern = r"\[([0-9., ]+)\]"

        with open(os.path.join(config.output, "eval_save_file.json")) as f:
            predictions = json.load(f)

        parse_result = []
        fail_instance = 0
        for item in predictions:
            pred_match = re.findall(pattern, item["pred"])
            if len(pred_match) == 0:
                fail_instance += 1

            try:
                pred_result = [list(map(float, match.split(","))) for match in pred_match]
            except:
                fail_instance += 1
                continue

            target_match = re.findall(pattern, item["target"])
            target_result = [list(map(float, match.split(","))) for match in target_match]

            new_pred_result = []
            new_target_result = []
            for pred, target in zip(pred_result, target_result):
                if len(pred) == 4:
                    new_pred_result.append(pred)
                    new_target_result.append(target)
                elif len(pred) > 4:
                    while len(pred) != 4:
                        pred.pop()
                    new_pred_result.append(pred)
                    new_target_result.append(target)
                else:
                    fail_instance += 1

            if len(new_pred_result) > 0:
                parse_result.append(
                    dict(
                        filename=item["filename"],
                        pred=new_pred_result,
                        target=new_target_result,
                    )
                )

        count = 0
        total = 0
        for item in parse_result:
            preds = item["pred"]
            targets = item["target"]

            for pred, target in zip(preds, targets):
                iou_score = calculate_iou(pred, target)
                if iou_score > 0.5:
                    count += 1
                total += 1

        logger.info(f"Accuracy: {count / total * 100}")
        logger.info(f"Fail Sample: {fail_instance}")
        logger.info(f"Accuracy With Fail Sample: {count / (total + fail_instance) * 100}")


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
