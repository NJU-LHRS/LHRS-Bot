import json
import logging
import os

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
import deepspeed
import ml_collections.config_dict
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from lhrs.CustomTrainer import deepspeed_init_distributed
from lhrs.CustomTrainer.EpochBasedTrainer import EpochBasedTrainer
from lhrs.CustomTrainer.utils import (
    ConfigArgumentParser,
    auto_resume_helper,
    setup_logger,
    str2bool,
)
from lhrs.Dataset.build_loader import build_loader
from lhrs.models import build_model
from lhrs.optimizer import build_optimizer

logger = logging.getLogger("train")


def build_ds_config(config: ml_collections.ConfigDict):
    opt_lower = config.optimizer.lower()
    if opt_lower == "adamw":
        optimizer = {
            "type": "AdamW",
            "params": {
                "lr": config.lr,
                "eps": 1e-8,
                "betas": (0.9, 0.95),
                "weight_decay": config.wd,
            },
        }

        ds_config = {
            "train_micro_batch_size_per_gpu": config.batch_size,
            "optimizer": optimizer,
            "fp16": {
                "enabled": True if config.fp16 else False,
                "auto_cast": False,
                "initial_scale_power": 16,
                "loss_scale_window": 500,
            },
            "bf16": {
                "enabled": True if config.bf16 else False,
                "auto_cast": False,
            },
            "zero_optimization": {
                "stage": 2,
                "sub_group_size": 1e9,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "stage3_gather_16bit_weights_on_model_save": True,
            },
            "gradient_accumulation_steps": config.accumulation_steps,
            "gradient_clipping": config.max_grad_norm,
        }

    else:
        ds_config = {
            "train_micro_batch_size_per_gpu": config.batch_size,
            "bf16": {
                "enabled": True,
                "auto_cast": True,
            },
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu",
                },
                "offload_param": {"device": "cpu"},
            },
            "gradient_accumulation_steps": config.accumulation_steps,
            "gradient_clipping": config.max_grad_norm,
            "zero_force_ds_cpu_optimizer": False,
            "zero_allow_untested_optimizer": True,
        }

    return ds_config


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
    parser.add_argument("--eval-data-path", type=str, help="path to evaluate dataset")
    parser.add_argument("--workers", type=int, default=8, help="workers of dataloader")
    parser.add_argument(
        "--auto-resume", action="store_true", help="resume from checkpoint"
    )
    parser.add_argument(
        "--resume-path", type=str, default=None, help="resume checkpoint path"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="pretrained checkpoint path for model (maybe stage 1)",
    )
    parser.add_argument(
        "--accumulation-steps", type=int, default=1, help="gradient accumulation steps"
    )
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--enable-amp", type=str2bool, default=False, help="mixed precision"
    )
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--seed", type=int, default=322, help="random seed")
    parser.add_argument("--gpus", type=int, default=0, help="gpus ID")
    parser.add_argument(
        "--inf_sampler",
        type=str2bool,
        default=False,
        help="Use Infinite loader if ture, else default datalodaer (Usually, inf_sampler for iterbased training)",
    )
    parser.add_argument(
        "--torch-compile",
        type=str2bool,
        default=False,
        help="Use torch.compile to accelerate model or not",
    )

    # wandb
    parser.add_argument("--wandb", type=str2bool, default=False, help="wandb logger")
    parser.add_argument("--entity", type=str, default="pumpkinn", help="wandb entity")
    parser.add_argument(
        "--project", type=str, default="MultiModal", help="wandb project"
    )
    parser.add_argument(
        "--job-type", type=str, default="vlm_test", help="wandb job_type"
    )
    parser.add_argument(
        "--tags", type=str, default="MultiModal", nargs="+", help="wandb tags"
    )
    parser.add_argument("--name", type=str, default="first_run", help="wandb run name")
    parser.add_argument("--notes", type=str, default=None, help="wandb run's notes")

    # HardWare
    parser.add_argument(
        "--accelerator",
        default="cpu",
        type=str,
        choices=["cpu", "gpu", "mps"],
        help="accelerator",
    )
    parser.add_argument("--local_rank", type=int)

    config = parser.parse_args(wandb=True)
    config = ml_collections.config_dict.ConfigDict(config)

    return config


def main(config):
    logger.info(f"Creating model")
    model = build_model(
        config,
        activate_modal=("rgb", "text"),
    )
    logger.info(str(model) + "\n")

    logger.info(f"Building Dataset")
    data_loader_train = build_loader(
        config,
        mode="pretrain",
        tokenizer=model.text.tokenizer,
        prompt_type=config.prompt_template,
    )

    compute_dtype = (
        torch.float16
        if config.fp16
        else (torch.bfloat16 if config.bf16 else torch.float32)
    )

    model.prepare_for_training(
        freeze_vision=not config.tune_rgb_bk,
        freeze_text=not config.lora.enable,
        tune_rgb_pooler=config.tune_rgb_pooler,
        model_path=config.model_path,
        tune_im_start=config.tune_im_start,
        compute_dtype=compute_dtype,
    )

    if config.optimizer.lower() == "adamw":
        parameter = None
        optimizer = None
    else:
        parameter = None
        optimizer = build_optimizer(model, config, is_pretrain=True)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        config=build_ds_config(config),
        model=model,
        optimizer=optimizer if optimizer is not None else None,
        model_parameters=parameter if parameter is not None else None,
    )

    trainer = EpochBasedTrainer(
        model=model_engine,
        optimizer=optimizer,
        lr_scheduler=config.schedule,
        data_loader=data_loader_train,
        max_epochs=config.epochs,
        work_dir=config.output,
        log_period=1,
        save_ckpt_by="iter",
        ckpt_period=100,
        accelerator=config.accelerator,
        enable_amp=config.enable_amp,
        wandb=config.wandb,
        gpus=0,
        max_num_checkpoints=1,
        clip_grad_norm=config.max_grad_norm,
        is_distributed=config.is_distribute,
        torch_compile=config.torch_compile,
        dtype=compute_dtype,
        deepspeed=True,
    )

    if config.auto_resume:
        resume_file = auto_resume_helper(config.output)
        if resume_file:
            if config.resume_path is not None:
                logger.warning(
                    f"auto-resume changing resume file from {config.resume_path} to {resume_file}"
                )
            config.resume_path = resume_file
            logger.info(f"auto resuming from {resume_file}")
        else:
            logger.info(
                f"no checkpoint found in {config.output}/checkpoint, ignoring auto resume"
            )

    trainer.train(load_checkpoint=config.resume_path)

    if config.local_rank == 0 or config.local_rank == -1:
        state_dict = model.custom_save_checkpoint(
            os.path.join(config.output, "checkpoints")
        )
        torch.save(
            state_dict,
            os.path.join(os.path.join(config.output, "checkpoints"), "FINAL.pt"),
        )


if __name__ == "__main__":
    config = parse_option()

    config.rank, config.local_rank, config.world_size = deepspeed_init_distributed()
    config.is_distribute = config.world_size > 1
    print(config)

    setup_logger("train", output=config.output, rank=config.rank)
    os.makedirs(config.output, exist_ok=True)
    os.makedirs(os.path.join(config.output, "checkpoints"), exist_ok=True)

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
        wandb.init(
            config=config.to_dict(),
            entity=config.entity,
            project=config.project,
            job_type=config.job_type,
            tags=config.tags,
            name=config.name,
        )
        config = ml_collections.config_dict.ConfigDict(wandb.config)

    main(config)
