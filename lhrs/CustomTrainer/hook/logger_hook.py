import datetime
import logging
import time
from typing import Dict

import ml_collections
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from .hookbase import HookBase

logger = logging.getLogger("train")


class LoggerHook(HookBase):
    """Write metrics to console and tensorboard files."""

    def __init__(
        self,
        period: int = 50,
        tb_log_dir: str = "log_dir",
        use_wandb: bool = False,
        name: str = "train",
        project: str = "MaskIndexNet",
        entity: str = "pumpkinn",
        config: ml_collections.ConfigDict = None,
        **kwargs,
    ) -> None:
        """
        Args:
            period (int): The period to write metrics. Defaults to 50.
            tb_log_dir (str): The directory to save the tensorboard files. Defaults to "log_dir".
            kwargs: Other arguments passed to ``torch.utils.tensorboard.SummaryWriter(...)``
        """
        super(LoggerHook, self).__init__()
        self._period = period
        self._tb_writer = SummaryWriter(tb_log_dir, **kwargs)
        self._last_write: Dict[str, int] = {}
        self.wandb = use_wandb
        self._name = name
        self._project = project
        self._entity = entity
        self._config = config

    def before_train(self) -> None:
        if self.wandb:
            wandb.watch(
                self.trainer.model_or_module, log="parameters", log_freq=self._period
            )

        self._train_start_time = time.perf_counter()

    def after_train(self) -> None:
        if self.trainer.deepspeed:
            return
        self._tb_writer.close()
        total_train_time = time.perf_counter() - self._train_start_time
        logger.info(
            "Total training time: {} ({} on hooks)".format(
                str(datetime.timedelta(seconds=int(total_train_time))),
            )
        )

    def after_epoch(self) -> None:
        self._write_tensorboard_wandb()

    def _write_console(self) -> None:
        # These fields ("data_time", "iter_time", "lr", "loss") may does not
        # exist when user overwrites `self.trainer.train_one_iter()`
        data_time = (
            self.metric_storage["data_time"].avg
            if "data_time" in self.metric_storage
            else None
        )
        iter_time = (
            self.metric_storage["iter_time"].avg
            if "iter_time" in self.metric_storage
            else None
        )
        lr = self.metric_storage["lr"].latest if "lr" in self.metric_storage else None

        if iter_time is not None:
            eta_seconds = iter_time * (
                self.trainer.max_iters - self.trainer.cur_iter - 1
            )
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        else:
            eta_string = None

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        loss_strings = [
            f"{key}: {his_buf.avg:.4g}"
            for key, his_buf in self.metric_storage.items()
            if "loss" in key or "acc" in key
        ]

        if hasattr(self.trainer, "epoch"):
            process_string = f"Epoch: [{self.trainer.epoch}][{self.trainer.inner_iter}/{self.trainer.epoch_len - 1}]"
        else:
            process_string = (
                f"Iter: [{self.trainer.inner_iter}/{self.trainer.max_iters}]"
            )

        tau = (
            self.metric_storage["tau"].latest
            if "tau" in self.metric_storage.keys()
            else None
        )
        grad_norm = (
            self.metric_storage["grad_norm"].latest
            if "grad_norm" in self.metric_storage.keys()
            else None
        )

        space = " " * 2
        logger.info(
            "{process}{eta}{losses}{iter_time}{data_time}{lr}{memory}{tau}{grad_norm}".format(
                process=process_string,
                eta=space + f"ETA: {eta_string}" if eta_string is not None else "",
                losses=space + "  ".join(loss_strings) if loss_strings else "",
                iter_time=(
                    space + f"iter_time: {iter_time:.4f}"
                    if iter_time is not None
                    else ""
                ),
                data_time=(
                    space + f"data_time: {data_time:.4f}  "
                    if data_time is not None
                    else ""
                ),
                lr=space + f"lr: {lr:.5g}" if lr is not None else "",
                memory=(
                    space + f"max_mem: {max_mem_mb:.0f}M"
                    if max_mem_mb is not None
                    else ""
                ),
                tau=space + f"momentum: {tau:.4f}" if tau is not None else "",
                grad_norm=(
                    space + f"grad_norm: {grad_norm: .4f}"
                    if grad_norm is not None
                    else ""
                ),
            )
        )

    def after_iter(self) -> None:
        if self.every_n_inner_iters(self._period):
            self._write_console()
            self._write_tensorboard_wandb()

    def _write_tensorboard_wandb(self):
        for key, (iter, value) in self.metric_storage.values_maybe_smooth.items():
            if key not in self._last_write or iter > self._last_write[key]:
                self._tb_writer.add_scalar(key, value, iter)
                if self.wandb:
                    wandb.log({key: value}, iter)
                self._last_write[key] = iter
