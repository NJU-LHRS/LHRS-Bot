import logging
from typing import List

from .hook import (
    CosineAnnealingLrUpdaterHook,
    FixedLrUpdaterHook,
    HookBase,
    IterCheckpointerHook,
    LoggerHook,
)
from .trainer import Trainer
from .utils import collect_env, is_main_process

logger = logging.getLogger("train")


class IterBasedTrainer(Trainer):
    def __init__(self, max_iters: int, **kwargs):
        """
        Args:
            max_iters (int): Total training iterations.
        """
        super().__init__(**kwargs)
        self.target_iters = max_iters

        self.begin = 0

        if is_main_process() or self.deepspeed:
            self.register_hook(self._build_default_hook())
            logger.info(
                f"Registered default hooks for main process: {self.registered_hook_names}"
            )

        logger.info("Environment info:\n" + collect_env())

    @property
    def cur_stat(self) -> int:
        return self.inner_iter

    @property
    def max_iters(self) -> int:
        return self.target_iters

    @property
    def cur_iter(self) -> int:
        return self.inner_iter

    @property
    def start_iter(self) -> int:
        return self.begin

    def _build_default_hook(self) -> List[HookBase]:
        return [
            self.build_ckpt_hook(),
            LoggerHook(
                self._log_period, tb_log_dir=self.tb_log_dir, use_wandb=self.wandb
            ),
        ]

    def load_cur_stat(self, value):
        self.inner_iter = value
        self.begin = value

    def get_specific_hooks(self) -> List[HookBase]:
        if self.lr_scheduler.name == "cosine":
            lr_scheduler = CosineAnnealingLrUpdaterHook(
                by_epoch=False,
                warmup=self.lr_scheduler.warmup_method,
                warmup_ratio=self.lr_scheduler.warmup_factor,
                warmup_by_epoch=False,
                min_lr=self.lr_scheduler.min_lr,
                warmup_iters=self.lr_scheduler.warmup_epochs,
            )
        elif self.lr_scheduler.name == "const":
            lr_scheduler = FixedLrUpdaterHook()
        else:
            raise NotImplementedError(
                f"Unsupported lr scheduler: {self.lr_scheduler.name}"
            )

        return [lr_scheduler]

    def sub_classes_train(self):
        logger.info(
            f"Start training from iteration {self.inner_iter} to {self.target_iters}"
        )
        self.model.train()
        for self.inner_iter in range(self.start_iter, self.target_iters):
            self._call_hooks("before_iter")
            self.train_on_iter()
            self._call_hooks("after_iter")
