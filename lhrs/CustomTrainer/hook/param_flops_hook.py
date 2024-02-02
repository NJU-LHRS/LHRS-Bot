import logging

import torch
from thop import clever_format, profile

from .logger_hook import HookBase

logger = logging.getLogger("train")


class CounterHook(HookBase):
    def __init__(self, img_size: int = 224, channel: int = 3):
        super(CounterHook, self).__init__()
        self.channel = channel
        self.img_size = img_size

    def before_train(self) -> None:
        sample_data = (
            torch.randn(1, 3, 224, 224).to(self.trainer.device),
            torch.zeros(1).to(self.trainer.device),
        )
        flops, params = profile(
            self.trainer.model,
            inputs=dict(
                x=sample_data, return_loss=False, eval=True, device=self.trainer.device
            ),
        )
        flops, params = clever_format([flops, params], "%.3f")
        split_line = "=" * 30
        logger.info(
            "{0}\nFlops: {1}\nParams: {2}\n{0}".format(split_line, flops, params)
        )
