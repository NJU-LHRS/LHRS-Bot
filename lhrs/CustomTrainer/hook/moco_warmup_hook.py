from .hookbase import HookBase


class MoCoWarmup(HookBase):
    def __init__(self, warmup_epoch: int):
        super(MoCoWarmup, self).__init__()
        self.warmup_epoch = warmup_epoch

    def before_epoch(self) -> None:
        if self.trainer.epoch >= self.warmup_epoch:
            self.trainer.model_or_module.model_warmup = False
        else:
            self.trainer.model_or_module.model_warmup = True
