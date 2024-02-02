from .hookbase import HookBase


class EMAHook(HookBase):
    def __init__(self):
        super(EMAHook, self).__init__()

    def before_train(self) -> None:
        self.last_step = 0

    def after_iter(self) -> None:
        if self.trainer.cur_iter > self.last_step:
            momentum_pairs = self.trainer.model_or_module.momentum_pairs
            for mp in momentum_pairs:
                self.trainer.model_or_module.ema.update(*mp)

            self.trainer.log(
                self.trainer.cur_iter, tau=self.trainer.model_or_module.ema.cur_tau
            )
            cur_step = self.trainer.cur_iter

            if self.trainer._cumulative_iters > 1:
                cur_step = cur_step * self.trainer._cumulative_iters
            self.trainer.model_or_module.ema.update_tau(
                cur_step=cur_step, max_steps=self.trainer.max_iters
            )
        self.last_step = self.trainer.cur_iter
