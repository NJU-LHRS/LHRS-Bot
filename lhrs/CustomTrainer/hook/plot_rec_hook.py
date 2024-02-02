import os

from .hookbase import HookBase


class PlotSaver(HookBase):
    def __init__(self, suffix="png", save_interval=50):
        super(PlotSaver, self).__init__()
        self.suffix = suffix
        self.save_interval = save_interval
        self.save_dir = ""

    def before_train(self) -> None:
        save_name = self.trainer.model_or_module.save_name
        self.save_dir = os.path.join(self.trainer.work_dir, save_name)
        os.makedirs(self.save_dir, exist_ok=True)

    def before_iter(self) -> None:
        if self.every_n_inner_iters(self.save_interval):
            self.trainer.model_or_module.save = True
            self.trainer.model_or_module.save_name = os.path.join(
                self.save_dir,
                "epoch{}_iter{}.{}".format(
                    self.trainer.cur_stat, self.trainer.inner_iter, self.suffix
                ),
            )
        else:
            self.trainer.model_or_module.save = False
