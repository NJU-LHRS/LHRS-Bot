import os
import os.path as osp
import shutil
from typing import Any, Dict, List, Optional

from ..utils.distribute import is_main_process
from .hookbase import HookBase


class CheckpointerHook(HookBase):
    """
    Save checkpoints periodically.
    """

    def __init__(self, period: int, max_to_keep: Optional[int] = None) -> None:
        super(CheckpointerHook, self).__init__()
        self._period = period
        assert max_to_keep is None or max_to_keep > 0
        self._max_to_keep = max_to_keep

        self._recent_checkpoints: List[str] = []

    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != "trainer"}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)

    def remove_exceed_ckpt(self, checkpoint_name: str):
        if self._max_to_keep is not None:
            self._recent_checkpoints.append(checkpoint_name)
            if len(self._recent_checkpoints) > self._max_to_keep:
                file_name = self._recent_checkpoints.pop(0)
                file_path = osp.join(self.trainer.ckpt_dir, file_name)
                if os.path.exists(file_path):
                    if os.path.isdir(file_path) and is_main_process():
                        shutil.rmtree(file_path)
                    else:
                        if not self.trainer.deepspeed:
                            os.remove(file_path)


class EpochCheckpointerHook(CheckpointerHook):
    """
    Save checkpoint, if current epoch is a multiple of period or ``max_epochs`` is reached.
    """

    def after_epoch(self) -> None:
        if self.every_n_epochs(self._period) or self.is_last_epoch():
            epoch = self.trainer.epoch
            if not self.trainer.deepspeed:
                checkpoint_name = f"epoch_{epoch}.pth"
            else:
                checkpoint_name = f"epoch_{epoch}"
            self.trainer.save_checkpoint(checkpoint_name)

            self.remove_exceed_ckpt(checkpoint_name)


class IterCheckpointerHook(CheckpointerHook):
    def after_iter(self) -> None:
        if self.every_n_iters(self._period) or self.is_last_iter():
            iter = self.trainer.cur_iter
            if not self.trainer.deepspeed:
                checkpoint_name = f"iter_{iter}.pth"
            else:
                checkpoint_name = f"iter_{iter}"
            self.trainer.save_checkpoint(checkpoint_name)

            self.remove_exceed_ckpt(checkpoint_name)
