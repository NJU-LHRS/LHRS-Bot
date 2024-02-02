import json
import logging
import os

import numpy as np
import torch
import torch.distributed as dist
import wandb
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

from ..utils import MetricStroge, accuracy_at_k
from ..utils.distribute import get_rank, get_world_size, is_distributed, is_main_process
from .hookbase import HookBase
from .knn_eval_hook import MetricLogger
from .logger_hook import LoggerHook

logger = logging.getLogger("train")


class EvalHook(HookBase):
    """Run an evaluation function periodically.

    It is executed every ``period`` epochs and after the last epoch.
    """

    def __init__(self, period: int, task: str = "cls"):
        """
        Args:
            period (int): The period to run ``eval_func``. Set to 0 to
                not evaluate periodically (but still after the last iteration).
            eval_func (callable): A function which takes no arguments, and
                returns a dict of evaluation metrics.
        """
        super(EvalHook, self).__init__()
        self._period = period
        if task == "cls":
            self._eval_func = self._eval_func_cls
            self.max_acc = None
        elif task == "cap":
            self._eval_func = self._eval_func_cap
            self.best_bleu4 = None

    @torch.no_grad()
    def _eval_func_cls(self):
        self.trainer.model.eval()
        metric = MetricStroge(window_size=len(self.trainer.eval_loader))

        for idx, batch in enumerate(self.trainer.eval_loader):
            images = batch[0]
            target = batch[-1].to(self.trainer.device, non_blocking=True)

            with torch.autocast(
                device_type=self.trainer.autocast_type, enabled=self.trainer._enable_amp
            ):
                result = self.trainer.model(
                    batch, self.trainer.device, return_loss=False, eval=True
                )

            acc1, acc5 = accuracy_at_k(result, target, top_k=(1, 5))
            metric.update(acc1=acc1.detach().cpu().numpy()[0], smooth=False)
            metric.update(acc5=acc5.detach().cpu().numpy()[0], smooth=False)

        acc1 = metric["acc1"].global_avg
        acc5 = metric["acc5"].global_avg

        logger.info(
            "Epoch: {0}\tacc@1: {1:.4f}\tacc@5: {2:.4f}".format(
                self.trainer.cur_stat, acc1, acc5
            )
        )

        if self.max_acc is None:
            self.max_acc = acc1
        else:
            if acc1 > self.max_acc:
                self.max_acc = acc1
                self.trainer.save_checkpoint(
                    f"best_ckpt_{str(self.trainer.cur_stat)}_{str(acc1)}.pth"
                )

        logger.info(f"Max accuracy: {self.max_acc:.2f}%")

    @torch.no_grad()
    def _eval_func_cap(self):
        self.trainer.model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        header = "Caption Evaluation:"

        result = []
        for batch in metric_logger.log_every(self.trainer.eval_data_loader, 10, header):
            with torch.autocast(
                device_type=self.trainer.autocast_type,
                enabled=self.trainer._enable_amp,
                dtype=self.trainer.dtype,
            ) as autocast, torch.backends.cuda.sdp_kernel(
                enable_flash=False
            ) as disable:
                pred_cap, true_cap = self.trainer.model_or_module.caption_generate(
                    data=batch, device=self.trainer.device
                )

            img_ids = batch["img_id"]
            filenames = batch["filename"]
            for idx, cap, filename in zip(img_ids, pred_cap, filenames):
                result.append(
                    {"image_id": idx.item(), "caption": cap, "filename": filename}
                )

        save_result(result, self.trainer.work_dir, "eval_save_file", "image_id")
        if is_main_process():
            coco = COCO(self.trainer.eval_data_loader.dataset.json_dir, custom=True)
            cocoRes = coco.loadRes(
                os.path.join(self.trainer.work_dir, "eval_save_file.json")
            )
            coco_eval = COCOEvalCap(coco, cocoRes)
            coco_eval.evaluate()

            logger.info("Global stats:")
            for hooks in self.trainer._hooks:
                if isinstance(hooks, LoggerHook):
                    _tb_writer = hooks._tb_writer
                    use_wandb = hooks.wandb

            for metric, score in coco_eval.eval.items():
                logger.info(f"{metric}: {score:.4f}")
                _tb_writer.add_scalar(f"eval/{metric}", score, self.trainer.cur_stat)
                if use_wandb:
                    wandb.log({metric: score})

            BLEU_avg = coco_eval.eval["Bleu_4"]

            if self.best_bleu4 is None:
                self.best_bleu4 = BLEU_avg
            else:
                if BLEU_avg > self.best_bleu4:
                    self.best_bleu4 = BLEU_avg
                    self.trainer.save_checkpoint(
                        f"best_ckpt_{str(self.trainer.cur_stat)}_{str(np.around(BLEU_avg, 2))}.pth"
                    )

            logger.info(f"Best blue4: {self.best_bleu4:.2f}%")


class EpochEvalHook(EvalHook):
    def after_epoch(self):
        if self.every_n_epochs(self._period) or self.is_last_epoch():
            self._eval_func()


class IterEvalHook(EvalHook):
    def after_iter(self):
        if self.every_n_iters(self._period) or self.is_last_iter():
            self._eval_func()


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
