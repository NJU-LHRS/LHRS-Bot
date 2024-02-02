import logging
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.utils import clip_grad_norm_

from .hookbase import HookBase

logger = logging.getLogger("train")


def wrap_fp16_model(model):
    """Wrap the FP32 model to FP16.

    If you are using PyTorch >= 1.6, torch.cuda.amp is used as the
    backend, otherwise, original mmcv implementation will be adopted.

    For PyTorch >= 1.6, this function will
    1. Set fp16 flag inside the model to True.

    Otherwise:
    1. Convert FP32 model to FP16.
    2. Remain some necessary layers to be FP32, e.g., normalization layers.
    3. Set `fp16_enabled` flag inside the model to True.

    Args:
        model (nn.Module): Model in FP32.
    """
    # set `fp16_enabled` flag
    for m in model.modules():
        if hasattr(m, "fp16_enabled"):
            m.fp16_enabled = True


def patch_norm_fp32(module):
    """Recursively convert normalization layers from FP16 to FP32.

    Args:
        module (nn.Module): The modules to be converted in FP16.

    Returns:
        nn.Module: The converted module, the normalization layers have been
            converted to FP32.
    """
    if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.GroupNorm)):
        module.float()
    for child in module.children():
        patch_norm_fp32(child)
    return module


class OptimizerHook(HookBase):
    """A hook contains custom operations for the optimizer.

    Args:
        grad_clip (dict, optional): A config dict to control the clip_grad.
            Default: None.
    """

    def __init__(self, grad_clip=None):
        super(OptimizerHook, self).__init__()
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad_norm_(params, self.grad_clip)

    def do_backward(self):
        self.trainer._call_hooks("after_backward")

    def do_step(self):
        self.trainer._call_hooks("after_step")

    def after_iter(self):
        self.trainer.loss_dict["total_loss"].backward()
        self.do_backward()

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(self.trainer.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                self.trainer.log(
                    self.trainer.cur_iter,
                    smooth=False,
                    grad_norm=grad_norm.detach().cpu().item(),
                )
        self.trainer.optimizer.step()
        self.trainer.optimizer.zero_grad()
        self.do_step()


class GradientCumulativeOptimizerHook(OptimizerHook):
    """Optimizer Hook implements multi-iters gradient cumulating.

    Args:
        cumulative_iters (int, optional): Num of gradient cumulative iters.
            The optimizer will step every `cumulative_iters` iters.
            Defaults to 1.

    Examples:
        >>> # Use cumulative_iters to simulate a large batch size
        >>> # It is helpful when the hardware cannot handle a large batch size.
        >>> loader = DataLoader(data, batch_size=64)
        >>> optim_hook = GradientCumulativeOptimizerHook(cumulative_iters=4)
        >>> # almost equals to
        >>> loader = DataLoader(data, batch_size=256)
        >>> optim_hook = OptimizerHook()
    """

    def __init__(self, cumulative_iters=1, **kwargs):
        super(GradientCumulativeOptimizerHook, self).__init__(**kwargs)

        assert isinstance(cumulative_iters, int) and cumulative_iters > 0, (
            f"cumulative_iters only accepts positive int, but got "
            f"{type(cumulative_iters)} instead."
        )

        self.cumulative_iters = cumulative_iters
        self.divisible_iters = 0
        self.remainder_iters = 0
        self.initialized = False

    def has_batch_norm(self, module):
        if isinstance(module, _BatchNorm):
            return True
        for m in module.children():
            if self.has_batch_norm(m):
                return True
        return False

    def _init(self):
        if self.trainer.cur_iter % self.cumulative_iters != 0:
            logger.warning(
                "Resume iter number is not divisible by cumulative_iters in "
                "GradientCumulativeOptimizerHook, which means the gradient of "
                "some iters is lost and the result may be influenced slightly."
            )

        if self.has_batch_norm(self.trainer.model) and self.cumulative_iters > 1:
            logger.warning(
                "GradientCumulativeOptimizerHook may slightly decrease "
                "performance if the model has BatchNorm layers."
            )

        residual_iters = self.trainer.max_iters - self.trainer.cur_iter

        self.divisible_iters = (
            residual_iters // self.cumulative_iters * self.cumulative_iters
        )
        self.remainder_iters = residual_iters - self.divisible_iters

        self.initialized = True

    def after_iter(self):
        if not self.initialized:
            self._init()

        if self.trainer.cur_iter < self.divisible_iters:
            loss_factor = self.cumulative_iters
        else:
            loss_factor = self.remainder_iters
        loss = self.trainer.loss_dict["total_loss"]
        loss = loss / loss_factor
        loss.backward()

        if self.every_n_iters(self.cumulative_iters) or self.is_last_iter():
            self.do_backward()

            if self.grad_clip is not None:
                grad_norm = self.clip_grads(self.trainer.model.parameters())
                if grad_norm is not None:
                    # Add grad norm to the logger
                    self.trainer.log(
                        self.trainer.cur_iter,
                        smooth=False,
                        grad_norm=grad_norm.detach().cpu().item(),
                    )
            self.trainer.optimizer.step()
            self.trainer.optimizer.zero_grad()
            self.do_step()


class Fp16OptimizerHook(OptimizerHook):
    """FP16 optimizer hook (using PyTorch's implementation).

    If you are using PyTorch >= 1.6, torch.cuda.amp is used as the backend,
    to take care of the optimization procedure.

    Args:
        loss_scale (float | str | dict): Scale factor configuration.
            If loss_scale is a float, static loss scaling will be used with
            the specified scale. If loss_scale is a string, it must be
            'dynamic', then dynamic loss scaling will be used.
            It can also be a dict containing arguments of GradScalar.
            Defaults to 512. For Pytorch >= 1.6, mmcv uses official
            implementation of GradScaler. If you use a dict version of
            loss_scale to create GradScaler, please refer to:
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler
            for the parameters.

    Examples:
        >>> loss_scale = dict(
        ...     init_scale=65536.0,
        ...     growth_factor=2.0,
        ...     backoff_factor=0.5,
        ...     growth_interval=2000
        ... )
        >>> optimizer_hook = Fp16OptimizerHook(loss_scale=loss_scale)
    """

    def __init__(
        self,
        grad_clip=None,
        coalesce=True,
        bucket_size_mb=-1,
        loss_scale=512.0,
        distributed=True,
    ):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.distributed = distributed
        self._scale_update_param = None
        if loss_scale == "dynamic":
            self.loss_scaler = GradScaler()
        elif isinstance(loss_scale, float):
            self._scale_update_param = loss_scale
            self.loss_scaler = GradScaler(init_scale=loss_scale)
        elif isinstance(loss_scale, dict):
            self.loss_scaler = GradScaler(**loss_scale)
        else:
            raise ValueError(
                "loss_scale must be of type float, dict, or "
                f'"dynamic", got {loss_scale}'
            )

    def before_train(self):
        """Preparing steps before Mixed Precision Training."""
        # wrap model mode to fp16
        wrap_fp16_model(self.trainer.model)

    def copy_grads_to_fp32(self, fp16_net, fp32_weights):
        """Copy gradients from fp16 model to fp32 weight copy."""
        for fp32_param, fp16_param in zip(fp32_weights, fp16_net.parameters()):
            if fp16_param.grad is not None:
                if fp32_param.grad is None:
                    fp32_param.grad = fp32_param.data.new(fp32_param.size())
                fp32_param.grad.copy_(fp16_param.grad)

    def copy_params_to_fp16(self, fp16_net, fp32_weights):
        """Copy updated params from fp32 weight copy to fp16 model."""
        for fp16_param, fp32_param in zip(fp16_net.parameters(), fp32_weights):
            fp16_param.data.copy_(fp32_param.data)

    def after_iter(self):
        """Backward optimization steps for Mixed Precision Training. For
        dynamic loss scaling, please refer to
        https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.

        1. Scale the loss by a scale factor.
        2. Backward the loss to obtain the gradients.
        3. Unscale the optimizer’s gradient tensors.
        4. Call optimizer.step() and update scale factor.
        5. Save loss_scaler state_dict for resume purpose.
        """
        self.loss_scaler.scale(self.trainer.loss_dict["total_loss"]).backward()
        self.loss_scaler.unscale_(self.trainer.optimizer)
        self.do_backward()
        # grad clip
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(self.trainer.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                self.trainer.log(
                    self.trainer.cur_iter,
                    smooth=False,
                    grad_norm=grad_norm.detach().cpu().item(),
                )
        # backward and update scaler
        self.loss_scaler.step(self.trainer.optimizer)
        self.loss_scaler.update(self._scale_update_param)

        self.trainer.model.zero_grad()
        self.trainer.optimizer.zero_grad()
        self.do_step()

    def state_dict(self) -> Dict[str, Any]:
        return self.loss_scaler.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self.trainer._enable_amp:
            self.loss_scaler.load_state_dict(state_dict)
        else:
            return


class GradientCumulativeFp16OptimizerHook(
    GradientCumulativeOptimizerHook, Fp16OptimizerHook
):
    """Fp16 optimizer Hook (using PyTorch's implementation) implements
    multi-iters gradient cumulating.

    If you are using PyTorch >= 1.6, torch.cuda.amp is used as the backend,
    to take care of the optimization procedure.
    """

    def __init__(self, *args, **kwargs):
        super(GradientCumulativeFp16OptimizerHook, self).__init__(*args, **kwargs)

    def after_iter(self):
        if not self.initialized:
            self._init()

        if self.trainer.cur_iter < self.divisible_iters:
            loss_factor = self.cumulative_iters
        else:
            loss_factor = self.remainder_iters
        loss = self.trainer.loss_dict["total_loss"]
        loss = loss / loss_factor

        self.loss_scaler.scale(loss).backward()

        if self.every_n_iters(self.cumulative_iters) or self.is_last_iter():

            # copy fp16 grads in the model to fp32 params in the optimizer
            self.loss_scaler.unscale_(self.trainer.optimizer)
            self.do_backward()

            if self.grad_clip is not None:
                grad_norm = self.clip_grads(self.trainer.model.parameters())
                if grad_norm is not None:
                    # Add grad norm to the logger
                    self.trainer.log(
                        self.trainer.cur_iter, grad_norm=grad_norm.detach().cpu().item()
                    )

            # backward and update scaler
            self.loss_scaler.step(self.trainer.optimizer)
            self.loss_scaler.update(self._scale_update_param)

            # save state_dict of loss_scaler
            # TODO
            # runner.meta.setdefault(
            #     'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()

            # clear grads
            self.trainer.model.zero_grad()
            self.trainer.optimizer.zero_grad()
            self.do_step()

    def state_dict(self) -> Dict[str, Any]:
        return self.loss_scaler.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self.trainer._enable_amp:
            self.loss_scaler.load_state_dict(state_dict)
        else:
            return
