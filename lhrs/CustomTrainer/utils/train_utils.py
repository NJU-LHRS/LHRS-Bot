import math
from typing import Sequence

import torch
import torch.nn as nn


def accuracy_at_k(
    outputs: torch.Tensor, targets: torch.Tensor, top_k: Sequence[int] = (1, 5)
) -> Sequence[int]:
    """Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        outputs (torch.Tensor): output of a classifier (logits or probabilities).
        targets (torch.Tensor): ground truth labels.
        top_k (Sequence[int], optional): sequence of top k values to compute the accuracy over.
            Defaults to (1, 5).

    Returns:
        Sequence[int]:  accuracies at the desired k.
    """

    with torch.no_grad():
        maxk = max(top_k)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


@torch.no_grad()
def initialize_momentum_params(online_net: nn.Module, momentum_net: nn.Module):
    params_online = online_net.parameters()
    params_momentum = momentum_net.parameters()
    for po, pm in zip(params_online, params_momentum):
        pm.data.copy_(po.data)
        pm.requires_grad = False


class MomentumUpdater:
    def __init__(self, base_tau: float = 0.996, final_tau: float = 1.0):
        super().__init__()

        assert 0 <= base_tau <= 1
        assert 0 <= final_tau <= 1 and base_tau <= final_tau

        self.base_tau = base_tau
        self.cur_tau = base_tau
        self.final_tau = final_tau

    @torch.no_grad()
    def update(self, online_net: nn.Module, momentum_net: nn.Module):
        for op, mp in zip(online_net.parameters(), momentum_net.parameters()):
            mp.data = self.cur_tau * mp.data + (1 - self.cur_tau) * op.data

    def update_tau(self, cur_step: int, max_steps: int):
        """Computes the next value for the weighting decrease coefficient tau using cosine annealing.

        Args:
            cur_step (int): number of gradient steps so far.
            max_steps (int): overall number of gradient steps in the whole training.
        """

        self.cur_tau = (
            self.final_tau
            - (self.final_tau - self.base_tau)
            * (math.cos(math.pi * cur_step / max_steps) + 1)
            / 2
        )
