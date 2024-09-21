import math
import warnings
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.distributions.normal import Normal

valid_gate_type = ("linear", "mlp")


def get_gate_network(gate_type, input_size, num_experts):
    gate_type = gate_type.lower()

    if gate_type == "linear":
        gate_network = nn.Linear(input_size, num_experts, bias=False)
        nn.init.zeros_(gate_network.weight)
    elif gate_type == "mlp":
        gate_network = torch.nn.Sequential(
            torch.nn.Linear(input_size, num_experts, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(num_experts, num_experts, bias=False),
        )
    else:
        raise ValueError(f'Expected "gate_type" in {valid_gate_type}, got {gate_type}.')

    return gate_network


class WeightNorm(nn.Module):
    def __init__(self, hidden_size: int, scale: float = 1.0, device=None, dtype=None) -> None:
        super().__init__()

        self.hsz = hidden_size
        self.scale = scale

        self.weight = nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.scale)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden * self.weight

    def extra_repr(self) -> str:
        return "hsz={}, scale={}".format(self.hsz, self.scale)


class BaseGate(nn.Module):
    def __init__(self):
        super(BaseGate, self).__init__()

    def reset_gate_network(self):
        if "gate_network_type" not in vars(self):
            raise KeyError(f"{type(self)} does not have a gate network.")
        else:
            self.gate_network = get_gate_network(self.gate_network_type, self.input_size, self.num_experts)


class TopKBalancedNoisyGate(BaseGate):
    """
    Select the top-k experts each time, with a learnable gate_network controlling expert scores.
    https://arxiv.org/abs/1701.06538.
    https://github.com/YeonwooSung/Pytorch_mixture-of-experts
    """

    def __init__(
        self,
        input_size,
        num_experts,
        num_selects,
        gate_network="mlp",
        use_softmax=True,
        use_balance=True,
        balance_loss_weight=1e-2,
        add_noise=True,
        noise_epsilon=1e-2,
    ):
        super(TopKBalancedNoisyGate, self).__init__()
        assert num_selects <= num_experts
        self.input_size = input_size
        self.num_experts = num_experts
        self.num_selects = num_selects

        self.gate_network_type = gate_network
        self.gate_network = get_gate_network(gate_network, input_size, num_experts)

        self.use_softmax = use_softmax
        self.softmax = nn.Softmax(1)

        self.use_balance = use_balance
        self.balance_loss_weight = balance_loss_weight

        # add_noise
        self.add_noise = add_noise
        self.noise_epsilon = noise_epsilon
        self.warned = False
        if self.add_noise:
            self.weight_noise = nn.Linear(input_size, num_experts, bias=False)
            self.weight_noise.weight.data = torch.zeros(
                (num_experts, input_size),
                requires_grad=True,
                device=self.weight_noise.weight.data.device,
                dtype=self.weight_noise.weight.data.dtype,
            )
            self.mean = 0.0
            self.std = 1.0
            self.normal = Normal(self.mean, self.std)
            self.softplus = nn.Softplus()

        self.reset_parameters()

    def reset_parameters(self):
        if self.add_noise:
            nn.init.zeros_(self.weight_noise.weight)

    def cv_squared(self, x, eps=1e-10):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.s
        """
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.tensor(0.0, device=x.device)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    # fmt: off
    def forward(self, x):
        logits_gate = self.gate_network(x)  
        if self.training and self.add_noise:
            noise_mm = self.weight_noise(x)
            noise_control = self.softplus(noise_mm) + self.noise_epsilon
            logits_noise = torch.randn_like(logits_gate) * noise_control  
            logits = logits_gate + logits_noise  # 最终权重
        else:
            logits = logits_gate

        top_logits, top_indices = logits.topk(min(self.num_selects + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.num_selects]
        top_k_indices = top_indices[:, :self.num_selects]
        top_k_scores = self.softmax(top_k_logits.to(torch.float32)) if self.use_softmax else top_k_logits
        top_k_scores = top_k_scores.to(logits.dtype)

        zeros = torch.zeros_like(logits, requires_grad=True, device=logits.device)
        scores_filtered = zeros.scatter(dim=1, index=top_k_indices, src=top_k_scores)  # shape(batch_size, num_experts)
        importance = scores_filtered.sum(0)  # shape(num_experts)

        if self.training:
            if self.add_noise and self.num_selects != self.num_experts:
                batch_size = top_logits.size(0)
                m = top_logits.size(1)
                top_values_flat = top_logits.flatten()
                threshold_positions_if_in = torch.arange(batch_size, device=x.device) * m + self.num_selects
                threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
                is_in = torch.gt(logits_noise, threshold_if_in)
                threshold_positions_if_out = threshold_positions_if_in - 1
                threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
                # is each value currently in the top k.
                prob_if_in = self.normal.cdf((logits_gate - threshold_if_in) / noise_control)
                prob_if_out = self.normal.cdf((logits_gate - threshold_if_out) / noise_control)
                prob = torch.where(is_in, prob_if_in, prob_if_out)
                load = prob.sum(0)
            else:
                load = (scores_filtered > 0).sum(0)
                if not self.add_noise and not self.warned:
                    warnings.warn('Gradient-trackable implementation for load calculation is only available when "add_noise=True". '
                                  'Training without noise will block the gradient from "load" path and lead to inconsistency in optimization objectives.')
                    self.warned = True
        else:
            load = (scores_filtered > 0).sum(0)

        if self.use_balance:
            balance_loss = self.cv_squared(importance) + self.cv_squared(load)
            balance_loss *= self.balance_loss_weight
        else:
            balance_loss = torch.tensor(-100.0, device=x.device)

        # print("weight", self.gate_network.weight, sep="\n")
        # print("logits_gate", logits_gate, sep="\n")
        # print("importance", importance, sep="\n")
        # print("load", load, sep="\n")
        # print("balance_loss", balance_loss, sep="\n")

        return {
            "topK_indices": top_k_indices,
            "topK_scores": top_k_scores,
            "balance_loss": balance_loss,
            "load": load,
            "importance": importance,
        }

    def forward_return_scores(self, x):
        logits_gate = self.gate_network(x)
        if self.training and self.add_noise:
            noise_mm = self.weight_noise(x)
            noise_control = self.softplus(noise_mm) + self.noise_epsilon 
            logits_noise = torch.randn_like(logits_gate) * noise_control 
            logits = logits_gate + logits_noise 
        else:
            logits = logits_gate

        scores = self.softmax(logits) if self.use_softmax else logits

        top_logits, top_indices = logits.topk(min(self.num_selects + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.num_selects]
        top_k_indices = top_indices[:, :self.num_selects]
        top_k_scores = self.softmax(top_k_logits) if self.use_softmax else top_k_logits

        zeros = torch.zeros_like(logits, requires_grad=True, device=logits.device)
        scores_filtered = zeros.scatter(dim=1, index=top_k_indices, src=top_k_scores)  # shape(batch_size, num_experts)
        importance = scores_filtered.sum(0)  # shape(num_experts)

        if self.training:
            if self.add_noise and self.num_selects != self.num_experts:
                batch_size = top_logits.size(0)
                m = top_logits.size(1)
                top_values_flat = top_logits.flatten()
                threshold_positions_if_in = torch.arange(batch_size, device=x.device) * m + self.num_selects
                threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
                is_in = torch.gt(logits_noise, threshold_if_in)
                threshold_positions_if_out = threshold_positions_if_in - 1
                threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
                # is each value currently in the top k.
                prob_if_in = self.normal.cdf((logits_gate - threshold_if_in) / noise_control)
                prob_if_out = self.normal.cdf((logits_gate - threshold_if_out) / noise_control)
                prob = torch.where(is_in, prob_if_in, prob_if_out)
                load = prob.sum(0)
            else:
                load = (scores_filtered > 0).sum(0)
                if not self.add_noise and not self.warned:
                    warnings.warn('Gradient-trackable implementation for load calculation is only available when "add_noise=True". '
                                  'Training without noise will block the gradient from "load" path and lead to inconsistency in optimization objectives.')
                    self.warned = True
        else:
            load = (scores_filtered > 0).sum(0)

        if self.use_balance:
            balance_loss = self.cv_squared(importance) + self.cv_squared(load)
            balance_loss *= self.balance_loss_weight
        else:
            balance_loss = torch.tensor(0.0, device=x.device)

        return {
            "scores": scores,
            "balance_loss": balance_loss,
            "load": load,
            "importance": importance,
        }


class BaseCalculator(nn.Module):
    def __init__(self):
        super(BaseCalculator, self).__init__()

    def reset_experts(self):
        self.experts.reset_parameters()


class UniversalCalculator(BaseCalculator):
    # traditional calculation mode, forward $num_experts$ times with re-batch optimization
    """
    https://github.com/YeonwooSung/Pytorch_mixture-of-experts
    """

    def __init__(
        self,
        experts,
        multiply_gate_scores=True,
        score_scale_factor=1.0,
        add_weight_norm: bool = False,
    ):
        super(UniversalCalculator, self).__init__()
        self.experts = experts
        self.multiply_gate_scores = multiply_gate_scores
        self.score_scale_factor = score_scale_factor
        self.num_experts = experts.num_experts
        self.mlp_norm = None
        if multiply_gate_scores and add_weight_norm:
            self.mlp_norm = WeightNorm(1, scale=score_scale_factor)
            self.mlp_norm.reset_parameters()

    def forward(self, x, topK_indices, topK_scores, expert_batch_size=None, **kwargs):
        # fmt: off
        batch_size = topK_indices.size(0)  # topK_indices: (bsz*seq_len, num_selects)
        num_selects = topK_indices.size(1)
        topK_indices = topK_indices.flatten()  # shape(batch_size*num_selects)
        topK_scores = topK_scores.flatten()  # shape(batch_size*num_selects)
        batch_indices = torch.arange(batch_size, device=topK_scores.device).repeat_interleave(num_selects)

        _, index_sorted_topK_indices = topK_indices.sort(0)

        sorted_topK_scores = topK_scores.index_select(0, index_sorted_topK_indices)
        sorted_batch_indices = batch_indices.index_select(0, index_sorted_topK_indices)

        if expert_batch_size is None:
            expert_batch_size = topK_indices.bincount(minlength=self.num_experts).tolist()

        sorted_x = x.index_select(0, sorted_batch_indices)
        split_x = torch.split(sorted_x, expert_batch_size, dim=0) 

        expert_outputs = [self.experts(split_x[i], i) for i in range(self.num_experts) if split_x[i].shape[0] > 0]

        cat_expert_outputs = torch.cat(expert_outputs, 0)
        output_dim = cat_expert_outputs.size(1)
        if self.multiply_gate_scores:
            if self.mlp_norm is None:
                cat_expert_outputs = torch.mul(cat_expert_outputs, sorted_topK_scores.reshape(-1, 1) * self.score_scale_factor)
            else:
                cat_expert_outputs = torch.mul(cat_expert_outputs, sorted_topK_scores.reshape(-1, 1))
                cat_expert_outputs = self.mlp_norm(cat_expert_outputs)

        zeros = torch.zeros((batch_size, output_dim), device=cat_expert_outputs.device, dtype=cat_expert_outputs.dtype)
        y = zeros.index_add(0, sorted_batch_indices, cat_expert_outputs)

        return {
            "hidden_states": y,
            "num_dropped_tokens": torch.tensor(-1.0),
        }


class BaseMoELayer(nn.Module):
    def __init__(self):
        super(BaseMoELayer, self).__init__()

        self.gate: Union[TopKBalancedNoisyGate,]
        self.calculator: Union[UniversalCalculator]

    def _create_gate(self, **kwargs):
        self.gate_type = kwargs.get("gate_type", "TopKBalancedNoisyGate")

        if self.gate_type == "TopKBalancedNoisyGate":  # noisy gate
            self.gate = TopKBalancedNoisyGate(
                self.input_size,
                self.num_experts,
                self.num_selects,
                gate_network=kwargs.get("gate_network", "mlp"),
                use_softmax=kwargs.get("gate_use_softmax", True),
                use_balance=kwargs.get("gate_use_balance", True),
                balance_loss_weight=kwargs.get("gate_balance_loss_weight", 1e-2),
                add_noise=kwargs.get("gate_add_noise", True),
                noise_epsilon=kwargs.get("gate_noise_epsilon", 1e-2),
            )
        else:
            raise NotImplementedError

    def _create_calculator(self, experts, **kwargs):
        self.calculator_type = kwargs.get("calculator_type", "UniversalCalculator")

        if self.calculator_type == "UniversalCalculator":  # top K calculator
            self.calculator = UniversalCalculator(
                experts,
                multiply_gate_scores=kwargs.get("multiply_gate_scores", True),
                score_scale_factor=kwargs.get("score_scale_factor", 1.0),
                add_weight_norm=kwargs.get("add_weight_norm", False),
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        original_shape = x.shape[:-1]
        x = x.reshape(-1, self.input_size)

        gate_outputs: dict = self.gate(x)
        calc_outs = self.calculator(x, **gate_outputs)
        y = calc_outs["hidden_states"]
        y = y.reshape(original_shape + (self.output_size,))

        return dict(
            hidden_states=y,
            balance_loss=gate_outputs.get("balance_loss"),
        )

    def set_num_selects(self, num_selects):
        if "num_selects" not in vars(self.gate):
            raise KeyError(f'{self.gate_type} does not have a key named "num_selects".')
        elif num_selects > self.gate.num_experts:
            raise ValueError('The value of "num_selects" must satisfy "num_selects <= num_experts"!')
        elif self.gate_type in ("SwitchBalancedGate",):
            raise ValueError(f"{self.gate_type} doesn't support manually setting num_selects.")
        else:
            self.num_selects = num_selects
            self.gate.num_selects = num_selects

    def set_gate_use_softmax(self, use_softmax):
        if "use_softmax" not in vars(self.gate):
            raise KeyError(f'{self.gate_type} does not have a key named "use_softmax".')
        else:
            self.gate.use_softmax = use_softmax

    def set_gate_use_balance(self, use_balance):
        if "use_balance" not in vars(self.gate):
            raise KeyError(f'{self.gate_type} does not have a key named "use_balance".')
        else:
            self.gate.use_balance = use_balance

    def set_gate_balance_loss_weight(self, balance_loss_weight):
        if "balance_loss_weight" not in vars(self.gate):
            raise KeyError(f'{self.gate_type} does not have a key named "balance_loss_weight".')
        else:
            self.gate.balance_loss_weight = balance_loss_weight

    def set_gate_add_noise(self, add_noise):
        if "add_noise" not in vars(self.gate):
            raise KeyError(f'{self.gate_type} does not have a key named "add_noise".')
        else:
            self.gate.add_noise = add_noise

    def set_gate_noise_epsilon(self, noise_epsilon):
        if "noise_epsilon" not in vars(self.gate):
            raise KeyError(f'{self.gate_type} does not have a key named "noise_epsilon".')
        else:
            self.gate.noise_epsilon = noise_epsilon

    def set_calculator_multiply_gate_scores(self, multiply_gate_scores):
        if "multiply_gate_scores" not in vars(self.calculator):
            raise KeyError(f'{self.gate_type} does not have a key named "multiply_gate_scores".')
        else:
            self.calculator.multiply_gate_scores = multiply_gate_scores

    def set_calculator_score_scale_factor(self, score_scale_factor):
        if "score_scale_factor" not in vars(self.calculator):
            raise KeyError(f'{self.gate_type} does not have a key named "score_scale_factor".')
        else:
            self.calculator.score_scale_factor = score_scale_factor

    def set_calculator_drop_tokens(self, drop_tokens):
        if "drop_tokens" not in vars(self.calculator):
            raise KeyError(f'{self.gate_type} does not have a key named "drop_tokens".')
        elif drop_tokens and self.calculator.dropped_padding != "zero" and self.input_size != self.output_size:
            warnings.warn(
                'Setting "drop_tokens=True" without zero dropped padding when "input_size != output_size" will cause error!'
            )
        else:
            self.calculator.drop_tokens = drop_tokens

    def set_calculator_dropped_padding(self, dropped_padding):
        if "dropped_padding" not in vars(self.calculator):
            raise KeyError(f'{self.gate_type} does not have a key named "dropped_padding".')
        elif dropped_padding not in self.calculator.available_dropped_padding_choices:
            raise ValueError(
                f"'dropped_padding' type not available! (available choices: {self.calculator.available_dropped_padding_choices})"
            )
        elif self.calculator.drop_tokens and dropped_padding != "zero" and self.input_size != self.output_size:
            warnings.warn(
                f'Setting "dropped_padding={dropped_padding}" with "drop_tokens=True" when "input_size != output_size" will cause error!'
            )
        else:
            self.calculator.dropped_padding = dropped_padding

    def set_calculator_capacity_factor(self, capacity_factor):
        if "capacity_factor" not in vars(self.calculator):
            raise KeyError(f'{self.gate_type} does not have a key named "capacity_factor".')
        else:
            self.calculator.capacity_factor = capacity_factor

    def reset_gate_network(self):
        self.gate.reset_gate_network()

    def reset_experts(self):
        self.calculator.reset_experts()


class LinearGLUMoELayer(BaseMoELayer):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_experts,
        num_selects,
        size_experts=None,
        bias=True,
        **kwargs,
    ):
        # fmt: off
        super(LinearGLUMoELayer, self).__init__()
        assert (num_selects <= num_experts)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = input_size
        self.num_experts = num_experts
        self.num_selects = num_selects
        self.size_experts = size_experts
        self.bias = bias

        experts = LinearGLUExperts(
            input_size,
            hidden_size,
            num_experts,
            bias=bias
        )

        self._create_gate(**kwargs)
        self._create_calculator(experts, **kwargs)


class LinearGLUExperts(nn.Module):
    __constants__ = [
        "bias",
        "in_features",
        "hidden_features",
        "out_features",
        "num_experts",
        "size_experts",
    ]

    def __init__(
        self,
        in_features,
        hidden_features,
        num_experts,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(LinearGLUExperts, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = hidden_features
        self.num_experts = num_experts

        self.act_fn = nn.GELU()
        self.weight_up = nn.ParameterList()
        self.weight_down = nn.ParameterList()

        for i in range(num_experts):
            # this matrix will be transposed when performing linear forwarding
            this_expert_weight_up = nn.Parameter(torch.empty((hidden_features, in_features), **factory_kwargs))
            # this matrix will be transposed when performing linear forwarding
            this_expert_weight_down = nn.Parameter(torch.empty((hidden_features, in_features), **factory_kwargs))
            self.weight_up.append(this_expert_weight_up)
            self.weight_down.append(this_expert_weight_down)

        if bias:
            self.bias_up = nn.ParameterList()
            self.bias_down = nn.ParameterList()

            for i in range(num_experts):
                this_expert_bias_up = nn.Parameter(torch.empty((hidden_features,), **factory_kwargs))
                this_expert_bias_down = nn.Parameter(torch.empty((in_features,), **factory_kwargs))
                self.bias_up.append(this_expert_bias_up)
                self.bias_down.append(this_expert_bias_down)
        else:
            self.register_parameter("bias_up", None)
            self.register_parameter("bias_down", None)

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.weight_up[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.weight_down[i], a=math.sqrt(5))
            if self.bias_up is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_up[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias_up[i], -bound, bound)
            if self.bias_down is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_down[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias_down[i], -bound, bound)

    def forward(self, input, i):
        up = F.linear(
            input,
            self.weight_up[i],
            self.bias_up[i] if self.bias_up is not None else None,
        )
        down = F.linear(
            up,
            self.weight_down[i].mT,
            self.bias_down[i] if self.bias_down is not None else None,
        )
        return down

    def extra_repr(self):
        return "in_features={}, hidden_features={}, out_features={}," " num_experts={}".format(
            self.in_features,
            self.hidden_features,
            self.out_features,
            self.num_experts,
        )
