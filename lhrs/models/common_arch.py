from collections import OrderedDict
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lhrs.models.moe_layers import LinearGLUMoELayer


def get_2d_sincos_pos_embed(embed_dim, image_size):
    """
    image_size: image_size or (image_height, image_width)
    return:
    pos_embed: [image_height, image_width, embed_dim]
    """
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size, grid_w_size = image_size[0], image_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[0])  # (H, W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[1])  # (H, W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=-1)  # (H, W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_new(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (H, W)
    out: (H, W, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    out = np.einsum("hw,d->hwd", pos, omega)  # (H, W, D/2), outer product

    emb_sin = np.sin(out)  # (H, W, D/2)
    emb_cos = np.cos(out)  # (H, W, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (H, W, D)
    return emb


class AttnPooler(nn.Module):
    """
    Attention Pooler

    Args:
        hidden_size: hidden size of the model
        num_layers: number of layers
        num_attention_heads: number of attention heads
        encoder_hidden_size: hidden size of the encoder
        num_query: number of query vectors
        norm_layer: normalization layer
        output_size: output size of the model
    """

    def __init__(
        self,
        num_query: int,
        num_layers: int,
        num_attention_heads: int,
        encoder_hidden_size: int,
        hidden_size: int,
        output_size: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        checkpoint: bool = False,
        stage_num: Union[List, int] = [112, 96, 64],  # [64, 48, 32]
        split_part: List = [256, 256, 256],  # [256,256, 256]
        max_size: int = 64,
        num_patches: Tuple[int, int] = (8, 8),
        use_moe: bool = False,
        num_experts: int = 1,
        num_selects: int = 1,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.num_query = num_query
        self.stage_num = stage_num
        self.split_part = split_part
        self.max_size = max_size
        self.embed_dim = hidden_size
        self.num_patches = num_patches
        self.use_moe = use_moe
        self.num_experts = num_experts

        self.query = nn.Parameter(torch.zeros(1, num_query, hidden_size))
        nn.init.trunc_normal_(self.query, std=0.02, mean=0.0)

        if encoder_hidden_size != hidden_size:
            self.in_proj = nn.Linear(encoder_hidden_size, hidden_size)
        else:
            self.in_proj = nn.Identity()

        self.layers = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    d_model=hidden_size,
                    n_head=num_attention_heads,
                    is_cross_attention=True,
                    norm_layer=norm_layer,
                    use_moe=use_moe,
                    num_experts=num_experts,
                    num_selects=num_selects,
                )
                for _ in range(num_layers)
            ]
        )

        self.layernorm_query = norm_layer(hidden_size)
        self.layernorm_kv = norm_layer(hidden_size)
        self.layernorm_post = norm_layer(hidden_size)
        self.out_proj = nn.Linear(hidden_size, output_size)

        self._set_2d_pos_embed(self.max_size)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _set_2d_pos_embed(self, max_size, device="cpu"):
        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.embed_dim, max_size)).float().to(device)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def forward(
        self,
        image_embs: torch.Tensor,
    ) -> torch.Tensor:
        image_embs = self.in_proj(image_embs)

        query_tokens = self.query.expand(image_embs.size(0), -1, -1)

        if isinstance(self.stage_num, int):
            stage1_query, stage2_query, stage3_query = torch.split(
                query_tokens, self.num_query // self.stage_num, dim=1
            )
        else:
            stage1_query, stage2_query, stage3_query = torch.split(query_tokens, self.stage_num, dim=1)

        stage1_image, stage2_image, stage3_image = torch.split(image_embs, self.split_part, dim=1)

        all_tokens = []
        pos_embed = (
            self.pos_embed[: self.num_patches[0], : self.num_patches[1], :]
            .reshape(self.num_patches[0] * self.num_patches[1], -1)
            .to(image_embs.dtype)
        )
        pos_embed = pos_embed.unsqueeze(0).expand(image_embs.size(0), -1, -1)
        pos_embed = pos_embed.permute(1, 0, 2)  # (B, L, D) -> (L, B, D)
        for sub_token, sub_image in zip(
            [stage1_query, stage2_query, stage3_query],
            [stage1_image, stage2_image, stage3_image],
        ):
            sub_token = self.layernorm_query(sub_token)
            sub_image = self.layernorm_kv(sub_image)

            sub_image = sub_image.permute(1, 0, 2)  # (B, L, D) -> (L, B, D)
            sub_token = sub_token.permute(1, 0, 2)  # (B, L, D) -> (L, B, D)

            for layer in self.layers:
                sub_token = layer(sub_token, sub_image + pos_embed, sub_image)

            sub_token = sub_token.permute(1, 0, 2)  # (L, B, D) -> (B, L, D)
            all_tokens.append(sub_token)

        query_tokens = torch.cat(all_tokens, dim=1)
        query_tokens = self.layernorm_post(query_tokens)
        out = self.out_proj(query_tokens)
        return out

    def load_state_dict(self, state_dict, **kwrags):
        msg = super().load_state_dict(state_dict, strict=False)

        if len(msg.missing_keys) > 0:
            assert self.use_moe
            layer_up_weight = "layers.{}.mlp.c_fc.weight"
            layer_up_bias = "layers.{}.mlp.c_fc.bias"
            layer_down_weight = "layers.{}.mlp.c_proj.weight"
            layer_down_bias = "layers.{}.mlp.c_proj.bias"

            for layer_idx in range(len(self.layers)):
                up_weight = state_dict[layer_up_weight.format(layer_idx)]
                up_bias = state_dict[layer_up_bias.format(layer_idx)]
                down_weight = state_dict[layer_down_weight.format(layer_idx)]
                down_bias = state_dict[layer_down_bias.format(layer_idx)]

                for expert_idx in range(self.num_experts):
                    self.layers[layer_idx].mlp.calculator.experts.weight_up[expert_idx].data = up_weight
                    self.layers[layer_idx].mlp.calculator.experts.bias_up[expert_idx].data = up_bias
                    self.layers[layer_idx].mlp.calculator.experts.weight_down[expert_idx].data = down_weight.mT
                    self.layers[layer_idx].mlp.calculator.experts.bias_down[expert_idx].data = down_bias


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.0:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        is_cross_attention: bool = False,
        use_moe: bool = False,
        num_experts: int = 1,
        num_selects: int = 1,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        self.use_moe = use_moe
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        if not use_moe:
            self.mlp = nn.Sequential(
                OrderedDict(
                    [
                        ("c_fc", nn.Linear(d_model, mlp_width)),
                        ("gelu", act_layer()),
                        ("c_proj", nn.Linear(mlp_width, d_model)),
                    ]
                )
            )
        else:
            self.mlp = LinearGLUMoELayer(d_model, mlp_width, num_experts, num_selects, gate_use_balance=False)
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))

        if not self.use_moe:
            x = x + self.ls_2(self.mlp(self.ln_2(x)))
        else:
            ln_2 = self.ln_2(x)
            mlp_out = self.mlp(ln_2)["hidden_states"]
            x = x + self.ls_2(mlp_out)

        return x


if __name__ == "__main__":
    pooler = AttnPooler(
        272,
        num_layers=6,
        num_attention_heads=8,
        encoder_hidden_size=1152,
        hidden_size=1152,
        output_size=4096,
        norm_layer=LayerNorm,
        checkpoint=False,
        split_part=[729, 729, 729],
        num_patches=[27, 27],
        use_moe=True,
        num_experts=4,
        num_selects=2,
    )

    ckpt = torch.load("/home/aiscuser/Output/LHRS/Stage1/checkpoints/FINAL.pt", map_location="cpu")

    pooler.load_state_dict(ckpt["other_ckpt"]["rgb_pooler"])

    inputs = torch.randn(1, 729 * 3, 1152)
    outputs = pooler(inputs)
    print(outputs.shape)
