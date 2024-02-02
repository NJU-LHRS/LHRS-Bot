import math
from typing import Callable, Dict, Tuple, Union

import ml_collections
import torch
from torch.nn import functional as F
from transformers import CLIPVisionModel

from .base_modal import BaseModal

type_dict = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def kmeans_cosine(x: torch.Tensor, K: int, max_iter: int = 5, tol: float = 1e-6):
    B, N, L = x.size()

    x_norm = x / x.norm(dim=-1, keepdim=True)

    # Randomly select K tokens from each batch as the initial centers
    init_indices = torch.randint(0, N, (B, K)).to(x.device)
    centers = F.normalize(
        torch.gather(x, 1, init_indices.view(B, K, 1).expand(B, K, L)), dim=-1
    )

    for _ in range(max_iter):
        # Calculate cosine similarity between tokens and centers
        norm_centers = centers / centers.norm(dim=-1, keepdim=True)
        similarity = x_norm @ norm_centers.transpose(1, 2)

        # Find the nearest center for each token (highest similarity)
        _, nearest_centers = torch.max(similarity, dim=2)

        # Update Centers
        new_centers = torch.zeros_like(centers)
        for b in range(B):
            for k in range(K):
                # Select all tokens that are closest to the current center
                selected_tokens = x[b][nearest_centers[b] == k]

                # Update the center to be the mean of the selected tokens and re-normalize
                if len(selected_tokens) > 0:
                    new_centers[b, k] = selected_tokens.mean(dim=0)
                else:
                    # If no tokens are assigned to a center, reinitialize it randomly
                    new_centers[b, k] = x[b, torch.randint(0, N, (1,))]

        # Check for convergence
        center_shift = torch.norm(centers - new_centers)
        if center_shift < tol:
            break
        centers = new_centers

    return centers


def do_nothing(x, mode=None):
    return x


def cls_based_bipartite_soft_matching(
    metric: torch.Tensor,
    k: int,
    cls_token: torch.Tensor,
) -> Tuple[Callable, Callable]:
    """
    Inspired and modified from: https://github.com/facebookresearch/ToMe/blob/main/tome/merge.py
    Applies Token merge according to token similarity.
    As save k most similar tokens with cls_token as seed. Merge other tokens to the seed.

    Input size is [batch, tokens, channels].
    Output size is [batch, k, channels].
    """
    assert k > 1, "k should be larger than 1"

    B, N, C = metric.shape
    assert N >= k, "number of tokens should be larger than k"

    with torch.no_grad():
        token_embeddings_norm = F.normalize(metric, p=2, dim=2)
        class_embeddings_norm = F.normalize(cls_token, p=2, dim=1).unsqueeze(1)
        similarity = torch.bmm(
            token_embeddings_norm, class_embeddings_norm.transpose(1, 2)
        ).squeeze(
            2
        )  # [B, N]

        _, top_indices = torch.topk(similarity, k=k, dim=1)  # [B, top_k]
        all_indices = (
            torch.arange(N).unsqueeze(0).expand(B, -1).to(metric.device)
        )  # [B, N]
        mask = torch.ones(B, N).to(metric.device).bool()
        mask.scatter_(1, top_indices, False)
        remaining_indices = all_indices[mask].view(B, N - k)  # [B, N - top_k]

        # Gathering the top tokens and the remaining tokens
        top_token_embeddings = torch.gather(
            metric, 1, top_indices.unsqueeze(2).expand(-1, -1, C)
        )  # [B, top_k, C]
        remaining_token_embeddings = torch.gather(
            metric, 1, remaining_indices.unsqueeze(2).expand(-1, -1, C)
        )  # [B, N - top_k, C]

        # bipartite merge
        top_tokens_norm = F.normalize(top_token_embeddings, p=2, dim=2)
        remaining_tokens_norm = F.normalize(remaining_token_embeddings, p=2, dim=2)
        cosine_similarity = torch.bmm(
            remaining_tokens_norm, top_tokens_norm.transpose(1, 2)
        )  # [B, top_k, N - top_k]

        _, dst_idx = cosine_similarity.max(dim=-1)
        dst_idx = dst_idx[..., None]

        top_token_embeddings = top_token_embeddings.scatter_reduce(
            -2, dst_idx.expand(B, N - k, C), remaining_token_embeddings, reduce="mean"
        )

        return top_token_embeddings


class VisionModal(BaseModal):
    EMBEDDING_DIM = dict(
        vit_base=768,
        vit_large=1024,
    )

    def __init__(self, config: ml_collections.ConfigDict):
        super(VisionModal, self).__init__(config)

        self.arch = config.rgb_vision.arch.lower()
        assert self.arch in [
            "vit_base",
            "vit_large",
        ], "rgb vision arch should be one of swin_base, swin_large, vit_base, vit_large"

        kwargs = {}
        if config.dtype == "int8":
            kwargs["load_in_8bit"] = True
        elif config.dtype == "int4":
            kwargs["load_in_4bit"] = True
        else:
            kwargs["torch_dtype"] = type_dict[config.dtype]

        self.embedding_dim = self.EMBEDDING_DIM[self.arch]

        if self.arch == "vit_base":
            self.encoder = CLIPVisionModel.from_pretrained(config.rgb_vision.vit_name)
            if getattr(config, "use_checkpoint", False):
                self.encoder.gradient_checkpointing_enable()

        elif self.arch == "vit_large":
            self.encoder = CLIPVisionModel.from_pretrained(config.rgb_vision.vit_name)
            if getattr(config, "use_checkpoint", False):
                self.encoder.gradient_checkpointing_enable()

        if self.arch.startswith("vit"):
            self.extract_stage = [
                self.encoder.config.num_hidden_layers // 3 - 1,
                self.encoder.config.num_hidden_layers // 3 * 2 - 1,
                self.encoder.config.num_hidden_layers - 2,
            ]

    def encode(self, x: torch.Tensor):
        if self.arch.startswith("vit"):
            outputs = self.encoder(
                x,
                return_dict=True,
                output_hidden_states=True,
            )
            if hasattr(self, "extract_stage"):
                image_embeds = []
                for idx, stage in enumerate(self.extract_stage):
                    current_hidden_states = outputs.hidden_states[stage][:, 1:, :]
                    image_embeds.append(current_hidden_states)
                image_embeds = torch.cat(image_embeds, dim=1)
                return image_embeds
            else:
                img_embeds = outputs.hidden_states[-2][:, 1:, :]
                return img_embeds

        return self.encoder.forward_embedding(x)

    def get_modal_input(self, x: Dict[str, Union[str, torch.Tensor]]) -> torch.Tensor:
        img = x["rgb"]
        return img
