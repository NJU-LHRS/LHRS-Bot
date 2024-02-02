import logging
import os
import pathlib
from typing import Dict, List, Tuple

import ml_collections
import torch
import torch.nn as nn
from deepspeed.utils.zero_to_fp32 import (
    get_fp32_state_dict_from_zero_checkpoint,
    load_state_dict_from_zero_checkpoint,
)
from peft import PeftModel

from .common_arch import AttnPooler, LayerNorm, LayerNormFp32
from .rgb_vision_modal import VisionModal
from .text_modal import TextModal

logger = logging.getLogger("train")

MODAL_MAPPING = {"rgb": VisionModal, "text": TextModal}


class UniBind(nn.Module):
    def __init__(
        self, activate_modal: Tuple[str, str], config: ml_collections.ConfigDict
    ):
        assert len(activate_modal) > 0, "activate_modal should not be empty"
        for name in activate_modal:
            assert name in MODAL_MAPPING.keys(), f"Modal {name} is not supported"
        super(UniBind, self).__init__()
        self.modal = activate_modal
        self.stage = config.stage

        if config.adjust_norm:
            norm_layer = (
                LayerNormFp32 if config.dtype in ("float16", "bfloat16") else LayerNorm
            )
        else:
            norm_layer = LayerNorm

        for modal in activate_modal:
            self.add_module(modal, MODAL_MAPPING[modal](config))

            if modal == "rgb":
                self.rgb_pooler = AttnPooler(
                    num_query=config.rgb_vision.attn_pooler.num_query,
                    num_layers=config.rgb_vision.attn_pooler.num_layers,
                    num_attention_heads=config.rgb_vision.attn_pooler.num_attn_heads,
                    encoder_hidden_size=VisionModal.EMBEDDING_DIM[
                        config.rgb_vision.arch
                    ],
                    hidden_size=VisionModal.EMBEDDING_DIM[config.rgb_vision.arch],
                    output_size=config.text.hidden_size,
                    norm_layer=norm_layer,
                    checkpoint=getattr(config, "use_checkpoint", False),
                )

    def load_rgb_encoder(self, path: str):
        assert hasattr(self, "rgb"), "rgb modal is not activated"
        ckpt = torch.load(path, map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]

        msg = self.rgb.encoder.load_state_dict(ckpt, strict=False)
        logger.info(f"Loading RGB Model: {msg}")

    def custom_save_checkpoint(self, file_name: str):
        fp32_ckpt = get_fp32_state_dict_from_zero_checkpoint(file_name)
        rgb_ckpt = get_rgb_maybe_zero_3(fp32_ckpt.items())
        other_ckpt = get_other_maybe_zero_3(fp32_ckpt.items())

        if self.stage >= 2:
            file_name = pathlib.Path(file_name)
            if file_name.is_file():
                loar_output_path = file_name.parent / "TextLoRA"
            else:
                loar_output_path = file_name / "TextLoRA"
            self.text.text_encoder.save_pretrained(str(loar_output_path))

        return dict(rgb_ckpt=rgb_ckpt, other_ckpt=other_ckpt)

    def custom_load_state_dict(self, state_dict_path, strict=False):
        if os.path.isdir(state_dict_path):
            self = load_state_dict_from_zero_checkpoint(self, state_dict_path)
            if isinstance(self.text.text_encoder, PeftModel):
                self.text.text_encoder = self.text.text_encoder.merge_and_unload()
            return None

        ckpt = torch.load(state_dict_path, map_location="cpu")
        if "model" in ckpt.keys():
            ckpt = ckpt["model"]
        text_path = pathlib.Path(state_dict_path).parent / "TextLoRA"

        logger.info(f"Loading RGB encoder.")
        msg = self.rgb.load_state_dict(ckpt["rgb_ckpt"], strict=strict)
        logger.info(
            f"After loading RGB encoder: Missing: {msg.missing_keys}. Unexpected: {msg.unexpected_keys}"
        )

        other_ckpt = ckpt["other_ckpt"]
        self.rgb_pooler.load_state_dict(other_ckpt["rgb_pooler"])
        del ckpt

        if text_path.exists():
            logger.info(f"Loadding LoRA parameters.")
            self.text.text_encoder = PeftModel.from_pretrained(
                self.text.text_encoder,
                text_path,
                is_trainable=self.stage > 2,
                torch_dtype=torch.float16,
            )

            if self.stage == 0:  # Eval
                self.text.text_encoder = self.text.text_encoder.merge_and_unload()

        return None

    def prepare_for_training(
        self,
        freeze_vision: bool = False,
        freeze_text: bool = False,
        tune_rgb_pooler: bool = False,
        model_path: str = False,
        tune_im_start: bool = False,
        compute_dtype: torch.dtype = torch.float32,
    ):
        self.train()
        for param in self.rgb.parameters():
            if freeze_vision:
                param.requires_grad = False
            else:
                param.requires_grad = True
            param.data = param.data.to(dtype=compute_dtype)

        for name, buffer in self.rgb.named_buffers():
            if "index" not in name and "id" not in name:
                buffer.data = buffer.data.to(dtype=compute_dtype)

        if freeze_text:
            self.text.eval()
            for p in self.text.get_text_encoder().get_input_embeddings().parameters():
                p.requires_grad = False

            for p in self.text.get_text_encoder().get_output_embeddings().parameters():
                p.requires_grad = False
        else:
            for p in self.text.get_text_encoder().get_input_embeddings().parameters():
                p.requires_grad = False

            for p in self.text.get_text_encoder().get_output_embeddings().parameters():
                p.requires_grad = False

        if hasattr(self, "rgb_pooler"):
            for param in self.rgb_pooler.parameters():
                if tune_rgb_pooler:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                param.data = param.data.to(dtype=compute_dtype)

        if tune_im_start:
            if freeze_text:
                for p in (
                    self.text.get_text_encoder().get_input_embeddings().parameters()
                ):
                    p.requires_grad = True

                for p in (
                    self.text.get_text_encoder().get_output_embeddings().parameters()
                ):
                    p.requires_grad = False

        if model_path is not None:
            msg = self.custom_load_state_dict(model_path)
            logger.info(f"After loading ckpt {model_path}: {msg}")

    def forward(self, data: Dict):
        out = dict()
        total_loss = 0.0

        image_embedding = self.rgb(data)

        for modal in self.modal:
            if modal == "rgb":
                image_embedding = self.rgb_pooler(image_embedding)
                continue

            output = getattr(self, modal)(
                data,
                image_embedding=image_embedding,
            )
            if modal == "text":
                text_loss = output
                total_loss += text_loss
                out.update({"text_loss": text_loss})

        out.update({"total_loss": total_loss})
        return out

    def encode_image(self, image, pool):
        assert hasattr(self, "rgb"), "rgb modal is not activated"
        image_embedding = self.rgb.encode(image)

        if hasattr(self, "rgb_pooler"):
            attn_pool = getattr(self, "rgb_pooler")
            image_embedding = attn_pool(image_embedding)

        if pool:
            return image_embedding.mean(dim=1)
        else:
            return image_embedding

    def generate(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor = None,
        do_sample: bool = True,
        temperature: float = 0.2,
        max_new_tokens: int = 1024,
        streamer=None,
        use_cache: bool = True,
        stopping_criteria=None,
        **kwargs,
    ):
        assert hasattr(self, "text"), "text modal is not activate"

        if images is not None:
            image_embedding = self.encode_image(images, pool=False)
        else:
            image_embedding = None
        return self.text.generate(
            input_ids=input_ids,
            image_embedding=image_embedding,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=use_cache,
            stopping_criteria=stopping_criteria,
            **kwargs,
        )

    def convert_weight_to_dtype(self, dtype: torch.dtype, modal: str = "rgb"):
        assert (
            modal in MODAL_MAPPING.keys()
        ), f"modal should be one of {MODAL_MAPPING.keys()}"
        assert hasattr(self, modal), f"{modal} modal is not activated"
        modality = getattr(self, modal)

        if hasattr(self, "text" + "_attn_pooling"):
            pooler = getattr(self, "text" + "_attn_pooling")

        modality.to(dtype)
        pooler.to(dtype)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logger.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_other_maybe_zero_3(named_params):
    names = ["rgb_pooler", "embed_tokens"]
    rgb_pooler = dict()
    text_proj = dict()
    embed_tokens = dict()
    lm_head = dict()

    params = list(named_params)
    to_return = dict(
        rgb_pooler=rgb_pooler,
        text_proj=text_proj,
        embed_tokens=embed_tokens,
        lm_head=lm_head,
    )
    for k, v in params:
        for name in names:
            if name in k:
                to_return[name][k.split(name + ".")[-1]] = v

    return to_return


def get_rgb_maybe_zero_3(named_params):
    to_return = {k[len("rgb.") :]: t for k, t in named_params if "rgb." in k}
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return
