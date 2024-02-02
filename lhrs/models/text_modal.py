import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import ml_collections
import torch
import torch.nn.functional as F
from transformers.models.llama import LlamaForCausalLM
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast

from ..Dataset import conversation as conversation_lib
from . import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from .base_modal import BaseModal

logger = logging.getLogger("train")
type_dict = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class CustomLlamaForCausalLM(LlamaForCausalLM):
    _keep_in_fp32_modules = ["embed_tokens", "lm_head"]

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


class TextModal(BaseModal):
    def __init__(self, config: ml_collections.ConfigDict):
        """
        Large Language Model as Text encoder for image caption.
        Implementation highly based on Huggingface transformers library.

        Encoder: LLaMA-v2-7B
        """
        super(TextModal, self).__init__(config)

        self.embedding_dim = config.text.hidden_size
        self.tune_pooler = config.tune_rgb_pooler
        self.tune_im_start = config.tune_im_start
        self.tune_im_patch = config.tune_im_patch
        self.num_query = config.rgb_vision.attn_pooler.num_query

        compute_dtype = type_dict[config.dtype]
        bnb_model_from_pretrained_args = {}

        if getattr(config, "is_distribute", False):
            device = torch.device(getattr(config, "local_rank", 0))
        elif (
            "CUDA_VISABLE_DEVICES" in os.environ.keys()
            and len(os.environ["CUDA_VISABLE_DEVICES"].split(",")) == 1
        ):
            device = torch.device("cuda:" + os.environ["CUDA_VISABLE_DEVICES"])
        else:
            device = torch.device("cuda")
        if config.bits in [4, 8]:
            from transformers import BitsAndBytesConfig

            bnb_model_from_pretrained_args.update(
                dict(
                    device_map={"": device},
                    load_in_4bit=config.bits == 4,
                    load_in_8bit=config.bits == 8,
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=config.bits == 4,
                        load_in_8bit=config.bits == 8,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                        bnb_4bit_compute_dtype=compute_dtype,
                        bnb_4bit_use_double_quant=config.double_quant,
                        bnb_4bit_quant_type=config.quant_type,  # {'fp4', 'nf4'}
                    ),
                )
            )
        else:
            bnb_model_from_pretrained_args.update(
                dict(device_map={"": device}, torch_dtype=compute_dtype)
            )

        self.text_encoder = CustomLlamaForCausalLM.from_pretrained(
            config.text.path, **bnb_model_from_pretrained_args
        )

        self.tokenizer = self.init_tokenizer(config.text.path)

        if config.bits in [4, 8]:
            from peft import prepare_model_for_kbit_training

            self.text_encoder.config.torch_dtype = (
                torch.float32
                if config.fp16
                else (torch.bfloat16 if config.bf16 else torch.float32)
            )
            self.text_encoder = prepare_model_for_kbit_training(
                self.text_encoder, use_gradient_checkpointing=config.use_checkpoint
            )

        if config.lora.enable:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=config.lora.lora_r,
                lora_alpha=config.lora.lora_alpha,
                target_modules=find_all_linear_names(self.text_encoder),
                lora_dropout=config.lora.lora_dropout,
                bias=config.lora.lora_bias,
                task_type="CAUSAL_LM",
            )

            if config.bits == 16:
                if config.bf16:
                    self.text_encoder.to(torch.bfloat16)
                if config.fp16:
                    self.text_encoder.to(torch.float16)
            logger.info("Adding LoRA adapters...")
            self.text_encoder = get_peft_model(self.text_encoder, lora_config)

        if getattr(config, "use_checkpoint", False):
            self.text_encoder.gradient_checkpointing_enable()
            if hasattr(self.get_text_encoder(), "enable_input_require_grads"):
                self.get_text_encoder().enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                self.get_text_encoder().get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad
                )

        if config.bits in [4, 8]:
            from peft.tuners.lora import LoraLayer

            for name, module in self.text_encoder.named_modules():
                if isinstance(module, LoraLayer):
                    if config.bf16:
                        module = module.to(torch.bfloat16)
                if "norm" in name:
                    if config.bf16:
                        module = module.to(torch.bfloat16)
                    else:
                        module = module.to(torch.float32)
                if "lm_head" in name or "embed_tokens" in name or "cls_token" in name:
                    if hasattr(module, "weight"):
                        if config.bf16 and module.weight.dtype == torch.float32:
                            module = module.to(torch.bfloat16)
                        elif config.fp16 and module.weight.dtype == torch.float32:
                            module = module.to(torch.float16)

    def get_text_encoder(self):
        text_encoder = self.text_encoder
        while not isinstance(text_encoder, CustomLlamaForCausalLM):
            text_encoder = text_encoder.model
        return text_encoder

    def init_tokenizer(self, tokenizer_name: str):
        """
        Init tokenizer for LLaMA
        """

        tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_name)
        tokenizer.pad_token_id = tokenizer.unk_token_id

        if self.tune_im_patch:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.get_text_encoder().resize_token_embeddings(len(tokenizer))

        if self.tune_im_start:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
            self.get_text_encoder().resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = (
                    self.get_text_encoder().get_input_embeddings().weight.data
                )
                output_embeddings = (
                    self.get_text_encoder().get_output_embeddings().weight.data
                )

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if self.tune_pooler:
                for p in self.get_text_encoder().get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_text_encoder().get_output_embeddings().parameters():
                    p.requires_grad = False

        elif self.tune_im_patch:
            if self.tune_pooler:
                for p in self.get_text_encoder().get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_text_encoder().get_output_embeddings().parameters():
                    p.requires_grad = False

        return tokenizer

    def get_modal_input(self, x: Dict[str, Union[str, torch.Tensor]]) -> torch.Tensor:
        return dict(
            input_ids=x["input_ids"],
            labels=x["labels"],
            attention_mask=x["attention_mask"],
        )

    def encode(self, x: Dict) -> Dict:
        """
        Decoder Only LLaMA-v2-7B
        Parameters
        ----------
        x : input_ids, attention_mask, labels for input to the LLaMA model
        """
        return x

    def decode(
        self,
        input_ids: torch.Tensor,
        image_embedding: torch.Tensor = None,
        attention_mask: Optional[Union[torch.Tensor, None]] = None,
        labels: torch.Tensor = None,
    ) -> Tuple[torch.Tensor]:
        #    cl_loss_func: Callable = None,
        #    cl_logit_scale: torch.Tensor = None) -> Tuple[torch.Tensor]:
        (
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = self.prepare_inputs_for_multimodal(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            image_embedding=image_embedding if image_embedding is not None else None,
            past_key_values=None,
        )

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        text_loss = outputs["loss"]

        return text_loss

    def prepare_inputs_for_multimodal(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[Union[torch.Tensor, None]],
        labels: Optional[Union[torch.Tensor, None]],
        past_key_values: Optional[Union[List[torch.Tensor], None]] = None,
        image_embedding: Optional[Union[torch.Tensor, None]] = None,
    ):
        if image_embedding is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and image_embedding is not None
                and input_ids.shape[1] == 1
            ):
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            return input_ids, attention_mask, past_key_values, None, labels

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_text_encoder().model.embed_tokens(
                    cur_input_ids
                )
                cur_input_embeds = (
                    cur_input_embeds
                    + torch.zeros(
                        1,
                        self.get_text_encoder().config.hidden_size,
                        device=cur_input_embeds.device,
                        dtype=cur_input_embeds.dtype,
                    ).sum()
                )
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[
                0
            ]  # -200 That's <image>
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_embedding[
                    cur_image_idx
                ]  # cur_image_idx: batch idx
                image_token_start = image_token_indices[0]  # image index (-200)
                if getattr(self, "tune_pooler", False) and getattr(
                    self, "tune_im_start", False
                ):
                    cur_new_input_embeds.append(
                        self.get_text_encoder()
                        .model.embed_tokens(cur_input_ids[: image_token_start - 1])
                        .detach()
                    )  # before the <im_start>
                    cur_new_input_embeds.append(
                        self.get_text_encoder().model.embed_tokens(
                            cur_input_ids[image_token_start - 1 : image_token_start]
                        )
                    )  # add <im_start>
                    # add cur_image_features, it start index is image_token_start - 1.
                    # # Thus, whole image embedding indices [image_token_start - 1 : imagetoken_start - 1 + image_embedding_length + 1 + 1]
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.get_text_encoder().model.embed_tokens(
                            cur_input_ids[image_token_start + 1 : image_token_start + 2]
                        )
                    )
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_new_labels.append(
                            cur_labels[image_token_start : image_token_start + 1]
                        )
                        cur_labels = cur_labels[image_token_start + 2 :]
                else:
                    cur_new_input_embeds.append(
                        self.get_text_encoder().model.embed_tokens(
                            cur_input_ids[:image_token_start]
                        )
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_labels = cur_labels[image_token_start + 1 :]
                cur_image_idx += 1
                if getattr(self, "tune_pooler", False) and getattr(
                    self, "tune_im_start", False
                ):
                    cur_input_ids = cur_input_ids[image_token_start + 2 :]
                else:
                    cur_input_ids = cur_input_ids[image_token_start + 1 :]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

            if cur_input_ids.numel() > 0:
                if getattr(self, "tune_pooler", False) and getattr(
                    self, "tune_im_start", False
                ):
                    cur_new_input_embeds.append(
                        self.get_text_encoder()
                        .model.embed_tokens(cur_input_ids)
                        .detach()
                    )
                else:
                    cur_new_input_embeds.append(
                        self.get_text_encoder().model.embed_tokens(cur_input_ids)
                    )
                if labels is not None:
                    cur_new_labels.append(cur_labels)

            cur_new_input_embeds = [
                x.to(device=input_ids.device) for x in cur_new_input_embeds
            ]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (
                            cur_new_label,
                            torch.full(
                                (max_len - cur_new_label.shape[0],),
                                IGNORE_INDEX,
                                dtype=cur_new_label.dtype,
                                device=cur_new_label.device,
                            ),
                        ),
                        dim=0,
                    )
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(
                    attention_mask, _new_labels, new_labels
                ):
                    new_attn_mask_pad_left = torch.full(
                        (cur_new_labels.shape[0] - labels.shape[1],),
                        True,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                        False,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    cur_new_attention_mask = torch.cat(
                        (
                            new_attn_mask_pad_left,
                            cur_attention_mask,
                            new_attn_mask_pad_right,
                        ),
                        dim=0,
                    )
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (
                        attention_mask.shape[0],
                        new_input_embeds.shape[1] - input_ids.shape[1],
                    ),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    (new_attn_mask_pad_left, attention_mask), dim=1
                )
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def generate(
        self,
        image_embedding: torch.Tensor = None,
        prompt: Optional[
            Union[str, Dict]
        ] = "Describe the image in a sentence.\n<image>",  # TODO: implement prompt
        input_ids: Optional[torch.LongTensor] = None,
        do_sample: bool = True,
        temperature: float = 0.2,
        max_new_tokens: int = 1024,
        streamer=None,
        use_cache: bool = True,
        stopping_criteria=None,
        attention_mask=None,
        **kwargs,
    ):
        conv = conversation_lib.default_conversation.copy()

        if input_ids is None:
            """Caption Generation"""
            if DEFAULT_IMAGE_TOKEN in prompt:
                value = prompt.replace(DEFAULT_IMAGE_TOKEN, "").strip()
                value = DEFAULT_IMAGE_TOKEN + "\n" + value
                value = value.strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    value = value.replace(
                        DEFAULT_IMAGE_TOKEN,
                        "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>",
                    )
                replace_token = DEFAULT_IMAGE_TOKEN
                replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )
                value = value.replace(DEFAULT_IMAGE_TOKEN, replace_token)

                conv.append_message(conv.roles[0], value)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = (
                    tokenizer_image_token(
                        prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    )
                    .unsqueeze(0)
                    .expand(image_embedding.size(0), -1)
                )
                input_ids = input_ids.to(image_embedding.device)
                input_ids = self.prepare_inputs_for_multimodal(
                    input_ids=input_ids,
                    attention_mask=None,
                    labels=None,
                    image_embedding=image_embedding,
                )
                outputs = self.text_encoder.generate(inputs_embeds=input_ids[1])
                outputs = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                return outputs
        else:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_for_multimodal(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                image_embedding=image_embedding,
                past_key_values=None,
            )
            if input_ids is None:
                outputs = self.text_encoder.generate(
                    inputs_embeds=inputs_embeds,
                    do_sample=do_sample,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    streamer=streamer,
                    use_cache=use_cache,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    labels=labels,
                    stopping_criteria=stopping_criteria,
                    **kwargs,
                )
            else:
                outputs = self.text_encoder.generate(
                    input_ids=input_ids,
                    do_sample=do_sample,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    streamer=streamer,
                    use_cache=use_cache,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    labels=labels,
                    stopping_criteria=stopping_criteria,
                    **kwargs,
                )
            return outputs


def tokenizer_image_token(
    prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None
):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            lora_module_names.add(name)

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)
