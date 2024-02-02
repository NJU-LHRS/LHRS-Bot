from io import BytesIO

import ml_collections
import requests
import torch
import transformers
from PIL import Image
from transformers import TextStreamer

from lhrs.CustomTrainer.utils import ConfigArgumentParser, str2bool
from lhrs.Dataset.build_transform import build_vlp_transform
from lhrs.Dataset.conversation import SeparatorStyle, default_conversation
from lhrs.models import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    build_model,
    tokenizer_image_token,
)
from lhrs.utils import KeywordsStoppingCriteria, type_dict


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def parse_option():
    parser = ConfigArgumentParser()
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    # basic
    parser.add_argument("--image-file", type=str, help="path to image")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="pretrained checkpoint path for vision encoder",
    )
    parser.add_argument("--seed", type=int, default=322, help="random seed")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")

    # HardWare
    parser.add_argument(
        "--accelerator",
        default="cpu",
        type=str,
        choices=["cpu", "gpu", "mps"],
        help="accelerator",
    )
    parser.add_argument("--use-checkpoint", default=False, type=str2bool)

    config = parser.parse_args(wandb=True)
    config = ml_collections.config_dict.ConfigDict(config)

    return config


def main(config: ml_collections.ConfigDict):
    model = build_model(config, activate_modal=("rgb", "text"))
    if getattr(config, "hf_model", False):
        vision_processor = model.get_image_processor()
    else:
        vision_processor = build_vlp_transform(config, is_train=False)
    dtype = type_dict[config.dtype]
    model.to(dtype)

    conv = default_conversation.copy()
    roles = conv.roles

    if config.model_path is not None:
        if getattr(config, "hf_model", False):
            msg = model.custom_load_state_dict(config.model_path, strict=False)
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                config.path, use_fast=False
            )
        else:
            if hasattr(model, "custom_load_state_dict"):
                msg = model.custom_load_state_dict(config.model_path)
            else:
                ckpt = torch.load(config.model_path, map_location="cpu")
                if "model" in ckpt:
                    ckpt = ckpt["model"]
                msg = model.load_state_dict(ckpt, strict=False)
                del ckpt
            tokenizer = model.text.tokenizer
        print(msg)

    if config.accelerator == "gpu":
        device = torch.device("cuda")
    else:
        device = torch.device(config.accelerator)
    model.to(device)

    if config.image_file is not None:
        image = load_image(config.image_file)
        if config.rgb_vision.arch.startswith("vit"):
            image_tensor = (
                vision_processor(image, return_tensors="pt")
                .pixel_values.to(device)
                .to(dtype)
            )
        else:
            image_tensor = vision_processor(image).to(dtype).to(device).unsqueeze(0)
    else:
        image = None
        image_tensor = None

    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if config.tune_im_start:
                inp = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + inp
                )
            else:
                inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(device)
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                max_new_tokens=512,
                temperature=0.4,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = tokenizer.decode(output_ids[0]).strip()
        outputs = outputs.split("<s>")[-1].strip()  # remove <s>
        conv.messages[-1][-1] = outputs

        if config.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    config = parse_option()
    config.adjust_norm = False
    main(config)
