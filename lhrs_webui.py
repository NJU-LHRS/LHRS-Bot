import html
import os
import random
import re
from threading import Thread

import cv2
import gradio as gr
import ml_collections
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as T
import transformers
from PIL import Image
from transformers import StoppingCriteriaList, TextIteratorStreamer

from lhrs.CustomTrainer.utils import ConfigArgumentParser
from lhrs.Dataset.build_transform import build_vlp_transform
from lhrs.Dataset.conversation import default_conversation
from lhrs.models import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    build_model,
    tokenizer_image_token,
)
from lhrs.utils import StoppingCriteriaSub, type_dict

title = """<h1 align="center">LHRS-Bot🛰</h1>"""
description = """<h1 align="center">Welcome to Online LHRS-Bot Demo!</h1>"""
images_desc = """<p align="center"><img src="https://pumpkintypora.oss-cn-shanghai.aliyuncs.com/lhrsbot.png" style="height: 80px"/><p>"""
introduction = """
Using Instruction:
1. Visual Grounding: Input a description about the referring object (then choose [VG]) and CLICK **Send**.
2. Classification: Input the given categories (then choose [CLS]), and CLICK **Send**.
3. VQA: Input a visual question (then choose [VQA]) and CLICK **Send**.
4. No Tag: Input whatever you want and CLICK **Send** without any tagging
5. Identify: Input a bounding box (then choose [Identify]) and CLICK **Send**.

You can also simply chat in free form!
"""


def _get_args():
    parser = ConfigArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="Checkpoint name or path, default to %(default)r",
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run demo with CPU only"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a publicly shareable link for the interface.",
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        default=False,
        help="Automatically launch the interface in a new tab on the default browser.",
    )
    parser.add_argument(
        "--server-port", type=int, default=8000, help="Demo server port."
    )
    parser.add_argument(
        "--server-name", type=str, default="127.0.0.1", help="Demo server name."
    )

    args = parser.parse_args()
    args = ml_collections.config_dict.ConfigDict(args)
    return args


def _load_model_tokenizer(config: ml_collections.ConfigDict):
    model = build_model(config, activate_modal=("rgb", "text"))
    if config.cpu_only:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    dtype = type_dict[config.dtype]
    model.to(dtype)

    if config.checkpoint_path is not None:
        if getattr(config, "hf_model", False):
            msg = model.custom_load_state_dict(config.checkpoint_path, strict=False)
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                config.path, use_fast=False
            )
        else:
            if hasattr(model, "custom_load_state_dict"):
                msg = model.custom_load_state_dict(config.checkpoint_path)
            else:
                ckpt = torch.load(config.checkpoint_path, map_location="cpu")
                if "model" in ckpt:
                    ckpt = ckpt["model"]
                msg = model.load_state_dict(ckpt, strict=False)
                del ckpt
            tokenizer = model.text.tokenizer

    model.to(device).eval()

    return model, tokenizer


def escape_markdown(text):
    md_chars = ["<", ">"]

    for char in md_chars:
        text = text.replace(char, "\\" + char)

    return text


def reverse_escape(text):
    md_chars = ["\\<", "\\>"]

    for char in md_chars:
        text = text.replace(char, char[1:])

    return text


class WebUIDemo(object):
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: transformers.PreTrainedTokenizer,
        config: ml_collections.ConfigDict,
        stopping_criteria=None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        if config.cpu_only:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")

        self.device = device

        if getattr(config, "hf_model", False):
            self.vis_processor = self.model.get_image_processor()
        else:
            self.vis_processor = build_vlp_transform(config, is_train=False)

        self.dtype = type_dict[config.dtype]

        if stopping_criteria is not None:
            self.stopping_criteria = stopping_criteria
        else:
            stop_words_ids = [torch.tensor([2]).to(self.device)]
            self.stopping_criteria = StoppingCriteriaList(
                [StoppingCriteriaSub(stops=stop_words_ids)]
            )

    def ask(self, text, conv):
        if (
            len(conv.messages) > 0
            and conv.messages[-1][0] == conv.roles[0]
            and conv.messages[-1][1][-7:] == DEFAULT_IMAGE_TOKEN
        ):  # last message is image.
            conv.messages[-1][1] = " ".join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    def answer_prepare(
        self,
        conv,
        img_list,
        max_new_tokens=300,
        num_beams=1,
        min_length=1,
        top_p=0.95,
        repetition_penalty=1.05,
        length_penalty=1,
        temperature=1.0,
        max_length=2000,
    ):
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.device)
        )

        current_max_len = input_ids.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print(
                "Warning: The number of tokens in current conversation exceeds the max length. "
                "The model will not see the contexts outside the range."
            )
        begin_idx = max(0, current_max_len - max_length)
        input_ids = input_ids[:, begin_idx:]
        if isinstance(img_list, list):
            img_list = torch.cat(img_list, dim=0)

        generation_kwargs = dict(
            input_ids=input_ids,
            images=img_list,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=float(temperature),
        )
        return generation_kwargs

    def answer(self, conv, img_list, **kargs):
        generation_dict = self.answer_prepare(conv, img_list, **kargs)
        output_token = self.model_generate(**generation_dict)[0]
        output_text = self.tokenizer.decode(
            output_token, skip_special_tokens=True
        ).strip()

        output_text = output_text.split("<s>")[-1].stip()

        conv.messages[-1][1] = output_text
        return output_text, output_token.cpu().numpy()

    def stream_answer(self, conv, img_list, **kargs):
        generation_kwargs = self.answer_prepare(conv, img_list, **kargs)
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        generation_kwargs["streamer"] = streamer
        thread = Thread(target=self.model_generate, kwargs=generation_kwargs)
        thread.start()
        return streamer

    def model_generate(self, *args, **kwargs):
        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda" if self.device.type == "cuda" else "cpu",
                enabled=self.dtype == torch.float16 or self.dtype == torch.bfloat16,
                dtype=self.dtype,
            ):
                output = self.model.generate(*args, **kwargs)
        return output

    def encode_img(self, img_list):
        image = img_list[0]
        img_list.pop(0)
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert("RGB")
            if self.config.rgb_vision.arch.startswith("vit"):
                image = (
                    self.vis_processor(image, return_tensors="pt")
                    .pixel_values.to(self.device)
                    .to(self.dtype)
                )
            else:
                image = (
                    self.vis_processor(image)
                    .to(self.dtype)
                    .to(self.device)
                    .unsqueeze(0)
                )
        elif isinstance(image, Image.Image):
            raw_image = image
            if self.config.rgb_vision.arch.startswith("vit"):
                image = (
                    self.vis_processor(raw_image, return_tensors="pt")
                    .pixel_values.to(self.device)
                    .to(self.dtype)
                )
            else:
                image = (
                    self.vis_processor(raw_image)
                    .to(self.dtype)
                    .to(self.device)
                    .unsqueeze(0)
                )
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)

        img_list.append(image)

    def upload_img(self, image, conv, img_list):
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN)
        img_list.append(image)
        msg = "Received."

        return msg

    def launch_demo(self):
        text_input = gr.Textbox(
            placeholder="Upload your image and chat",
            interactive=True,
            show_label=False,
            container=False,
            scale=8,
        )
        with gr.Blocks() as demo:
            gr.Markdown(title)
            gr.Markdown(description)
            gr.Markdown(images_desc)

            with gr.Row():
                with gr.Column(scale=0.5):
                    image = gr.Image(type="pil", tool="sketch", brush_radius=20)

                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.5,
                        value=0.4,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )

                    clear = gr.Button("Restart")

                    gr.Markdown(introduction)

                with gr.Column():
                    chat_state = gr.State(value=None)
                    img_list = gr.State(value=[])
                    chatbot = gr.Chatbot(label="LHRS")

                    dataset = gr.Dataset(
                        components=[gr.Textbox(visible=False)],
                        samples=[
                            ["No Tag"],
                            ["[VG]"],
                            ["[CLS]"],
                            ["[VQA]"],
                            ["[Identify]"],
                        ],
                        type="index",
                        label="Task Shortcuts",
                    )
                    task_inst = gr.Markdown("**Hint:** Upload your image and chat")
                    with gr.Row():
                        text_input.render()
                        send = gr.Button("Send", variant="primary", size="sm", scale=1)

            upload_flag = gr.State(value=0)
            replace_flag = gr.State(value=0)
            image.upload(
                self.image_upload_trigger,
                [upload_flag, replace_flag, img_list],
                [upload_flag, replace_flag],
            )

            dataset.click(
                self.gradio_taskselect,
                inputs=[dataset],
                outputs=[text_input, task_inst],
                show_progress="hidden",
                postprocess=False,
                queue=False,
            )

            text_input.submit(
                self.gradio_ask,
                [
                    text_input,
                    chatbot,
                    chat_state,
                    image,
                    img_list,
                    upload_flag,
                    replace_flag,
                ],
                [text_input, chatbot, chat_state, img_list, upload_flag, replace_flag],
                queue=False,
            ).success(
                self.gradio_stream_answer,
                [chatbot, chat_state, img_list, temperature],
                [chatbot, chat_state],
            ).success(
                self.gradio_visualize,
                [chatbot, image],
                [chatbot],
                queue=False,
            )

            send.click(
                self.gradio_ask,
                [
                    text_input,
                    chatbot,
                    chat_state,
                    image,
                    img_list,
                    upload_flag,
                    replace_flag,
                ],
                [text_input, chatbot, chat_state, img_list, upload_flag, replace_flag],
                queue=False,
            ).success(
                self.gradio_stream_answer,
                [chatbot, chat_state, img_list, temperature],
                [chatbot, chat_state],
            ).success(
                self.gradio_visualize,
                [chatbot, image],
                [chatbot],
                queue=False,
            )

            clear.click(
                self.gradio_reset,
                [chat_state, img_list],
                [chatbot, image, text_input, chat_state, img_list],
                queue=False,
            )

        demo.launch(
            share=self.config.share,
            inbrowser=self.config.inbrowser,
            server_name=self.config.server_name,
            server_port=self.config.server_port,
            enable_queue=True,
        )

    def gradio_reset(self, chat_state, img_list):
        if chat_state is not None:
            chat_state.messages = []
        if img_list is not None:
            img_list = []
        return (
            None,
            gr.update(value=None, interactive=True),
            gr.update(placeholder="Upload your image and chat", interactive=True),
            chat_state,
            img_list,
        )

    def gradio_visualize(self, chatbot, gr_img):
        if isinstance(gr_img, dict):
            gr_img, mask = gr_img["image"], gr_img["mask"]

        unescaped = reverse_escape(chatbot[-1][1])
        visual_img, generation_color = visualize_all_bbox_together(gr_img, unescaped)
        if visual_img is not None:
            if len(generation_color):
                chatbot[-1][1] = generation_color
            file_path = save_tmp_img(visual_img)
            chatbot = chatbot + [[None, (file_path,)]]

        return chatbot

    def image_upload_trigger(self, upload_flag, replace_flag, img_list):
        # set the upload flag to true when receive a new image.
        # if there is an old image (and old conversation), set the replace flag to true to reset the conv later.
        upload_flag = 1
        if img_list:
            replace_flag = 1
        return upload_flag, replace_flag

    def gradio_taskselect(self, idx):
        prompt_list = [
            "",
            "[VG]",
            "[CLS] ",
            "[VQA] ",
            "[Identify] ",
        ]
        instruct_list = [
            "**Hint:** Type in whatever you want",
            "**Hint:** Send the command to generate bounding boxes",
            "**Hint:** Type in given categories, and see the classification results",
            "**Hint:** Type in a your question, and see the answer",
            "**Hint:** Type in a bounding box, and see the object",
        ]
        return prompt_list[idx], instruct_list[idx]

    def gradio_ask(
        self,
        user_message,
        chatbot,
        chat_state,
        gr_img,
        img_list,
        upload_flag,
        replace_flag,
    ):
        if len(user_message) == 0:
            text_box_show = "Input should not be empty!"
        else:
            text_box_show = ""

        if isinstance(gr_img, dict):
            gr_img, mask = gr_img["image"], gr_img["mask"]
        else:
            mask = None

        if "[Identify]" in user_message:
            integers = re.findall(r"-?\d+", user_message)
            if len(integers) != 4:
                bbox = mask2bbox(mask)
                user_message = user_message + bbox

        if chat_state is None:
            chat_state = default_conversation.copy()

        if upload_flag:
            if replace_flag:
                chat_state = default_conversation.copy()  # new image, reset everything
                replace_flag = 0
                chatbot = []
            img_list = []
            llm_message = self.upload_img(gr_img, chat_state, img_list)
            upload_flag = 0

        self.ask(user_message, chat_state)

        chatbot = chatbot + [[user_message, None]]

        if "[Identify]" in user_message:
            visual_img, _ = visualize_all_bbox_together(gr_img, user_message)
            if visual_img is not None:
                file_path = save_tmp_img(visual_img)
                chatbot = chatbot + [[(file_path,), None]]

        return text_box_show, chatbot, chat_state, img_list, upload_flag, replace_flag

    def gradio_stream_answer(self, chatbot, chat_state, img_list, temperature):
        if len(img_list) > 0:
            if not isinstance(img_list[0], torch.Tensor):
                self.encode_img(img_list)
        streamer = self.stream_answer(
            conv=chat_state,
            img_list=img_list,
            temperature=temperature,
            max_new_tokens=500,
            max_length=2000,
        )
        output = ""
        for new_output in streamer:
            escapped = escape_markdown(new_output)
            output += escapped
            chatbot[-1][1] = output
            yield chatbot, chat_state
        chat_state.messages[-1][1] = output
        return chatbot, chat_state

    def gradio_visualize(self, chatbot, gr_img):
        if isinstance(gr_img, dict):
            gr_img, mask = gr_img["image"], gr_img["mask"]

        unescaped = reverse_escape(chatbot[-1][1])
        visual_img, generation_color = visualize_all_bbox_together(gr_img, unescaped)
        if visual_img is not None:
            if len(generation_color):
                chatbot[-1][1] = generation_color
            file_path = save_tmp_img(visual_img)
            chatbot = chatbot + [[None, (file_path,)]]

        return chatbot


def extract_substrings(string):
    # first check if there is no-finished bracket
    index = string.rfind("}")
    if index != -1:
        string = string[: index + 1]

    pattern = r"\[([0-9., ]+)\]"
    matches = re.findall(pattern, string)
    pred_result = [list(map(float, match.split(","))) for match in matches if match]
    return pred_result


def is_overlapping(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)


def computeIoU(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(
        0, intersection_y2 - intersection_y1 + 1
    )
    bbox1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    bbox2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / union_area
    return iou


def save_tmp_img(visual_img):
    file_name = "".join([str(random.randint(0, 9)) for _ in range(5)]) + ".jpg"
    file_path = "/tmp/gradio" + file_name
    visual_img.save(file_path)
    return file_path


def mask2bbox(mask):
    if mask is None:
        return ""
    mask = mask.resize([100, 100], resample=Image.NEAREST)
    mask = np.array(mask)[:, :, 0]

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if rows.sum():
        # Get the top, bottom, left, and right boundaries
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox = "{{<{}><{}><{}><{}>}}".format(cmin, rmin, cmax, rmax)
    else:
        bbox = ""

    return bbox


colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (210, 210, 0),
    (255, 0, 255),
    (0, 255, 255),
    (114, 128, 250),
    (0, 165, 255),
    (0, 128, 0),
    (144, 238, 144),
    (238, 238, 175),
    (255, 191, 0),
    (0, 128, 0),
    (226, 43, 138),
    (255, 0, 255),
    (0, 215, 255),
]

color_map = {
    f"{color_id}": f"#{hex(color[2])[2:].zfill(2)}{hex(color[1])[2:].zfill(2)}{hex(color[0])[2:].zfill(2)}"
    for color_id, color in enumerate(colors)
}

used_colors = colors


def visualize_all_bbox_together(image, generation):
    if image is None:
        return None, ""

    generation = html.unescape(generation)

    image_width, image_height = image.size

    pred_list = extract_substrings(generation)
    if len(pred_list) > 0:  # it is grounding or detection
        mode = "all"
        entities = []
        i = 0
        j = 0
        for pred in pred_list:
            if len(pred) == 4:
                x0, y0, x1, y1 = (
                    pred[0],
                    pred[1],
                    pred[2],
                    pred[3],
                )
                left = x0 * image_width
                bottom = y0 * image_height
                right = x1 * image_width
                top = y1 * image_height

                entities.append([left, bottom, right, top])

                j += 1
                flag = True
            elif len(pred) > 4:
                while len(pred) != 4:
                    pred.pop()
                x0, y0, x1, y1 = (
                    pred[0],
                    pred[1],
                    pred[2],
                    pred[3],
                )
                left = x0 * image_width
                bottom = y0 * image_height
                right = x1 * image_width
                top = y1 * image_height

                entities.append([left, bottom, right, top])

                j += 1
                flag = True

        if flag:
            i += 1
    else:
        return None, ""

    if len(entities) == 0:
        return None, ""

    if isinstance(image, Image.Image):
        image_h = image.height
        image_w = image.width
        image = np.array(image)

    elif isinstance(image, str):
        if os.path.exists(image):
            pil_img = Image.open(image).convert("RGB")
            image = np.array(pil_img)[:, :, [2, 1, 0]]
            image_h = pil_img.height
            image_w = pil_img.width
        else:
            raise ValueError(f"invaild image path, {image}")
    elif isinstance(image, torch.Tensor):
        image_tensor = image.cpu()
        reverse_norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[
            :, None, None
        ]
        reverse_norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[
            :, None, None
        ]
        image_tensor = image_tensor * reverse_norm_std + reverse_norm_mean
        pil_img = T.ToPILImage()(image_tensor)
        image_h = pil_img.height
        image_w = pil_img.width
        image = np.array(pil_img)[:, :, [2, 1, 0]]
    else:
        raise ValueError(f"invaild image format, {type(image)} for {image}")

    indices = list(range(len(entities)))

    new_image = image.copy()

    previous_bboxes = []
    # size of text
    text_size = 0.5
    # thickness of text
    text_line = 1  # int(max(1 * min(image_h, image_w) / 512, 1))
    box_line = 2
    (c_width, text_height), _ = cv2.getTextSize(
        "F", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line
    )
    base_height = int(text_height * 0.675)
    text_offset_original = text_height - base_height
    text_spaces = 2

    # num_bboxes = sum(len(x[-1]) for x in entities)
    used_colors = colors  # random.sample(colors, k=num_bboxes)

    color_id = -1
    for entity_idx, entity in enumerate(entities):
        if mode == "single" or mode == "identify":
            bboxes = entity
            bboxes = [bboxes]
        else:
            bboxes = entities
        color_id += 1
        for bbox_id, (x1_norm, y1_norm, x2_norm, y2_norm) in enumerate(bboxes):
            skip_flag = False
            orig_x1, orig_y1, orig_x2, orig_y2 = (
                int(x1_norm),
                int(y1_norm),
                int(x2_norm),
                int(y2_norm),
            )

            color = used_colors[
                entity_idx % len(used_colors)
            ]  # tuple(np.random.randint(0, 255, size=3).tolist())
            new_image = cv2.rectangle(
                new_image, (orig_x1, orig_y1), (orig_x2, orig_y2), color, box_line
            )

            if mode == "all":
                l_o, r_o = (
                    box_line // 2 + box_line % 2,
                    box_line // 2 + box_line % 2 + 1,
                )

                x1 = orig_x1 - l_o
                y1 = orig_y1 - l_o

                if y1 < text_height + text_offset_original + 2 * text_spaces:
                    y1 = (
                        orig_y1
                        + r_o
                        + text_height
                        + text_offset_original
                        + 2 * text_spaces
                    )
                    x1 = orig_x1 + r_o

                # add text background
                (text_width, text_height), _ = cv2.getTextSize(
                    f"  {entity_idx}", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line
                )
                text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2 = (
                    x1,
                    y1 - (text_height + text_offset_original + 2 * text_spaces),
                    x1 + text_width,
                    y1,
                )

                for prev_bbox in previous_bboxes:
                    if (
                        computeIoU(
                            (text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2),
                            prev_bbox["bbox"],
                        )
                        > 0.95
                        and prev_bbox["phrase"] == entity_idx
                    ):
                        skip_flag = True
                        break
                    while is_overlapping(
                        (text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2),
                        prev_bbox["bbox"],
                    ):
                        text_bg_y1 += (
                            text_height + text_offset_original + 2 * text_spaces
                        )
                        text_bg_y2 += (
                            text_height + text_offset_original + 2 * text_spaces
                        )
                        y1 += text_height + text_offset_original + 2 * text_spaces

                        if text_bg_y2 >= image_h:
                            text_bg_y1 = max(
                                0,
                                image_h
                                - (
                                    text_height + text_offset_original + 2 * text_spaces
                                ),
                            )
                            text_bg_y2 = image_h
                            y1 = image_h
                            break
                if not skip_flag:
                    alpha = 0.5
                    for i in range(text_bg_y1, text_bg_y2):
                        for j in range(text_bg_x1, text_bg_x2):
                            if i < image_h and j < image_w:
                                if j < text_bg_x1 + 1.35 * c_width:
                                    # original color
                                    bg_color = color
                                else:
                                    # white
                                    bg_color = [255, 255, 255]
                                new_image[i, j] = (
                                    alpha * new_image[i, j]
                                    + (1 - alpha) * np.array(bg_color)
                                ).astype(np.uint8)

                    cv2.putText(
                        new_image,
                        f"  {entity_idx}",
                        (x1, y1 - text_offset_original - 1 * text_spaces),
                        cv2.FONT_HERSHEY_COMPLEX,
                        text_size,
                        (0, 0, 0),
                        text_line,
                        cv2.LINE_AA,
                    )

                    previous_bboxes.append(
                        {
                            "bbox": (text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2),
                            "phrase": entity_idx,
                        }
                    )

    if mode == "all":

        def color_iterator(colors):
            while True:
                for color in colors:
                    yield color

        color_gen = color_iterator(colors)

        # Add colors to phrases and remove <p></p>
        def colored_phrases(match):
            phrase = match.group(1)
            color = next(color_gen)
            return f'<span style="color:rgb{color}">{phrase}</span>'

        # generation = re.sub(r"\[([0-9., ]+)\]", "", generation)
        generation_colored = re.sub(r"\[([0-9., ]+)\]", colored_phrases, generation)
    else:
        generation_colored = ""

    pil_image = Image.fromarray(new_image)
    return pil_image, generation_colored


if __name__ == "__main__":
    random.seed(322)
    np.random.seed(322)
    torch.manual_seed(322)
    cudnn.benchmark = False
    cudnn.deterministic = True

    args = _get_args()
    model, tokenizer = _load_model_tokenizer(args)
    demo = WebUIDemo(model, tokenizer, args)
    demo.launch_demo()
