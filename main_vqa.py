import json
import logging
import os
import re
from collections import defaultdict
from typing import List

import ml_collections.config_dict
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from lhrs.CustomTrainer import init_distributed
from lhrs.CustomTrainer.utils import ConfigArgumentParser, setup_logger, str2bool
from lhrs.CustomTrainer.utils.distribute import (
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
)
from lhrs.Dataset import RSVQAHR, RSVQALR, DataCollatorForVQASupervisedDataset
from lhrs.models import build_model
from lhrs.utils import type_dict
from tqdm import tqdm
from transformers import CLIPImageProcessor

logger = logging.getLogger("train")


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


def parse_option():
    parser = ConfigArgumentParser()
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    # basic
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument("--data-target", type=str, help="path to dataset annotation file ")
    parser.add_argument(
        "--data-type",
        type=str,
        choices=["LR", "HR"],
        help="VQA dataset type",
        default="HR",
    )
    parser.add_argument("--workers", type=int, default=8, help="workers of dataloader")
    parser.add_argument("--model-path", type=str, default=None, help="pretrained checkpoint path")
    parser.add_argument("--enable-amp", type=str2bool, default=False, help="mixed precision")
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--seed", type=int, default=322, help="random seed")
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )
    parser.add_argument("--gpus", type=int, default=0, help="gpus ID")
    parser.add_argument(
        "--inf_sampler",
        type=str2bool,
        default=False,
        help="Use Infinite loader if ture, else default datalodaer (Usually, inf_sampler for iterbased training)",
    )

    # wandb
    parser.add_argument("--wandb", type=str2bool, default=False, help="wandb logger")
    parser.add_argument("--entity", type=str, default="pumpkinn", help="wandb entity")
    parser.add_argument("--project", type=str, default="MaskIndexNet", help="wandb project")

    # HardWare
    parser.add_argument(
        "--accelerator",
        default="cpu",
        type=str,
        choices=["cpu", "gpu", "mps"],
        help="accelerator",
    )
    parser.add_argument("--local_rank", type=int, help="local rank")

    config = parser.parse_args(wandb=True)
    config = ml_collections.config_dict.ConfigDict(config)

    return config


def main(config: ml_collections.ConfigDict):
    logger.info(f"Creating model")
    model = build_model(config, activate_modal=("rgb", "text"))
    dtype = type_dict[config.dtype]
    model.to(dtype)

    vis_transform = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    if config.data_type == "HR":
        dataset = RSVQAHR(
            root=config.data_target,
            image_root=config.data_path,
            image_transform=vis_transform,
            split="test",
            token_prefix="<image>[VQA] ",
            prompt_type=config.prompt_template,
            tokenizer=model.text.tokenizer,
        )
    else:
        dataset = RSVQALR(
            root=config.data_target,
            image_root=config.data_path,
            image_transform=vis_transform,
            split="test",
            token_prefix="<image>[VQA] ",
            prompt_type=config.prompt_template,
            tokenizer=model.text.tokenizer,
        )
    logger.info(f"Data Length: {len(dataset)}")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=config.workers,
        pin_memory=True,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=DataCollatorForVQASupervisedDataset(model.text.tokenizer),
    )

    if config.model_path is not None:
        logger.info(f"Loading pretrained checkpoint from {config.model_path}")
        if getattr(model, "custom_load_state_dict", False):
            msg = model.custom_load_state_dict(config.model_path)
        else:
            ckpt = torch.load(config.model_path, map_location="cpu")
            msg = model.load_state_dict(ckpt["model"], strict=False)
        if msg is not None:
            logger.info(f"After loading, missing keys: {msg.missing_keys}, unexpected keys: {msg.unexpected_keys}")
            logger.info(str(model))

    if config.accelerator == "gpu":
        device = torch.device("cuda")
    else:
        device = torch.device(config.accelerator)
    model.to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for data_dict in tqdm(data_loader, unit_scale=config.batch_size, desc="Evaluating"):
            images = data_dict["images"].to(device)
            questions = data_dict["questions"].to(device)
            attention_mask = data_dict["attn_mask"].to(device)

            targets = data_dict["targets"]
            types = data_dict["types"]
            questions_idx = data_dict["questions_idx"]

            if questions.shape[0] != images.shape[0]:
                # last iter
                questions = questions[: images.shape[0]]

            with torch.autocast(
                device_type="cuda" if config.accelerator == "gpu" else "cpu",
                enabled=config.enable_amp,
                dtype=dtype,
            ):
                output_ids = model.generate(
                    input_ids=questions,
                    images=images,
                    do_sample=False,
                    num_beams=1,
                    temperature=1.0,
                    top_p=1.0,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                )

            outputs = model.text.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            outputs = [output.strip() for output in outputs]
            for pred, target, types, question_id in zip(outputs, targets, types, questions_idx):
                preds.append(dict(pred=pred, target=target, types=types, question_id=question_id))

    save_result(preds, config.output, "eval_save_file", "question_id")
    if is_main_process():
        with open(os.path.join(config.output, "eval_save_file.json")) as f:
            predictions = json.load(f)

        evaluator = TextVQAAccuracyEvaluator()
        result = evaluator.eval_pred_list(predictions)
        logger.info(f"Total: {100.0 * result}")


class EvalAIAnswerProcessor:
    CONTRACTIONS = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        "Im": "I'm",
        "Ive": "I've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "let's": "let's",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }

    NUMBER_MAP = {
        "none": "0",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    ARTICLES = ["a", "an", "the"]
    PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
    COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
    PUNCTUATIONS = [
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    ]

    def __init__(self, *args, **kwargs):
        pass

    def word_tokenize(self, word):
        word = word.lower()
        word = word.replace(",", "").replace("?", "").replace("'s", " 's")
        return word.strip()

    def process_punctuation(self, in_text):
        out_text = in_text
        for p in self.PUNCTUATIONS:
            if (p + " " in in_text or " " + p in in_text) or (re.search(self.COMMA_STRIP, in_text) is not None):
                out_text = out_text.replace(p, "")
            else:
                out_text = out_text.replace(p, " ")
        out_text = self.PERIOD_STRIP.sub("", out_text, re.UNICODE)
        return out_text

    def process_digit_article(self, in_text):
        out_text = []
        temp_text = in_text.lower().split()
        for word in temp_text:
            word = self.NUMBER_MAP.setdefault(word, word)
            if word not in self.ARTICLES:
                out_text.append(word)
            else:
                pass
        for word_id, word in enumerate(out_text):
            if word in self.CONTRACTIONS:
                out_text[word_id] = self.CONTRACTIONS[word]
        out_text = " ".join(out_text)
        return out_text

    def __call__(self, item):
        item = self.word_tokenize(item)
        item = item.replace("\n", " ").replace("\t", " ").strip()
        item = self.process_punctuation(item)
        item = self.process_digit_article(item)
        return item


class TextVQAAccuracyEvaluator:
    def __init__(self):
        self.answer_processor = EvalAIAnswerProcessor()

    def _compute_answer_scores(self, raw_answers):
        """
        compute the accuracy (soft score) of human answers
        """
        unique_answer_scores = {}
        if isinstance(raw_answers, List):
            answers = [self.answer_processor(a) for a in raw_answers]
            gt_answers = list(enumerate(answers))
            unique_answers = set(answers)
        else:
            unique_answer_scores[raw_answers] = 1
            return unique_answer_scores

        for unique_answer in unique_answers:
            accs = []
            for gt_answer in gt_answers:
                other_answers = [item for item in gt_answers if item != gt_answer]
                matching_answers = [item for item in other_answers if item[1] == unique_answer]
                acc = min(1, float(len(matching_answers)) / 3)
                accs.append(acc)
            unique_answer_scores[unique_answer] = sum(accs) / len(accs)

        return unique_answer_scores

    def eval_pred_list(self, pred_list):
        pred_scores = []
        diff_type_score = defaultdict(list)
        for entry in tqdm(pred_list):
            pred_answer = self.answer_processor(entry["pred"])
            unique_answer_scores = self._compute_answer_scores(entry["target"])
            score = unique_answer_scores.get(pred_answer, 0.0)
            if score == 0.0:
                if pred_answer in entry["target"]:
                    score = 1.0
            pred_scores.append(score)
            diff_type_score[entry["types"]].append(score)

        accuracy = sum(pred_scores) / len(pred_scores)
        for type, type_score in diff_type_score.items():
            logger.info(f"{type}: {100.0 * (sum(type_score) / len(type_score))}")
        return accuracy


if __name__ == "__main__":
    config = parse_option()

    config.rank, config.local_rank, config.world_size = init_distributed()
    config.is_distribute = config.world_size > 1
    config.adjust_norm = False
    print(config)

    setup_logger("train", output=config.output, rank=config.rank)
    os.makedirs(config.output, exist_ok=True)

    if config.is_distribute:
        seed = config.seed + dist.get_rank()
    else:
        seed = config.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    if config.rank == 0:
        path = os.path.join(config.output, "config.json")
        with open(path, "w") as f:
            configDict = dict(config.to_dict())
            json.dump(configDict, f, indent=4)
        logger.info(f"Full config saved to {path}")
        logger.info(config)

    if config.wandb and config.rank == 0:
        wandb.init(config=config.to_dict(), entity=config.entity, project=config.project)
        config = wandb.config

    main(config)
