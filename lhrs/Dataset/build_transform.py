import ml_collections
import PIL
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from transformers import CLIPImageProcessor


def build_cls_transform(config, is_train=True):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train:
        transform = create_transform(
            input_size=config.transform.input_size,
            is_training=True,
            color_jitter=config.color_jitter,
            auto_augment=config.aa,
            interpolation="bicubic",
            re_prob=config.reprob,  # re means random erasing
            re_mode=config.remode,
            re_count=config.recount,
            mean=mean,
            std=std,
        )
        return transform

    t = []
    crop_pct = 224 / 256
    size = int(config.transform.input_size[0] / crop_pct)
    t.append(
        transforms.Resize(
            size, interpolation=PIL.Image.BICUBIC
        ),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(config.transform.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def build_vlp_transform(config: ml_collections.ConfigDict, is_train: bool = True):
    if config.rgb_vision.arch.startswith("vit"):
        return CLIPImageProcessor.from_pretrained(config.rgb_vision.vit_name)

    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train:
        transform = create_transform(
            is_training=True,
            input_size=config.transform.input_size,
            auto_augment=config.transform.rand_aug,
            interpolation="bicubic",
            mean=mean,
            std=std,
        )
        return transform

    t = []
    crop_pct = 224 / 256
    size = int(config.transform.input_size[0] / crop_pct)
    t.append(
        transforms.Resize(
            size, interpolation=PIL.Image.BICUBIC
        ),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(config.transform.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
