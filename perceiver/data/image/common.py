import torch
from PIL import Image


class ImagePreprocessor:
    def __init__(self, transform):
        self.transform = transform

    def preprocess(self, img):
        return self.transform(img)

    def preprocess_batch(self, img_batch):
        return torch.stack([self.preprocess(img) for img in img_batch])


def lift_transform(transform):
    def apply(examples):
        examples["image"] = [transform(image) for image in examples["image"]]
        return examples

    return apply


def to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def channels_to_last(img: torch.Tensor) -> torch.Tensor:
    return img.permute(1, 2, 0).contiguous()
