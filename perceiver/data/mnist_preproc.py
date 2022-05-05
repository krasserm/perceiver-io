from typing import Optional

import torch
from torchvision import transforms


class MnistPreprocessor:
    def __init__(self, transform=None):
        if transform is None:
            self.transform = mnist_transform()
        else:
            self.transform = transform

    def preprocess(self, img):
        return self.transform(img)

    def preprocess_batch(self, img_batch):
        return torch.stack([self.preprocess(img) for img in img_batch])


def mnist_transform(normalize: bool = True, channels_last: bool = True, random_crop: Optional[int] = None):
    transform_list = []

    if random_crop:
        transform_list.append(transforms.RandomCrop(random_crop))

    transform_list.append(transforms.ToTensor())

    if normalize:
        transform_list.append(transforms.Normalize(mean=(0.5,), std=(0.5,)))

    if channels_last:
        transform_list.append(channels_to_last)

    return transforms.Compose(transform_list)


def channels_to_last(img: torch.Tensor):
    return img.permute(1, 2, 0).contiguous()
