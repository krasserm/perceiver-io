from PIL import Image
from torchvision.transforms import Compose
from transformers.models.perceiver.feature_extraction_perceiver import PerceiverFeatureExtractor

from perceiver.data.image.common import channels_to_last, ImagePreprocessor


class ImageNetPreprocessor(ImagePreprocessor):
    def __init__(self, crop_size: int = 256, size: int = 224, channels_last: bool = True):
        super().__init__(imagenet_valid_transform(crop_size, size, channels_last))


def imagenet_valid_transform(crop_size: int, size: int, channels_last: bool):
    extractor = PerceiverFeatureExtractor(
        do_center_crop=True,
        crop_size=crop_size,
        do_resize=True,
        size=size,
        resample=Image.BICUBIC,
        do_normalize=True,
    )

    def extract(img):
        return extractor(img, return_tensors="pt")["pixel_values"][0]

    transforms = [extract]

    if channels_last:
        transforms.append(channels_to_last)

    return Compose(transforms)
