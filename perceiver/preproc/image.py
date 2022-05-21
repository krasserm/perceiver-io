import torch


class ImagePreprocessor:
    def __init__(self, transform):
        self.transform = transform

    def preprocess(self, img):
        return self.transform(img)

    def preprocess_batch(self, img_batch):
        return torch.stack([self.preprocess(img) for img in img_batch])
