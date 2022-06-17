import random

import numpy as np
from torch.utils.data.dataset import Dataset

from perceiver.data.image.common import channels_to_last
from perceiver.data.image.cutmix.utils import onehot, rand_bbox


class CutMix(Dataset):
    def __init__(
        self,
        dataset,
        num_class: int,
        channels_last: bool = False,
        num_mix: int = 1,
        beta: float = 1.0,
        prob: float = 1.0,
    ):
        self.dataset = dataset
        self.num_class = num_class
        self.channels_last = channels_last
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        img, lb = self._get_image_and_label(index)
        lb_onehot = onehot(self.num_class, lb)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            img2, lb2 = self._get_image_and_label(rand_index)
            lb2_onehot = onehot(self.num_class, lb2)

            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            lb_onehot = lb_onehot * lam + lb2_onehot * (1.0 - lam)

        if self.channels_last:
            img = channels_to_last(img)

        return {"image": img, "label": lb_onehot}

    def _get_image_and_label(self, index):
        item = self.dataset[index]
        return item["image"], item["label"]

    def __len__(self):
        return len(self.dataset)
