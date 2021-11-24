from typing import Callable, Optional, Union

from pl_bolts.datamodules import mnist_datamodule
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from torchvision import transforms


@DATAMODULE_REGISTRY
class MNISTDataModule(mnist_datamodule.MNISTDataModule):
    def __init__(self,
                 channels_last: bool = True,
                 random_crop: Optional[int] = None,
                 data_dir: Optional[str] = '.cache',
                 val_split: Union[int, float] = 10000,
                 num_workers: int = 3,
                 normalize: bool = True,
                 pin_memory: bool = False,
                 *args,
                 **kwargs):
        super().__init__(data_dir=data_dir,
                         val_split=val_split,
                         num_workers=num_workers,
                         normalize=normalize,
                         pin_memory=pin_memory,
                         *args,
                         **kwargs)
        self.save_hyperparameters()
        self._image_shape = super().dims

        if channels_last:
            self._image_shape = self._image_shape[1], self._image_shape[2], self._image_shape[0]

    @property
    def image_shape(self):
        return self._image_shape

    @staticmethod
    def _channel_to_last(img):
        return img.permute(1, 2, 0).contiguous()

    def default_transforms(self) -> Callable:
        transform_list = []

        if self.hparams.random_crop:
            transform_list.append(transforms.RandomCrop(self.hparams.random_crop))

        transform_list.append(transforms.ToTensor())

        if self.hparams.normalize:
            transform_list.append(transforms.Normalize(mean=(0.5, ), std=(0.5, )))

        if self.hparams.channels_last:
            transform_list.append(self._channel_to_last)

        return transforms.Compose(transform_list)
