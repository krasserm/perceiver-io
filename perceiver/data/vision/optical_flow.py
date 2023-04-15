import itertools
import math
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from perceiver.data.vision.video_utils import write_video


class OpticalFlowProcessor:
    def __init__(self, patch_size: Tuple[int, int], patch_min_overlap: int = 20, flow_scale_factor: int = 20):
        if patch_min_overlap >= patch_size[0] or patch_min_overlap >= patch_size[1]:
            raise ValueError(
                f"Overlap should be smaller than the patch size "
                f"(patch-size='{patch_size}', patch_min_overlap='{patch_min_overlap}')."
            )

        self.patch_size = patch_size
        self.patch_min_overlap = patch_min_overlap
        self.flow_scale_factor = flow_scale_factor

    @staticmethod
    def _to_tensor(x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        raise ValueError("Invalid input type. Provide input as np.array or torch.Tensor.")

    def _preprocess(
        self,
        image_pair: Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]],
        grid_indices: List[Tuple[int, int]],
    ) -> torch.Tensor:
        img1 = self._to_tensor(image_pair[0])
        img2 = self._to_tensor(image_pair[1])

        if img1.shape != img2.shape:
            raise ValueError(f"Shapes of images must match. (shape image1='{img1.shape}', shape image2='{img2.shape}')")

        height = img1.shape[0]
        width = img1.shape[1]
        if height < self.patch_size[0]:
            raise ValueError(
                f"Height of image (height='{height}') must be at least {self.patch_size[0]}."
                "Please pad or resize your image to the minimum dimension."
            )
        if width < self.patch_size[1]:
            raise ValueError(
                f"Width of image (width='{width}') must be at least {self.patch_size[1]}."
                "Please pad or resize your image to the minimum dimension."
            )

        img1 = self._transform(img1)
        img2 = self._transform(img2)
        transformed_image_pair = torch.stack([img1, img2], dim=0)

        patch_input_feature_batch = []
        for y, x in grid_indices:
            image_pair_patch = transformed_image_pair[..., y : y + self.patch_size[0], x : x + self.patch_size[1]]
            patch_input_features = self._extract_image_patches(image_pair_patch, kernel=3).float()
            patch_input_feature_batch.append(patch_input_features)

        return torch.stack(patch_input_feature_batch, dim=0)

    def _transform(self, img: torch.Tensor) -> torch.Tensor:
        x = self._normalize(img).to(torch.float32)

        if x.shape[-1] == 3:  # check if channels are at last position
            x = rearrange(x, "h w c -> c h w")
        return x

    @staticmethod
    def _normalize(img: torch.Tensor) -> torch.Tensor:
        return img / 255.0 * 2 - 1

    def _extract_image_patches(self, x: torch.Tensor, kernel: int, stride: int = 1, dilation: int = 1):
        """Equivalent to the implementation of https://www.tensorflow.org/api_docs/python/tf/image/extract_patches
        using "SAME" padding.

        From: https://discuss.pytorch.org/t/tf-extract-image-patches-in-pytorch/43837/9
        """
        b = x.shape[0]
        x = self._pad(x, kernel, stride, dilation)
        # extract patches
        patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
        # re-order patch dimensions
        patches = patches.permute(0, 4, 5, 1, 2, 3).contiguous()
        # stack patches along second dimension
        return patches.view(b, -1, patches.shape[-2], patches.shape[-1])

    @staticmethod
    def _pad(x: torch.Tensor, kernel: int, stride: int = 1, dilation: int = 1) -> torch.Tensor:
        """Applies a pad to the input using "SAME" strategy."""
        *_, h, w = x.shape
        h2 = math.ceil(h / stride)
        w2 = math.ceil(w / stride)
        pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
        pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
        return F.pad(x, (pad_col // 2, pad_col - pad_col // 2, pad_row // 2, pad_row - pad_row // 2))

    def _compute_patch_grid_indices(self, img_shape: Tuple[int, ...]) -> List[Tuple[int, int]]:
        """From https://github.com/deepmind/deepmind-research/blob/master/perceiver/colabs/optical_flow.ipynb."""
        ys = list(range(0, img_shape[0], self.patch_size[0] - self.patch_min_overlap))
        xs = list(range(0, img_shape[1], self.patch_size[1] - self.patch_min_overlap))
        ys[-1] = img_shape[0] - self.patch_size[0]
        xs[-1] = img_shape[1] - self.patch_size[1]
        return list(itertools.product(ys, xs))

    def preprocess(
        self, image_pair: Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """Creates the input features for the model for a pair of images.

        The input images are stacked and split into image patches of size `patch_size`. For each pixel of each
        individual patch, 3x3 patches are extracted and stacked in the channel dimension.

        Output shape: torch.Size(nr_patches, 2, 27, patch_size[0], patch_size[1])
        """
        grid_indices = self._compute_patch_grid_indices(image_pair[0].shape)
        return self._preprocess(image_pair, grid_indices)

    def preprocess_batch(
        self,
        image_pairs: Union[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[torch.Tensor, torch.Tensor]]],
    ) -> torch.Tensor:
        """Creates the input features for the model for a batch of image pairs.

        For each image pair the images are stacked and split into image patches of size `patch_size`. For each pixel
        of each individual patch, 3x3 patches are extracted and stacked in the channel dimension.

        Output shape: torch.Size(batch_size, nr_patches, 2, 27, patch_size[0], patch_size[1])
        """
        grid_indices = self._compute_patch_grid_indices(image_pairs[0][0].shape)
        return self._preprocess_batch(image_pairs, grid_indices)

    def _preprocess_batch(
        self,
        image_pairs: Union[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[torch.Tensor, torch.Tensor]]],
        grid_indices: List[Tuple[int, int]],
    ) -> torch.Tensor:
        shapes = []
        for image1, image2 in image_pairs:
            shapes += [image1.shape, image2.shape]

        if not np.all(np.array(shapes) == shapes[0]):
            raise ValueError("Shapes of images must match. Not all input images have the same shape.")

        return torch.stack([self._preprocess(image_pair, grid_indices) for image_pair in image_pairs], dim=0)

    def postprocess(self, predictions: torch.Tensor, img_shape: Tuple[int, ...]) -> torch.Tensor:
        """Combines optical flow predictions for individual image patches into a single prediction per image pair.

        Predictions can be supplied for a single image pair or a batch of image pairs, hence the supported input shapes
        are:
        * (nr_patches, patch_size[0], patch_size[1], 2) and
        * (batch_size, nr_patches, patch_size[0], patch_size[1], 2).

        Returns combined predictions for each supplied image pair.

        Output shape: (batch_size, height, width, 2)
        """

        flow_batch = []
        height = img_shape[0]
        width = img_shape[1]

        grid_indices = self._compute_patch_grid_indices(img_shape)
        prediction_batch = predictions.unsqueeze(0).cpu() if predictions.dim() == 4 else predictions.cpu()

        b, p, *_ = prediction_batch.shape
        if p != len(grid_indices):
            raise ValueError(
                f"Number of patches in the input does not match the number of calculated patches based "
                f"on the supplied image size (nr_patches='{p}', calculated={len(grid_indices)})."
            )

        for prediction in prediction_batch:
            flow = torch.zeros(1, height, width, 2).type(torch.float32)
            flow_weights = torch.zeros(1, height, width, 1).type(torch.float32)

            for flow_patch, (y, x) in zip(prediction, grid_indices):
                flow_patch = flow_patch * self.flow_scale_factor

                weights_y, weights_x = torch.meshgrid(
                    torch.arange(self.patch_size[0]), torch.arange(self.patch_size[1]), indexing="ij"
                )
                weights_x = torch.minimum(torch.add(weights_x, 1), self.patch_size[1] - weights_x)
                weights_y = torch.minimum(torch.add(weights_y, 1), self.patch_size[0] - weights_y)
                weights = rearrange(torch.minimum(weights_x, weights_y), "h w -> 1 h w 1")

                pad = (0, 0, x, width - x - self.patch_size[1], y, height - y - self.patch_size[0], 0, 0)
                flow += F.pad(flow_patch * weights, pad, "constant", 0)
                flow_weights += F.pad(weights, pad, "constant", 0)

            flow /= flow_weights
            flow_batch.append(flow)

        return torch.concat(flow_batch, dim=0)

    def process(
        self,
        model,
        image_pairs: Union[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[torch.Tensor, torch.Tensor]]],
        batch_size: int,
    ) -> torch.Tensor:
        """Combines preprocessing, inference and postprocessing steps for the optical flow.

        The input features for model are created by stacking each image pair in the channel dimension and splitting the
        result into image patches of size `patch_size`. For each pixel in each individual patch, 3x3 patches
        are extracted and stacked in the channel dimension.

        The input is processed using the supplied optical flow model and the optical flow predictions per image pair
        are returned.

        Output shape: (batch_size, height, width, 2)
        """

        image_shape = image_pairs[0][0].shape
        grid_indices = self._compute_patch_grid_indices(image_shape)

        predictions = []
        with torch.no_grad():
            for i in tqdm(range(0, len(image_pairs), batch_size)):
                input_features_batch = self._preprocess_batch(image_pairs[i : i + batch_size], grid_indices)
                input_features_batch = rearrange(input_features_batch, "b p t c h w -> (b p) t c h w")
                for j in range(0, input_features_batch.shape[0], batch_size):
                    input_features_micro_batch = input_features_batch[j : (j + batch_size)]
                    pred = model(input_features_micro_batch)
                    predictions.append(pred.cpu().detach())

        flow_predictions = torch.concat(predictions, dim=0)
        flow_predictions = rearrange(flow_predictions, "(b p) h w c -> b p h w c", b=len(image_pairs))
        return self.postprocess(flow_predictions, image_shape)


def render_optical_flow(flow: np.ndarray) -> np.ndarray:
    """Renders optical flow predictions produced by an optical flow model."""
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv[..., 0] = ang / np.pi / 2 * 180
    hsv[..., 1] = np.clip(mag * 255 / 24, 0, 255)
    hsv[..., 2] = 255

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def write_optical_flow_video(video_path: Path, frames: List[torch.Tensor], fps: int = 30) -> None:
    """Writes optical flow predictions as individual frames to a video file."""
    write_video(video_path=video_path, frames=[render_optical_flow(f.numpy()) for f in frames], fps=fps)
