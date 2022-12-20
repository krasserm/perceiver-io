import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat


def positions(b, n, shift: Optional[torch.Tensor] = None, device: Optional[torch.device] = None):
    pos = repeat(torch.arange(n, device=device), "n -> b n", b=b)

    if shift is not None:
        if shift.shape != (b, 1):
            raise ValueError(f"shift must have shape {(1, b)} but has shape {shift.shape}")
        pos = pos - shift

    return torch.clamp(pos, min=0)


class RotaryPositionEmbedding:
    # Specified in https://arxiv.org/abs/2104.09864
    # Modified from https://github.com/lucidrains/rotary-embedding-torch
    def __init__(self, frq_pos_enc: torch.Tensor, right_align: bool = False):
        # frq_pos_enc shape is (b, n, c).
        # frq_pos_enc is broadcast to (b, h, n, c).
        self.frq_pos_enc = rearrange(frq_pos_enc, "b n c -> b 1 n c")
        self.rotate_dim = frq_pos_enc.shape[-1]
        self.right_align = right_align

    def rotate(self, t):
        seq_len = t.shape[-2]
        if self.right_align:
            # q and k are right-aligned in Perceiver AR
            pos_enc = self.frq_pos_enc[..., -seq_len:, :]
        else:
            # q and k are left-aligned
            pos_enc = self.frq_pos_enc[..., :seq_len, :]

        t_rot, t_pass = t[..., : self.rotate_dim], t[..., self.rotate_dim :]
        t_rot = (t_rot * pos_enc.cos()) + (self._rotate_half(t_rot) * pos_enc.sin())

        return torch.cat((t_rot, t_pass), dim=-1)

    @staticmethod
    def _rotate_half(x):
        # Rearranges channel dimension [x1, x2, x3, x4, ...] -> [-x2, x1, -x4, x3, ...]
        x = rearrange(x, "... (c r) -> ... c r", r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, "... c r -> ... (c r)")


class FrequencyPositionEncoding(nn.Module):
    """Encodes positions as inverse frequencies.

    ```
    inverse_frequencies = 10000 ** -2(i−1)/dim, i = 1, 2, ... dim // 2
    position_encodings = p * inverse_frequencies, p = 0, 1, ... seq_len - 1
    ```
    """

    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, abs_pos):
        # outer product of absolute positions and inverse frequencies
        pos_enc = torch.einsum("b n, f -> b n f", abs_pos, self.inv_freq)
        # frequency position encodings (per example in batch b)
        return repeat(pos_enc, "... pf -> ... (pf r)", r=2)


class FourierPositionEncoding(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], num_frequency_bands: int):
        super().__init__()

        self.input_shape = input_shape
        self.num_frequency_bands = num_frequency_bands

        # create encodings for single example
        pos = self._positions()
        enc = self._position_encodings(pos)

        # flatten encodings along spatial dimensions
        enc = rearrange(enc, "... c -> (...) c")

        # position encoding prototype
        self.register_buffer("position_encoding", enc)

    def _positions(self, v_min=-1.0, v_max=1.0):
        """Create evenly spaced position coordinates for self.spatial_shape with values in [v_min, v_max].

        :param v_min: minimum coordinate value per dimension.
        :param v_max: maximum coordinate value per dimension.
        :return: position coordinates tensor of shape (*shape, len(shape)).
        """
        coords = [torch.linspace(v_min, v_max, steps=s) for s in self.input_shape]
        return torch.stack(torch.meshgrid(*coords), dim=len(self.input_shape))

    def _position_encodings(
        self, p: torch.Tensor, max_frequencies: Optional[Tuple[int, ...]] = None, include_positions: bool = True
    ) -> torch.Tensor:
        """Fourier-encode positions p using self.num_bands frequency bands.

        :param p: positions of shape (*d, c) where c = len(d).
        :param max_frequencies: maximum frequency for each dimension (1-tuple for sequences,
               2-tuple for images, ...). If `None` values are derived from shape of p.
        :param include_positions: whether to include input positions p in returned encodings tensor.
        :returns: position encodings tensor of shape (*d, c * (2 * num_bands + include_positions)).
        """
        encodings = []

        if max_frequencies is None:
            max_frequencies = p.shape[:-1]

        frequencies = [
            torch.linspace(1.0, max_freq / 2.0, self.num_frequency_bands, device=p.device)
            for max_freq in max_frequencies
        ]
        frequency_grids = []

        for i, frequencies_i in enumerate(frequencies):
            frequency_grids.append(p[..., i : i + 1] * frequencies_i[None, ...])

        if include_positions:
            encodings.append(p)

        encodings.extend([torch.sin(math.pi * frequency_grid) for frequency_grid in frequency_grids])
        encodings.extend([torch.cos(math.pi * frequency_grid) for frequency_grid in frequency_grids])

        return torch.cat(encodings, dim=-1)

    def num_position_encoding_channels(self, include_positions: bool = True) -> int:
        return len(self.input_shape) * (2 * self.num_frequency_bands + include_positions)

    def forward(self, b):
        return repeat(self.position_encoding, "... -> b ...", b=b)
