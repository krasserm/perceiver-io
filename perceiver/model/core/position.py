import torch
import torch.nn as nn


class RotaryPositionEmbedding:
    # See section 3.4.2 in https://arxiv.org/abs/2104.09864
    # (here, a different permutation of channels is used)

    def __init__(self, frq_pos_enc: torch.Tensor, right_align: bool = False):
        # frq_pos_enc shape is either (n, c) or (b, 1, n, c).
        # frq_pos_enc is broadcast to (b, h, n, c).
        self.frq_pos_enc = frq_pos_enc
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

    def _rotate_half(self, x):
        x1 = x[..., : self.rotate_dim // 2]
        x2 = x[..., self.rotate_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class FrequencyPositionEncoding(nn.Module):
    def __init__(self, encoded_channels_per_head):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, encoded_channels_per_head, 2).float() / encoded_channels_per_head))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        pos = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        pos_enc = torch.outer(pos, self.inv_freq)
        return torch.cat((pos_enc, pos_enc), dim=-1)
