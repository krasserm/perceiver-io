import pytest
import torch

from perceiver.model.text.clm import CausalLanguageModel, CausalLanguageModelConfig


class MockCausalLanguageModel(CausalLanguageModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls_x = []
        self.calls_prefix_len = []
        self.calls_pad_mask = []

    def forward(self, x, prefix_len, pad_mask=None):
        self.calls_x.append(x)
        self.calls_prefix_len.append(prefix_len)
        self.calls_pad_mask.append(pad_mask)
        return super().forward(x, prefix_len, pad_mask)


@pytest.fixture(scope="function")
def model():
    config = CausalLanguageModelConfig(
        vocab_size=262,
        max_seq_len=12,
        max_latents=6,
        num_channels=16,
        num_self_attention_layers=2,
    )
    yield MockCausalLanguageModel(config)


@pytest.fixture(scope="module")
def prompt_and_pad_mask():
    prompt = torch.randint(6, 262, size=(2, 8))
    prompt[1, :2] = 5

    pad_mask = torch.zeros_like(prompt, dtype=torch.bool)
    pad_mask[1, :2] = True

    yield prompt, pad_mask


def test_generate_neg_num_latents(model, prompt_and_pad_mask):
    with pytest.raises(ValueError):
        model.generate(*prompt_and_pad_mask, num_latents=-1)


def test_generate_exceed_max_latents(model, prompt_and_pad_mask):
    with pytest.raises(ValueError):
        model.generate(*prompt_and_pad_mask, num_latents=9)


def test_generate_zero_tokens(model, prompt_and_pad_mask):
    assert model.generate(*prompt_and_pad_mask, num_tokens=0, pbar=False).shape == (2, 0)


def test_generate_n_tokens(model, prompt_and_pad_mask):
    prompt, pad_mask = prompt_and_pad_mask
    model.generate(prompt, pad_mask, num_tokens=8, num_latents=4, pbar=False)

    assert model.calls_prefix_len == [4, 4, 5, 6, 6, 6, 6, 6]

    calls_pm = model.calls_pad_mask
    assert torch.equal(calls_pm[0], pad_mask)
    assert torch.equal(calls_pm[1], torch.cat([pad_mask, torch.zeros(2, 1, dtype=torch.bool)], dim=1))
    assert torch.equal(calls_pm[2], torch.cat([pad_mask, torch.zeros(2, 2, dtype=torch.bool)], dim=1))
    assert torch.equal(calls_pm[3], torch.cat([pad_mask, torch.zeros(2, 3, dtype=torch.bool)], dim=1))
    assert torch.equal(calls_pm[4], torch.cat([pad_mask, torch.zeros(2, 4, dtype=torch.bool)], dim=1))
    assert torch.equal(calls_pm[5], torch.cat([pad_mask[:, 1:], torch.zeros(2, 5, dtype=torch.bool)], dim=1))
    assert torch.equal(calls_pm[6], torch.cat([pad_mask[:, 2:], torch.zeros(2, 6, dtype=torch.bool)], dim=1))
    assert torch.equal(calls_pm[7], torch.cat([pad_mask[:, 3:], torch.zeros(2, 7, dtype=torch.bool)], dim=1))

    calls_x = model.calls_x
    assert torch.equal(calls_x[0], prompt)
    assert torch.equal(calls_x[1][:, :-1], prompt)
    assert torch.equal(calls_x[2][:, :-2], prompt)
    assert torch.equal(calls_x[3][:, :-3], prompt)
    assert torch.equal(calls_x[4][:, :-4], prompt)
    assert torch.equal(calls_x[5][:, :-5], prompt[:, 1:])
    assert torch.equal(calls_x[6][:, :-6], prompt[:, 2:])
    assert torch.equal(calls_x[7][:, :-7], prompt[:, 3:])
