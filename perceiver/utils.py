import torch
import torch.nn as nn

from perceiver.tokenizer import MASK_TOKEN


class Sequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self:
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


def freeze(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False


def predict_masked_samples(masked_samples, encode_fn, tokenizer, model, num_predictions=5, device=None):
    n = len(masked_samples)

    xs, ms = encode_fn(masked_samples)
    xs = xs.to(device)
    ms = ms.to(device)

    with torch.no_grad():
        x_logits, _ = model(xs, ms, masking=False)

    pred_mask = xs == tokenizer.token_to_id(MASK_TOKEN)
    _, pred = torch.topk(x_logits[pred_mask], k=num_predictions, dim=-1)

    output = xs.clone()
    output_dec = [[] for _ in range(n)]

    for i in range(num_predictions):
        output[pred_mask] = pred[:, i]
        for j in range(n):
            output_dec[j].append(tokenizer.decode(output[j].tolist(), skip_special_tokens=True))

    return output_dec
