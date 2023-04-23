import torch


class MaskFiller:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def fill(self, model, masked_text_batch, num_predictions, device="cpu"):
        masked_text_batch = [ms.replace("<mask>", self.preprocessor.tokenizer.mask_token) for ms in masked_text_batch]

        xs, ms = self.preprocessor.preprocess_batch(masked_text_batch)
        xs = xs.to(device)
        ms = ms.to(device)

        with torch.no_grad():
            x_logits = model(xs, ms)

        pred_mask = xs == self.preprocessor.tokenizer.mask_token_id
        pred_ids = torch.topk(x_logits[pred_mask, :], k=num_predictions, dim=1).indices

        results = []

        for i in range(num_predictions):
            xs[pred_mask] = pred_ids[:, i]
            results.append(self.preprocessor.tokenizer.batch_decode(xs, skip_special_tokens=True))

        return masked_text_batch, list(map(list, zip(*results)))  # transpose results (a list of lists)
