import html
from typing import Any, List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger


class MaskedSamplePrediction(pl.LightningModule):
    def __init__(self, *args: Any, masked_samples: Optional[List[str]] = None, num_predictions: int = 3, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.preprocessor = None

    def setup(self, stage: Optional[str] = None):
        self.preprocessor = self.trainer.datamodule.text_preprocessor()

    def on_validation_epoch_end(self) -> None:
        if self.hparams.masked_samples:
            masked_samples, filled_samples = self.fill_masks(self.hparams.masked_samples, self.hparams.num_predictions)

            if isinstance(self.logger, TensorBoardLogger):
                rendered_samples = "\n\n".join(
                    ["  \n".join([html.escape(s)] + ps) for s, ps in zip(masked_samples, filled_samples)]
                )
                self.logger.experiment.add_text("sample predictions", rendered_samples, self.trainer.global_step)
            else:
                # support other loggers here ...
                ...

    def fill_masks(self, masked_samples, num_predictions):
        masked_samples = [ms.replace("<mask>", self.preprocessor.tokenizer.mask_token) for ms in masked_samples]

        xs, ms = self.preprocessor.preprocess_batch(masked_samples)
        xs = xs.to(self.device)
        ms = ms.to(self.device)

        with torch.no_grad():
            x_logits = self(xs, ms)

        pred_mask = xs == self.preprocessor.tokenizer.mask_token_id
        pred_ids = torch.topk(x_logits[pred_mask, :], k=num_predictions, dim=1).indices

        results = []

        for i in range(num_predictions):
            xs[pred_mask] = pred_ids[:, i]
            results.append(self.preprocessor.tokenizer.batch_decode(xs, skip_special_tokens=True))

        return masked_samples, list(map(list, zip(*results)))  # transpose results (a list of lists)


class MaskedSamplePredictionUtil(MaskedSamplePrediction):
    def __init__(self, preprocessor):
        super().__init__()
        self.preprocessor = preprocessor
        self.model = None

    def forward(self, x, pad_mask=None):
        return self.model(x, pad_mask)
