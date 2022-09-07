import html
import os
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import torch.nn as nn
from einops import rearrange
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import PerceiverConfig as HuggingfacePerceiverConfig, PerceiverForMaskedLM

from perceiver.model.core import DecoderConfig, LitModel, OutputAdapter, PerceiverConfig, PerceiverDecoder, PerceiverIO
from perceiver.model.core.convert import copy_cross_attention_layer_params
from perceiver.model.core.utils import is_checkpoint
from perceiver.model.text.common import copy_encoder_params, TextEncoder, TextEncoderConfig


@dataclass
class TextDecoderConfig(DecoderConfig):
    num_output_query_channels: Optional[int] = None
    vocab_size: int = 10003
    max_seq_len: int = 512


class TextOutputAdapter(OutputAdapter):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        num_output_query_channels: int,
        init_scale: float = 0.02,
    ):
        super().__init__(output_query=torch.empty(max_seq_len, num_output_query_channels), init_scale=init_scale)
        self.linear = nn.Linear(num_output_query_channels, vocab_size)

    def forward(self, x):
        return self.linear(x).squeeze(dim=1)


class TiedTextOutputAdapter(OutputAdapter):
    def __init__(self, max_seq_len: int, vocab_size: int, num_input_channels: int, init_scale: float = 0.02):
        super().__init__(output_query=torch.empty(max_seq_len, num_input_channels), init_scale=init_scale)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x, txt_embedding: nn.Embedding):
        return torch.matmul(x, txt_embedding.weight.T) + self.bias


class MaskedLanguageModel(PerceiverIO):
    def __init__(self, config: PerceiverConfig[TextEncoderConfig, TextDecoderConfig]):
        encoder = TextEncoder(
            config.encoder,
            num_latents=config.num_latents,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
            activation_offloading=config.activation_offloading,
        )
        if config.decoder.num_output_query_channels is None:
            output_adapter = TiedTextOutputAdapter(
                max_seq_len=config.decoder.max_seq_len,
                vocab_size=config.decoder.vocab_size,
                num_input_channels=config.encoder.num_input_channels,
                init_scale=config.decoder.init_scale,
            )
        else:
            output_adapter = TextOutputAdapter(
                vocab_size=config.decoder.vocab_size,
                max_seq_len=config.decoder.max_seq_len,
                num_output_query_channels=config.decoder.num_output_query_channels,
                init_scale=config.decoder.init_scale,
            )
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
            activation_offloading=config.activation_offloading,
            **config.decoder.base_kwargs()
        )
        super().__init__(encoder, decoder)

        if config.params is None or is_checkpoint(config.params):
            pass
        elif os.path.isfile(config.params):
            self.load_state_dict(torch.load(config.params))
        else:
            # import model params from Huggingface Perceiver
            model = PerceiverForMaskedLM.from_pretrained(config.params)
            copy_encoder_params(model, self.encoder)
            copy_decoder_params(model, self.decoder)

    def forward(self, x_masked, pad_mask=None):
        _, n = x_masked.shape

        x_latent = self.encoder(x_masked, pad_mask)
        if isinstance(self.decoder.output_adapter, TiedTextOutputAdapter):
            x_logits = self.decoder(x_latent, txt_embedding=self.encoder.input_adapter.txt_embedding)
        else:
            x_logits = self.decoder(x_latent)

        return x_logits[:, :n, :]


class LitMaskedLanguageModel(LitModel):
    def __init__(
        self,
        encoder: TextEncoderConfig,
        decoder: TextDecoderConfig,
        *args: Any,
        num_predictions: int = 3,
        masked_samples: Optional[List[str]] = None,
        **kwargs: Any
    ):
        super().__init__(encoder, decoder, *args, **kwargs)
        self.model = MaskedLanguageModel(
            PerceiverConfig(
                encoder=encoder,
                decoder=decoder,
                num_latents=self.hparams.num_latents,
                num_latent_channels=self.hparams.num_latent_channels,
                activation_checkpointing=self.hparams.activation_checkpointing,
                activation_offloading=self.hparams.activation_offloading,
                params=self.hparams.params,
            )
        )
        self.loss = nn.CrossEntropyLoss()

        if self.hparams.params is not None and is_checkpoint(self.hparams.params):
            lit_model = LitMaskedLanguageModel.load_from_checkpoint(self.hparams.params, params=None)
            self.model.load_state_dict(lit_model.model.state_dict())

    def setup(self, stage: Optional[str] = None):
        self.filler = MaskedSampleFiller(preprocessor=self.trainer.datamodule.text_preprocessor(), model=self)

    def forward(self, x, pad_mask):
        return self.model(x, pad_mask)

    def step(self, batch):
        labels, x, pad_mask = batch
        logits = self(x, pad_mask)
        logits = rearrange(logits, "b n c -> b c n")
        return self.loss(logits, labels)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("test_loss", loss, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        if self.hparams.masked_samples:
            masked_samples, filled_samples = self.filler.fill(
                self.hparams.masked_samples, self.hparams.num_predictions, self.device
            )

            if isinstance(self.logger, TensorBoardLogger):
                rendered_samples = "\n\n".join(
                    ["  \n".join([html.escape(s)] + ps) for s, ps in zip(masked_samples, filled_samples)]
                )
                self.logger.experiment.add_text("sample predictions", rendered_samples, self.trainer.global_step)
            else:
                # support other loggers here ...
                ...


class MaskedSampleFiller:
    def __init__(self, preprocessor, model=None):
        self.preprocessor = preprocessor
        self.model = model

    def fill(self, masked_samples, num_predictions, device="cpu"):
        masked_samples = [ms.replace("<mask>", self.preprocessor.tokenizer.mask_token) for ms in masked_samples]

        xs, ms = self.preprocessor.preprocess_batch(masked_samples)
        xs = xs.to(device)
        ms = ms.to(device)

        with torch.no_grad():
            x_logits = self.model(xs, ms)

        pred_mask = xs == self.preprocessor.tokenizer.mask_token_id
        pred_ids = torch.topk(x_logits[pred_mask, :], k=num_predictions, dim=1).indices

        results = []

        for i in range(num_predictions):
            xs[pred_mask] = pred_ids[:, i]
            results.append(self.preprocessor.tokenizer.batch_decode(xs, skip_special_tokens=True))

        return masked_samples, list(map(list, zip(*results)))  # transpose results (a list of lists)


def copy_output_adapter_params(src: PerceiverForMaskedLM, tgt: TiedTextOutputAdapter):
    bias_src = src.embedding_decoder.bias
    bias_tgt = tgt.bias

    with torch.no_grad():
        bias_tgt.copy_(bias_src)

    query_src = src.perceiver.decoder.output_position_encodings.position_embeddings
    query_tgt = tgt._output_query

    with torch.no_grad():
        query_tgt.copy_(query_src)


def copy_decoder_params(src: PerceiverForMaskedLM, tgt: PerceiverDecoder):
    copy_cross_attention_layer_params(
        src.perceiver.decoder.decoding_cross_attention, tgt.cross_attn, query_residual=False
    )
    copy_output_adapter_params(src, tgt.output_adapter)


def convert_config(config: HuggingfacePerceiverConfig) -> PerceiverConfig[TextEncoderConfig, TextDecoderConfig]:
    assert config.hidden_act == "gelu"
    assert config.tie_word_embeddings

    encoder_config = TextEncoderConfig(
        vocab_size=config.vocab_size,
        max_seq_len=config.max_position_embeddings,
        num_input_channels=config.d_model,
        num_cross_attention_qk_channels=config.qk_channels,
        num_cross_attention_v_channels=config.v_channels,
        num_cross_attention_heads=config.num_cross_attention_heads,
        num_self_attention_qk_channels=config.qk_channels,
        num_self_attention_v_channels=config.v_channels,
        num_self_attention_heads=config.num_self_attention_heads,
        num_self_attention_layers_per_block=config.num_self_attends_per_block,
        num_self_attention_blocks=config.num_blocks,
        cross_attention_widening_factor=config.cross_attention_widening_factor,
        self_attention_widening_factor=config.self_attention_widening_factor,
        dropout=config.attention_probs_dropout_prob,
        init_scale=config.initializer_range,
    )
    decoder_config = TextDecoderConfig(
        vocab_size=config.vocab_size,
        max_seq_len=config.max_position_embeddings,
        num_cross_attention_qk_channels=config.qk_channels,
        num_cross_attention_v_channels=config.d_model,
        num_cross_attention_heads=config.num_cross_attention_heads,
        cross_attention_widening_factor=config.cross_attention_widening_factor,
        cross_attention_residual=False,
        dropout=config.attention_probs_dropout_prob,
        init_scale=config.initializer_range,
    )
    return PerceiverConfig(
        encoder_config,
        decoder_config,
        num_latents=config.num_latents,
        num_latent_channels=config.d_latents,
        params=config.name_or_path,
    )
