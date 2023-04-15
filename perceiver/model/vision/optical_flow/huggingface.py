from dataclasses import asdict, dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoConfig, Pipeline, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.pipelines import PIPELINE_REGISTRY

from perceiver.data.vision.optical_flow import OpticalFlowProcessor, render_optical_flow
from perceiver.model.core.huggingface import (
    copy_cross_attention_layer_params,
    copy_latent_provider_params,
    copy_params,
    copy_self_attention_block_params,
)
from perceiver.model.vision.optical_flow.backend import (
    OpticalFlow,
    OpticalFlowConfig,
    OpticalFlowDecoderConfig,
    OpticalFlowEncoderConfig,
    PerceiverDecoder,
    PerceiverEncoder,
)


class OpticalFlowPerceiverConfig(PretrainedConfig):
    model_type = "perceiver-io-optical-flow"

    def __init__(self, backend_config: Optional[OpticalFlowConfig] = None, **kwargs):
        if backend_config is None:
            backend_config = OpticalFlowConfig(
                OpticalFlowEncoderConfig(), OpticalFlowDecoderConfig(), num_latents=512, num_latent_channels=512
            )
        self.model_config = asdict(backend_config)
        super().__init__(**kwargs)

    @property
    def backend_config(self) -> OpticalFlowConfig:
        model_config = self.model_config.copy()
        encoder_config = model_config.pop("encoder")
        decoder_config = model_config.pop("decoder")
        return OpticalFlowConfig(
            encoder=OpticalFlowEncoderConfig(**encoder_config),
            decoder=OpticalFlowDecoderConfig(**decoder_config),
            **model_config,
        )


class OpticalFlowPerceiver(PreTrainedModel):
    config_class = OpticalFlowPerceiverConfig

    def __init__(self, config: OpticalFlowPerceiverConfig):
        super().__init__(config)
        self.backend_model = OpticalFlow(config.backend_config)

    def forward(self, inputs: torch.LongTensor):
        return OpticalFlowPerceiverOutput(logits=self.backend_model(inputs))


@dataclass
class OpticalFlowPerceiverOutput(ModelOutput):
    logits: torch.FloatTensor = None


ImagePair = Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor], Tuple[Image.Image, Image.Image]]


class OpticalFlowPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size = self.model.config.backend_config.encoder.image_shape
        self.processor = OpticalFlowProcessor(patch_size=self.patch_size)

    def _sanitize_parameters(self, **kwargs):
        forward_kwargs = {}
        postprocess_kwargs = {}
        if "micro_batch_size" in kwargs:
            forward_kwargs["micro_batch_size"] = kwargs["micro_batch_size"]
        if "render" in kwargs:
            postprocess_kwargs["render"] = kwargs["render"]
        return {}, forward_kwargs, postprocess_kwargs

    def preprocess(self, image_pair: ImagePair, **kwargs):
        if isinstance(image_pair[0], Image.Image):
            image_pair = (np.array(image_pair[0]), np.array(image_pair[1]))

        return {
            "input_features": self.processor.preprocess(image_pair),
            "input_image_shape": image_pair[0].shape,
        }

    def _forward(self, inputs, micro_batch_size=1, **kwargs):
        input_features = inputs["input_features"]
        output_tensors = []

        for i in range(0, input_features.shape[0], micro_batch_size):
            input_features_micro_batch = input_features[i : i + micro_batch_size]
            output_tensor = self.model(input_features_micro_batch).logits
            output_tensors.append(output_tensor)

        model_output = OpticalFlowPerceiverOutput(logits=torch.concat(output_tensors, dim=0))
        model_output["input_image_shape"] = inputs["input_image_shape"]
        return model_output

    def postprocess(self, model_output, render=False, **kwargs):
        optical_flow = self.processor.postprocess(model_output.logits, img_shape=model_output.input_image_shape)
        optical_flow = optical_flow[0].numpy()

        if render:
            return render_optical_flow(optical_flow)
        else:
            return optical_flow


AutoConfig.register(OpticalFlowPerceiverConfig.model_type, OpticalFlowPerceiverConfig)

PIPELINE_REGISTRY.register_pipeline(
    "optical-flow",
    pipeline_class=OpticalFlowPipeline,
    pt_model=OpticalFlowPerceiver,
)


# -------------------------------------------------------------------------------------------
#  Conversion utilities
# -------------------------------------------------------------------------------------------


def convert_config(config: transformers.PerceiverConfig) -> OpticalFlowConfig:
    assert config.hidden_act == "gelu"

    image_shape = tuple(config.train_size)

    encoder_config = OpticalFlowEncoderConfig(
        image_shape=image_shape,
        num_patch_input_channels=27,
        num_patch_hidden_channels=64,
        num_frequency_bands=64,
        num_cross_attention_layers=1,
        num_cross_attention_heads=config.num_cross_attention_heads,
        num_self_attention_heads=config.num_self_attention_heads,
        num_self_attention_layers_per_block=config.num_self_attends_per_block,
        num_self_attention_blocks=config.num_blocks,
        first_self_attention_block_shared=True,
        cross_attention_widening_factor=config.cross_attention_widening_factor,
        self_attention_widening_factor=config.self_attention_widening_factor,
        dropout=config.attention_probs_dropout_prob,
        init_scale=config.initializer_range,
    )
    decoder_config = OpticalFlowDecoderConfig(
        image_shape=image_shape,
        num_cross_attention_qk_channels=512,
        num_cross_attention_v_channels=512,
        num_cross_attention_heads=config.num_cross_attention_heads,
        cross_attention_widening_factor=config.cross_attention_widening_factor,
        cross_attention_residual=False,
        dropout=config.attention_probs_dropout_prob,
        init_scale=config.initializer_range,
        rescale_factor=100.0,
    )
    return OpticalFlowConfig(
        encoder_config,
        decoder_config,
        num_latents=config.num_latents,
        num_latent_channels=config.d_latents,
    )


# -------------------------------------------------------------------------------------------
#  Conversion utilities
# -------------------------------------------------------------------------------------------


def convert_model(save_dir, source_repo_id="deepmind/optical-flow-perceiver", **kwargs):
    """Convert a Hugging Face `PerceiverForOpticalFlow` to a persistent `OpticalFlowPerceiver`."""

    src_model = transformers.PerceiverForOpticalFlow.from_pretrained(source_repo_id)
    tgt_config = OpticalFlowPerceiverConfig(convert_config(src_model.config))
    tgt_model = OpticalFlowPerceiver(tgt_config)

    copy_flow_encoder_params(src_model.perceiver, tgt_model.backend_model.encoder)
    copy_flow_decoder_params(src_model.perceiver, tgt_model.backend_model.decoder)

    tgt_model.save_pretrained(save_dir, **kwargs)


def copy_flow_encoder_params(src: transformers.PerceiverModel, tgt: PerceiverEncoder):
    copy_cross_attention_layer_params(src.encoder.cross_attention, tgt.cross_attn_1, query_residual=True)
    copy_self_attention_block_params(src.encoder.self_attends, tgt.self_attn_1)
    copy_latent_provider_params(src, tgt)
    # Copy input adapter parameters
    copy_params(src.input_preprocessor.conv_after_patches, tgt.input_adapter.linear)


def copy_flow_decoder_params(src: transformers.PerceiverModel, tgt: PerceiverDecoder):
    copy_cross_attention_layer_params(
        src.decoder.decoder.decoding_cross_attention, tgt.cross_attn, query_residual=False
    )
    # Copy output adapter parameters
    copy_params(src.decoder.decoder.final_layer, tgt.output_adapter.linear)
