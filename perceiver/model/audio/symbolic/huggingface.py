import enum
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
from pretty_midi import PrettyMIDI
from transformers import AutoConfig, AutoModelForCausalLM, Pipeline, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.pipelines.base import GenericTensor
from transformers.utils import ModelOutput

from perceiver.data.audio.midi_processor import decode_midi, encode_midi
from perceiver.model.audio.symbolic.backend import SymbolicAudioModel, SymbolicAudioModelConfig
from perceiver.model.audio.symbolic.lightning import LitSymbolicAudioModel


class ReturnType(enum.Enum):
    TENSORS = 0
    AUDIO = 1


class PerceiverSymbolicAudioModelConfig(PretrainedConfig):
    model_type = "perceiver-ar-symbolic-audio-model"

    def __init__(self, backend_config: Optional[SymbolicAudioModelConfig] = None, **kwargs):
        if backend_config is None:
            backend_config = SymbolicAudioModelConfig()
        self.model_config = asdict(backend_config)
        super().__init__(**kwargs)

    @property
    def backend_config(self) -> SymbolicAudioModelConfig:
        return SymbolicAudioModelConfig.create(**self.model_config)


@dataclass
class PerceiverSymbolicAudioModelOutput(CausalLMOutput):
    prefix_len: Optional[int] = None


# TODO: create common base class for PerceiverSymbolicAudioModel and PerceiverCausalLanguageModel
class PerceiverSymbolicAudioModel(PreTrainedModel):
    config_class = PerceiverSymbolicAudioModelConfig

    def __init__(self, config: PerceiverSymbolicAudioModelConfig, **kwargs):
        super().__init__(config)
        if "backend_model" in kwargs:
            self.backend_model = kwargs["backend_model"]  # internal use only
        else:
            self.backend_model = SymbolicAudioModel(config.backend_config)

    @staticmethod
    def from_checkpoint(ckpt_path):
        model = LitSymbolicAudioModel.load_from_checkpoint(ckpt_path).model

        hgf_config = PerceiverSymbolicAudioModelConfig(model.config)
        hgf_config.is_decoder = True

        hgf_model = PerceiverSymbolicAudioModel(hgf_config)
        hgf_model.backend_model.load_state_dict(model.state_dict())

        return hgf_model

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)
        prefix_len = kwargs.get("prefix_len", None)

        if (
            input_ids.shape[1] - prefix_len == self.backend_model.max_latents
            and prefix_len < self.backend_model.max_prefix_len
        ):
            # num_latents == max_latents reached, but not max_prefix_len yet.
            # Extend prefix by 1 token and keep num_latents == max_latents.
            prefix_len += 1

        input_ids = input_ids[:, -self.backend_model.max_seq_len :]

        if attention_mask is not None:
            attention_mask = attention_mask[:, -self.backend_model.max_seq_len :]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prefix_len": prefix_len,
        }

    def _update_model_kwargs_for_generation(self, outputs: PerceiverSymbolicAudioModelOutput, model_kwargs, **kwargs):
        model_kwargs = super()._update_model_kwargs_for_generation(outputs, model_kwargs, **kwargs)
        model_kwargs["prefix_len"] = outputs.prefix_len
        return model_kwargs

    def forward(
        self,
        input_ids: torch.LongTensor,
        prefix_len: int,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ):
        if labels is not None:
            raise ValueError("Loss computation from labels not supported yet")

        if attention_mask is None:
            pad_mask = None
        else:
            pad_mask = ~attention_mask.type(torch.bool)

        logits = self.backend_model(input_ids, prefix_len=prefix_len, pad_mask=pad_mask)
        return PerceiverSymbolicAudioModelOutput(logits=logits, prefix_len=prefix_len)

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        num_latents: int = 1,
        **kwargs,
    ):
        """Augments `GenerationMixin.generate` to support a `num_latents` argument.

        This argument determines the initial number of latents positions assigned to the end of a prompt. During
        generation, first, the number of latent positions grows until `self.backend_model.max_latents` is reached,
        then the prefix length grows until `self.backend_model.max_prefix_len` is reached.

        If the sequence reaches `self.backend_model.max_seq_len`, the left-most prefix token is discarded so that a
        new latent position becomes available for generating the next token.

        :param num_latents: Initial number of latent positions assigned to the end of the input.
        """

        if input_ids is not None:
            seq_len = input_ids.shape[1]
        elif inputs is not None:
            seq_len = inputs.shape[1]
        else:
            raise ValueError("Either inputs or input_ids must be defined")

        if not 0 < seq_len <= self.backend_model.max_seq_len:
            raise ValueError(f"Input sequence length out of valid range [1..{self.backend_model.max_seq_len}]")

        if not 0 < num_latents <= self.backend_model.max_latents:
            raise ValueError(f"num_latents={num_latents} out of valid range [1..{self.backend_model.max_latents}]")
        else:
            num_latents = min(seq_len, num_latents)

        prefix_len = seq_len - num_latents

        if prefix_len > self.backend_model.max_prefix_len:
            num_latents_min = num_latents + prefix_len - self.backend_model.max_prefix_len
            raise ValueError(
                f"For given sequence of length={seq_len}, num_latents must "
                f"be in range [{num_latents_min}..{self.backend_model.max_latents}]"
            )

        return super().generate(inputs=inputs, input_ids=input_ids, prefix_len=prefix_len, **kwargs)


class SymbolicAudioPipeline(Pipeline):
    """Audio pipeline using a :class:`~transformers.PerceiverSymbolicAudioModel` to generate symbolic audio data.
    This pipeline can be loaded from :func:`~transformers.pipeline` using the task identifier `"symbolic-audio-
    generation"`.

    The pipeline accepts as inputs prompts of type `pretty_midi.PrettyMIDI` or `List[pretty_midi.PrettyMIDI]` and
    outputs the generated symbolic audio data either as `pretty_midi.PrettyMIDI` or as a `bytes` array containing the
    rendered audio content in wav format.

    param: max_prompt_length: The maximum length of the prompt. If the prompt is longer than this value, it will be
                              truncated.
    param: return_full_audio: Whether to return the full audio including the prompt in the output or only the generated
                              audio sequence.
    param: return_type: The return type of the pipeline. Can be :obj:`ReturnType.TENSORS` or :obj:`ReturnType.AUDIO`.
    param: render: Whether to render the generated MIDI file to audio wav format. Requires the library `fluidsynth`
                   (https://www.fluidsynth.org/) to be installed.
    param: sf2_path: The path to the soundfont file to use for rendering. If not provided, the default soundfont will
                     be used.
    param: generate_kwargs: Additional keyword arguments passed to the :meth:`~transformers.PreTrainedModel.generate`
                            method.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _sanitize_parameters(
        self,
        max_prompt_length=None,
        return_full_audio=True,
        return_type=ReturnType.AUDIO,
        render=False,
        sf2_path=None,
        **generate_kwargs,
    ):
        preprocess_params = {}
        forward_params = generate_kwargs
        postprocess_params = {}

        if max_prompt_length is not None:
            if max_prompt_length == 0:
                raise ValueError("max_prompt_length must be > 0")
            preprocess_params["max_prompt_length"] = max_prompt_length

        if render and not self._can_render_midi_files():
            raise ValueError("Rendering requires the library `fluidsynth` to be installed.")
        postprocess_params["render"] = render

        if sf2_path is not None and not os.path.exists(sf2_path):
            raise ValueError(f"Provided sf2_path=`{sf2_path}` does not exist")
        postprocess_params["sf2_path"] = sf2_path

        postprocess_params["return_full_audio"] = return_full_audio
        postprocess_params["return_type"] = return_type

        return preprocess_params, forward_params, postprocess_params

    @staticmethod
    def _can_render_midi_files() -> bool:
        try:
            subprocess.check_output(["which", "fluidsynth"])
            return True
        except subprocess.CalledProcessError:
            return False

    def preprocess(self, prompt_midi: PrettyMIDI, max_prompt_length: int = None, **kwargs) -> Dict[str, GenericTensor]:
        encoded_input = torch.tensor(encode_midi(midi=prompt_midi))
        if max_prompt_length is not None:
            encoded_input = encoded_input[:max_prompt_length]

        return {"input_features": torch.unsqueeze(encoded_input, dim=0)}

    def _forward(self, model_inputs: Dict[str, GenericTensor], **generate_kwargs) -> ModelOutput:
        input_features = model_inputs["input_features"]

        generated = self.model.generate(input_ids=input_features, **generate_kwargs)
        model_output = PerceiverSymbolicAudioModelOutput(logits=generated)
        model_output["prompt_length"] = input_features.shape[1]
        return model_output

    def postprocess(
        self,
        model_outputs: ModelOutput,
        return_type=ReturnType.AUDIO,
        return_full_audio=True,
        render=False,
        sf2_path=None,
        **kwargs,
    ) -> Any:
        generated_sequence = model_outputs["logits"][0].numpy().tolist()
        prompt_length = model_outputs["prompt_length"]

        if return_type == ReturnType.TENSORS:
            return {
                "generated_token_ids": generated_sequence if return_full_audio else generated_sequence[prompt_length:]
            }

        if return_type == ReturnType.AUDIO:
            sequence = generated_sequence if return_full_audio else generated_sequence[prompt_length:]
            midi = decode_midi(sequence)

            if not render:
                return {"generated_audio_midi": midi}

            with tempfile.TemporaryDirectory() as tmp_dir:
                midi_file = os.path.join(tmp_dir, "generated_audio.mid")
                wav_file = os.path.join(tmp_dir, "generated_audio.wav")

                midi.write(midi_file)

                cmd = [
                    "fluidsynth",
                    "-F",
                    wav_file,
                ]
                if sf2_path is not None:
                    cmd.append(sf2_path)
                cmd.append(midi_file)

                subprocess.run(cmd)

                with open(wav_file, "rb") as f:
                    generated_wav = f.read()

                return {
                    "generated_audio_wav": generated_wav,
                }

        raise ValueError(f"Invalid return_type={return_type}")


AutoConfig.register(PerceiverSymbolicAudioModelConfig.model_type, PerceiverSymbolicAudioModelConfig)
AutoModelForCausalLM.register(PerceiverSymbolicAudioModelConfig, PerceiverSymbolicAudioModel)

PIPELINE_REGISTRY.register_pipeline(
    "symbolic-audio-generation",
    pipeline_class=SymbolicAudioPipeline,
    pt_model=PerceiverSymbolicAudioModel,
)

# -------------------------------------------------------------------------------------------
#  Conversion utilities
# -------------------------------------------------------------------------------------------


def convert_checkpoint(save_dir, ckpt_url, **kwargs):
    """Convert a `LitSymbolicAudioModel` checkpoint to a persistent `PerceiverSymbolicAudioModel`."""

    model = PerceiverSymbolicAudioModel.from_checkpoint(ckpt_url)
    model.save_pretrained(save_dir, **kwargs)
