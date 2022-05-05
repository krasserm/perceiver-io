from perceiver.model.adapter import ClassificationOutputAdapter, ImageInputAdapter, TextInputAdapter, TextOutputAdapter
from perceiver.model.model import PerceiverDecoder, PerceiverEncoder, PerceiverIO, PerceiverMLM, TextMasking


def create_image_classifier(hparams):
    input_adapter = ImageInputAdapter(
        image_shape=hparams.encoder.image_shape, num_frequency_bands=hparams.encoder.num_frequency_bands
    )
    output_adapter = ClassificationOutputAdapter(
        num_classes=hparams.decoder.num_classes,
        num_output_queries=hparams.decoder.num_output_queries,
        num_output_query_channels=hparams.decoder.num_output_query_channels,
    )

    encoder = PerceiverEncoder(
        input_adapter=input_adapter,
        num_latents=hparams.num_latents,
        num_latent_channels=hparams.num_latent_channels,
        activation_checkpointing=hparams.activation_checkpointing,
        **hparams.encoder.gen_kwargs
    )
    decoder = PerceiverDecoder(
        output_adapter=output_adapter,
        num_latent_channels=hparams.num_latent_channels,
        activation_checkpointing=hparams.activation_checkpointing,
        **hparams.decoder.gen_kwargs
    )
    return PerceiverIO(encoder, decoder)


def create_text_classifier(hparams, encoder=None):
    if encoder is None:
        encoder = create_text_encoder(hparams)
    output_adapter = ClassificationOutputAdapter(
        num_classes=hparams.decoder.num_classes,
        num_output_queries=hparams.decoder.num_output_queries,
        num_output_query_channels=hparams.decoder.num_output_query_channels,
    )
    decoder = PerceiverDecoder(
        output_adapter=output_adapter, num_latent_channels=hparams.num_latent_channels, **hparams.decoder.gen_kwargs
    )
    return PerceiverIO(encoder, decoder)


def create_text_encoder(hparams):
    input_adapter = TextInputAdapter(
        vocab_size=hparams.encoder.vocab_size,
        max_seq_len=hparams.encoder.max_seq_len,
        num_input_channels=hparams.encoder.num_input_channels,
    )
    encoder = PerceiverEncoder(
        input_adapter=input_adapter,
        num_latents=hparams.num_latents,
        num_latent_channels=hparams.num_latent_channels,
        activation_checkpointing=hparams.activation_checkpointing,
        **hparams.encoder.gen_kwargs
    )
    return encoder


def create_masked_lm(hparams):
    encoder = create_text_encoder(hparams)
    output_adapter = TextOutputAdapter(
        vocab_size=hparams.decoder.vocab_size,
        max_seq_len=hparams.decoder.max_seq_len,
        num_output_query_channels=hparams.decoder.num_output_query_channels,
        embedding_weights=encoder.input_adapter.text_embedding.weight,
    )
    decoder = PerceiverDecoder(
        output_adapter=output_adapter, num_latent_channels=hparams.num_latent_channels, **hparams.decoder.gen_kwargs
    )
    return PerceiverMLM(encoder, decoder, TextMasking(hparams.encoder.vocab_size))
