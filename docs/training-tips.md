# Training tips

## Additional optimizers

The command line interface does not only support [torch.optim](https://pytorch.org/docs/stable/optim.html) optimizers
but also optimizers from the [pytorch-optimizers](https://github.com/jettify/pytorch-optimizer) project. For example,
if you want to use a [LAMB](https://arxiv.org/abs/1904.00962) optimizer, you can configure it with
`--optimizer=Lamb --optimizer.lr=1e-3 ...` on the command line.

## Activation checkpointing

`PerceiverEncoder` and `PerceiverDecoder` implement [fairscale](https://github.com/facebookresearch/fairscale)
activation checkpoints for self-attention layers and cross-attention layers. These can be activated with
`--model.activation_checkpointing=true` (default is `false`) which saves GPU memory at the cost of more compute.
Activations can also be offloaded to CPU with `--model.activation_offloading=true` (default is `false`).

If the number of cross-attention layers or self-attention blocks is greater than 1 and activation checkpointing is
enabled then `trainer.strategy=ddp_static_graph` must be set. This is necessary to support usage of activation
checkpoints on layers with shared weights (details [here](architecture.md#perceiver)).
