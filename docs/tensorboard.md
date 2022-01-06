## Masked token predictions

When pretraining on a [masked language modeling](../README.md#masked-language-modeling) task, predictions of masked
tokens are logged to Tensorboard for user-defined samples specified via command line option `--model.masked_samples`.
For example,

```shell
python scripts/mlm.py fit \
  ... \
  --model.masked_samples=['i have watched this <MASK> and it was awesome'] \
  --model.num_predictions=3
```

writes the top 3 predictions for `I have watched this [MASK] and it was awesome` to Tensorboard's `TEXT` page after
each validation epoch:

```
i have watched this [MASK] and it was awesome
i have watched this movie and it was awesome
i have watched this show and it was awesome
i have watched this film and it was awesome
```
