# Tokenizers

NLP examples in this project use a [SentencePiece](https://arxiv.org/abs/1808.06226) tokenizer [constructed](../perceiver/preproc/text/tokenizer.py)
with the 🤗 [Tokenizers](https://huggingface.co/docs/tokenizers/index) library and trained on English Wikipedia. The
trained tokenizer is stored at `tokenizers/sentencepiece-wikipedia.json`. It was trained with:

```shell
python -m perceiver.scripts.text.tokenizer train wikipedia \
  --dataset_dir=.cache/wikipedia \
  --output_file=tokenizers/sentencepiece-wikipedia.json \
  --vocab_size=32000
```

Tokenizer `.cache/sentencepiece-wikipedia-ext.json` is the same as the previous tokenizer except that its normalization
pipeline has been extended to replace `<br />` with `\n` in input text, a useful extension when working with IMDb
reviews. It has been created with:

```shell
python -m perceiver.scripts.text.tokenizer extend \
  --replacement=["<br />","\n"] \
  --tokenizer_file=tokenizers/sentencepiece-wikipedia.json \
  --output_file=tokenizers/sentencepiece-wikipedia-ext.json
```
