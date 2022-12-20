python -m perceiver.scripts.text.preproc wikitext \
  --tokenizer=deepmind/language-perceiver \
  --random_train_shift=true \
  --add_special_tokens=false \
  --max_seq_len=4096 \
  --task=clm
