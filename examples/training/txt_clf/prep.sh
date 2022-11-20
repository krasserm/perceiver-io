python -m perceiver.scripts.text.preproc imdb \
  --tokenizer=deepmind/language-perceiver \
  --add_special_tokens=true \
  --static_masking=false \
  --max_seq_len=2048 \
  --task=clf
