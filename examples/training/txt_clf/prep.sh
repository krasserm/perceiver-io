python -m perceiver.scripts.text.preproc imdb \
  --tokenizer=krasserm/perceiver-io-mlm \
  --add_special_tokens=true \
  --static_masking=false \
  --max_seq_len=2048 \
  --task=clf
