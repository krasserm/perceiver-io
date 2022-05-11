import torch
from tokenizers import Tokenizer

from perceiver.tokenizer import load_tokenizer, PAD_TOKEN


class TextPreprocessor:
    def __init__(self, tokenizer_path: str, max_seq_len: int = 512):
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.collator = TextCollator(self.tokenizer, max_seq_len)

    def preprocess(self, text):
        return self.preprocess_batch([text])[0][0]

    def preprocess_batch(self, text_batch):
        return self.collator.encode(text_batch)


class TextCollator:
    def __init__(self, tokenizer: Tokenizer, max_seq_len: int):
        self.pad_id = tokenizer.token_to_id(PAD_TOKEN)
        self.tokenizer = tokenizer
        self.tokenizer.enable_padding(pad_id=self.pad_id, pad_token=PAD_TOKEN)
        self.tokenizer.enable_truncation(max_length=max_seq_len)

    def collate(self, batch):
        ys, xs = zip(*batch)
        xs_ids = [x.ids for x in self.tokenizer.encode_batch(xs)]
        xs_ids = torch.tensor(xs_ids)
        pad_mask = xs_ids == self.pad_id
        return torch.tensor(ys), xs_ids, pad_mask

    def encode(self, text_batch):
        batch = [(0, text) for text in text_batch]
        return self.collate(batch)[1:]
