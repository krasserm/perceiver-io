from typing import Iterable

from tokenizers import decoders, normalizers, Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import Lowercase, Replace, StripAccents, NFD
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer


PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
MASK_TOKEN = '[MASK]'


SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, MASK_TOKEN]


def load_tokenizer(path):
    return Tokenizer.from_file(path)


def save_tokenizer(tokenizer: Tokenizer, path):
    tokenizer.save(path)


def train_tokenizer(tokenizer: Tokenizer, data: Iterable[str], vocab_size=10003):
    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(data, trainer)


def create_tokenizer():
    tokenizer = Tokenizer(WordPiece(unk_token=UNK_TOKEN))
    tokenizer.normalizer = normalizers.Sequence([Replace('<br />', ' '), NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = decoders.WordPiece()
    return tokenizer
