from typing import Iterable

from tokenizers import decoders, Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import Normalizer, Sequence, Lowercase, StripAccents, NFD
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer


PAD_TOKEN = '[PAD]'
PAD_TOKEN_ID = 0

UNK_TOKEN = '[UNK]'
UNK_TOKEN_ID = 1

MASK_TOKEN = '[MASK]'
MASK_TOKEN_ID = 2

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, MASK_TOKEN]


def load_tokenizer(path):
    return Tokenizer.from_file(path)


def save_tokenizer(tokenizer: Tokenizer, path):
    tokenizer.save(path)


def train_tokenizer(tokenizer: Tokenizer, data: Iterable[str], vocab_size):
    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(data, trainer)


def create_tokenizer(*normalizer: Normalizer):
    tokenizer = Tokenizer(WordPiece(unk_token=UNK_TOKEN))
    tokenizer.normalizer = Sequence(list(normalizer) + [NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = decoders.WordPiece()
    return tokenizer
