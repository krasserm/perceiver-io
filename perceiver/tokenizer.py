from typing import Iterable

from tokenizers import decoders, Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import Lowercase, NFD, Normalizer, Sequence, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
MASK_TOKEN = "[MASK]"


def load_tokenizer(path):
    return Tokenizer.from_file(path)


def save_tokenizer(tokenizer: Tokenizer, path):
    tokenizer.save(path)


def train_tokenizer(tokenizer: Tokenizer, data: Iterable[str], vocab_size):
    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=[PAD_TOKEN, UNK_TOKEN, MASK_TOKEN])
    tokenizer.train_from_iterator(data, trainer)


def create_tokenizer(*normalizer: Normalizer):
    tokenizer = Tokenizer(WordPiece(unk_token=UNK_TOKEN))
    tokenizer.normalizer = Sequence(list(normalizer) + [NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = decoders.WordPiece()
    return tokenizer
