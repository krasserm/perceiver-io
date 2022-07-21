from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, Regex, Tokenizer, trainers
from transformers import PreTrainedTokenizerFast


UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
MASK_TOKEN = "<mask>"
CLS_TOKEN = "<cls>"
SEP_TOKEN = "<sep>"

SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN, MASK_TOKEN, CLS_TOKEN, SEP_TOKEN]


def create_default_normalizer(lowercase: bool):
    normalizer_list = [
        normalizers.Replace("``", '"'),
        normalizers.Replace("''", '"'),
        normalizers.NFKD(),
    ]

    if lowercase:
        normalizer_list.append(normalizers.Lowercase())

    normalizer_list.append(normalizers.StripAccents())
    normalizer_list.append(normalizers.Replace(Regex(" {2,}"), " "))
    return normalizers.Sequence(normalizer_list)


def create_default_pre_tokenizer(whitespace_split: bool):
    if whitespace_split:
        return pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Metaspace()])
    else:
        return pre_tokenizers.Metaspace()


def create_default_tokenizer(lowercase: bool, whitespace_split: bool) -> Tokenizer:
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = create_default_normalizer(lowercase=lowercase)
    tokenizer.pre_tokenizer = create_default_pre_tokenizer(whitespace_split=whitespace_split)
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{CLS_TOKEN}:0 $A:0 {SEP_TOKEN}:0",
        pair=f"{CLS_TOKEN}:0 $A:0 {SEP_TOKEN}:0 $B:1 {SEP_TOKEN}:1",
        special_tokens=[(SEP_TOKEN, SPECIAL_TOKENS.index(SEP_TOKEN)), (CLS_TOKEN, SPECIAL_TOKENS.index(CLS_TOKEN))],
    )
    tokenizer.decoder = decoders.Metaspace()
    return tokenizer


def adapt_default_tokenizer(tokenizer: Tokenizer) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        mask_token=MASK_TOKEN,
        cls_token=CLS_TOKEN,
        sep_token=SEP_TOKEN,
        padding_side="right",
    )


def train_default_tokenizer(text_generator, vocab_size: int, lowercase: bool, whitespace_split: bool):
    trainer = trainers.UnigramTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS, unk_token=UNK_TOKEN)
    tokenizer = create_default_tokenizer(lowercase=lowercase, whitespace_split=whitespace_split)
    tokenizer.train_from_iterator(text_generator, trainer=trainer)
    return adapt_default_tokenizer(tokenizer)
