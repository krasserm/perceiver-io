from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, Regex, Tokenizer
from transformers import PreTrainedTokenizerFast


UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
MASK_TOKEN = "<mask>"
CLS_TOKEN = "<cls>"
SEP_TOKEN = "<sep>"

SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN, MASK_TOKEN, CLS_TOKEN, SEP_TOKEN]


def create_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
            normalizers.NFKD(),
            normalizers.StripAccents(),
            normalizers.Replace(Regex(" {2,}"), " "),
        ]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{CLS_TOKEN}:0 $A:0 {SEP_TOKEN}:0",
        pair=f"{CLS_TOKEN}:0 $A:0 {SEP_TOKEN}:0 $B:1 {SEP_TOKEN}:1",
        special_tokens=[(SEP_TOKEN, SPECIAL_TOKENS.index(SEP_TOKEN)), (CLS_TOKEN, SPECIAL_TOKENS.index(CLS_TOKEN))],
    )
    tokenizer.decoder = decoders.Metaspace()
    return tokenizer


def adapt_tokenizer(tokenizer: Tokenizer) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        mask_token=MASK_TOKEN,
        cls_token=CLS_TOKEN,
        sep_token=SEP_TOKEN,
        padding_side="right",
    )
