import string

from transformers import PerceiverTokenizer


class PerceiverTokenizerUtil:
    def __init__(self, tokenizer: PerceiverTokenizer):
        assert isinstance(tokenizer, PerceiverTokenizer)
        self.tokenizer = tokenizer
        self.whitespace_ids = set(tokenizer(string.whitespace, add_special_tokens=False).input_ids)

    def word_ids(self, token_ids):
        word_ids = []
        curr_id = -1

        special_mask = self.tokenizer.get_special_tokens_mask(token_ids, already_has_special_tokens=True)
        special = True

        for i, token_id in enumerate(token_ids):
            if token_id in self.whitespace_ids or special_mask[i]:
                word_ids.append(None)
                special = True
            else:
                if special:
                    curr_id += 1
                    special = False
                word_ids.append(curr_id)

        return word_ids
