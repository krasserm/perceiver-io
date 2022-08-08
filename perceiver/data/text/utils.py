import string

from transformers import PerceiverTokenizer


class PerceiverTokenizerUtil:
    def __init__(self, tokenizer: PerceiverTokenizer):
        assert isinstance(tokenizer, PerceiverTokenizer)
        self.tokenizer = tokenizer
        self.whitespace_ids = set(tokenizer(string.whitespace, add_special_tokens=False).input_ids)

    def word_ids(self, token_ids):
        """Creates word ids from `token_ids`.

        Words boundaries are defined using whitespace boundaries. Whitespaces preceding a word have the same word id as
        the actual word following these whitespaces. Special tokens are assigned a `None` word id. Consecutive words do
        not necessarily have consecutive word ids. This method only guarantees that distinct words have distinct word
        ids. This is sufficient for `WordMaskingCollator` to function properly.
        """
        word_ids = []
        curr_id = 0

        special_mask = self.tokenizer.get_special_tokens_mask(token_ids, already_has_special_tokens=True)
        regular_token = True

        for i, token_id in enumerate(token_ids):
            if special_mask[i]:
                word_ids.append(None)
                curr_id += 1
            elif token_id in self.whitespace_ids:
                if regular_token:
                    regular_token = False
                    curr_id += 1
                word_ids.append(curr_id)
            else:
                regular_token = True
                word_ids.append(curr_id)

        return word_ids
