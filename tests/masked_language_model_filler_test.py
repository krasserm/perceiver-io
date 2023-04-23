import pytest

from perceiver.model.text import mlm  # noqa: F401

from transformers import pipeline


@pytest.fixture(scope="module")
def filler():
    yield pipeline("fill-mask", model="krasserm/perceiver-io-mlm-imdb", top_k=1)


def test_fill(filler):
    masked_text = [
        "I watched this[MASK][MASK][MASK][MASK] and it was awesome.",
        "I watched this[MASK][MASK][MASK][MASK][MASK] and it was awesome.",
        "I watched this[MASK][MASK][MASK][MASK][MASK][MASK] and it was awesome.",
        "I watched this[MASK][MASK][MASK][MASK][MASK][MASK][MASK] and it was awesome.",
        "I watched this[MASK][MASK][MASK][MASK][MASK][MASK][MASK][MASK] and it was awesome.",
        "I watched this[MASK][MASK][MASK][MASK][MASK][MASK][MASK][MASK][MASK] and it was awesome.",
        "I watched this[MASK][MASK][MASK][MASK][MASK][MASK][MASK][MASK][MASK][MASK] and it was awesome.",
    ]

    expected_text = [
        "I watched this one and it was awesome.",
        "I watched this film and it was awesome.",
        "I watched this movie and it was awesome.",
        "I watched this movie, and it was awesome.",
        "I watched this episode and it was awesome.",
        "I watched this recently and it was awesome.",
        "I watched this yesterday and it was awesome.",
    ]

    result = filler(masked_text)
    result_text = [replace_masks(t, extract_tokens(r)) for t, r in zip(masked_text, result)]

    assert result_text == expected_text


def extract_tokens(result):
    return [item[0]["token_str"] for item in result]


def replace_masks(text, tokens):
    for token in tokens:
        text = text.replace("[MASK]", token, 1)
    return text
