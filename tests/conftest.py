"""A real, tiny Transformers model/tokenizer saved locally; no network in unit tests."""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast


@pytest.fixture(scope="session")
def tiny_snapshot(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny-model")
    tokenizer = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "hello": 3, "world": 4, "text": 5},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    fast.save_pretrained(path)
    torch.manual_seed(7)
    GPT2LMHeadModel(
        GPT2Config(
            vocab_size=6,
            n_positions=128,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=2,
            eos_token_id=2,
            pad_token_id=1,
        )
    ).save_pretrained(path)
    return path
