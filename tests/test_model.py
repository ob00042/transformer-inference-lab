from dataclasses import replace

import pytest
import torch

from inference_lab.config import Config, unsupported_reason
from inference_lab.model import generate, load_model, make_inputs, prepare_snapshot


def test_cpu_loading_shapes_and_greedy_consistency(tiny_snapshot):
    config = Config(model=str(tiny_snapshot), prompt_length=8, batch_size=4, max_new_tokens=4)
    model, tokenizer, load_s = load_model(config, prepare_snapshot(config))
    inputs = make_inputs(tokenizer, config)
    assert inputs["input_ids"].shape == (4, 8)
    assert model.device.type == "cpu"
    assert not model.training
    assert load_s > 0
    first = generate(model, tokenizer, inputs, 4)
    assert first.shape == (4, 12)
    assert torch.equal(first[:, :8], inputs["input_ids"])
    assert torch.equal(first, generate(model, tokenizer, inputs, 4))
    single = generate(model, tokenizer, make_inputs(tokenizer, replace(config, batch_size=1)), 4)
    assert torch.equal(first, single.repeat(4, 1))


def test_variable_length_left_padding(tiny_snapshot):
    model, tokenizer, _ = load_model(Config(), tiny_snapshot)
    texts = ["hello", "hello world text"]
    inputs = tokenizer(texts, padding=True, return_tensors="pt")
    assert tokenizer.padding_side == "left"
    assert inputs["attention_mask"][0].tolist() == [0, 0, 1]
    batch = generate(model, tokenizer, inputs, 4)[:, -4:]
    for i, text in enumerate(texts):
        single = generate(model, tokenizer, tokenizer(text, return_tensors="pt"), 4)[:, -4:]
        assert torch.equal(batch[i], single[0])


@pytest.mark.parametrize(
    "name",
    [
        "batch_size",
        "prompt_length",
        "max_new_tokens",
        "warmups",
        "repetitions",
        "threads",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_invalid_counts(name, value):
    with pytest.raises(ValueError, match=name):
        Config(**{name: value})


@pytest.mark.parametrize("kwargs", [{"device": "tpu"}, {"dtype": "int8"}, {"model": ""}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_unsupported_devices(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert unsupported_reason(Config(device="cuda"))
    assert unsupported_reason(Config(device="mps"))
    assert unsupported_reason(Config(dtype="float16"))
    assert unsupported_reason(Config()) is None
