"""Opt-in check with actual trained weights; CI uses a much smaller public model."""

import os

import pytest
import torch

from inference_lab.config import DEFAULT_MODEL, Config
from inference_lab.model import generate, load_model, prepare_snapshot


@pytest.mark.integration
def test_public_model_cpu_greedy_and_batch_consistency():
    config = Config(
        model=os.getenv("INFERENCE_LAB_TEST_MODEL", DEFAULT_MODEL),
        revision=os.getenv("INFERENCE_LAB_TEST_REVISION", "main"),
    )
    model, tokenizer, _ = load_model(config, prepare_snapshot(config))
    texts = ["The capital of France is", "Inference engineering measures latency and throughput."]
    inputs = tokenizer(texts, padding=True, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits
    assert logits.shape[:2] == inputs["input_ids"].shape
    assert torch.isfinite(logits).all()
    output = generate(model, tokenizer, inputs, 4)
    assert torch.equal(output, generate(model, tokenizer, inputs, 4))
    for index, text in enumerate(texts):
        single = generate(model, tokenizer, tokenizer(text, return_tensors="pt"), 4)
        assert torch.equal(output[index, -4:], single[0, -4:])
