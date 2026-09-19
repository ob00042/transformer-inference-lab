import json
import sys

import pytest
import torch

from inference_lab.benchmark import FirstTokenTimer, run, summarize
from inference_lab.cli import main
from inference_lab.config import Config


def test_measured_cpu_run(tiny_snapshot):
    result = run(
        Config(
            model=str(tiny_snapshot),
            batch_size=4,
            prompt_length=8,
            max_new_tokens=3,
            warmups=1,
            repetitions=2,
        )
    )
    assert result["status"] == "ok"
    assert result["total_generated_tokens"] == 12
    assert len(result["latency_samples_seconds"]) == 2
    assert all(t > 0 for t in result["ttft_samples_seconds"])
    assert result["cuda_peak_allocated_bytes"] is None
    assert result["latency_seconds"]["p95"] is None
    assert result["generated_tokens_per_second"]["median"] == pytest.approx(
        sum(12 / t for t in result["latency_samples_seconds"]) / 2
    )
    assert result["correctness"]["batch_matches_single"]


def test_percentiles():
    assert summarize([1.0, 3.0])["median"] == 2
    assert summarize(list(range(1, 21)))["p95"] == pytest.approx(19.05)


def test_context_limit(tiny_snapshot):
    with pytest.raises(ValueError, match="context limit"):
        run(Config(model=str(tiny_snapshot), prompt_length=128, max_new_tokens=2))


def test_skip_does_not_load_model():
    result = run(Config(model="does-not-exist", dtype="float16"))
    assert result["status"] == "skipped"
    assert "CPU low precision" in result["reason"]


def test_first_token_ignores_prompt(monkeypatch):
    monkeypatch.setattr("inference_lab.benchmark.perf_counter", lambda: 12.5)
    timer = FirstTokenTimer("cpu", 10)
    timer.put(torch.ones(1, 3))
    assert timer.seconds is None
    timer.put(torch.ones(1))
    assert timer.seconds == 2.5
    timer.put(torch.ones(1))
    assert timer.seconds == 2.5


def test_cli_json(tiny_snapshot, tmp_path, monkeypatch):
    output = tmp_path / "run.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inference-lab",
            "benchmark",
            "--model",
            str(tiny_snapshot),
            "--prompt-length",
            "8",
            "--max-new-tokens",
            "2",
            "--warmups",
            "1",
            "--repetitions",
            "2",
            "--output",
            str(output),
        ],
    )
    main()
    data = json.loads(output.read_text())
    assert data["schema_version"] == 1
    assert data["environment"]["python"]
    assert data["results"][0]["status"] == "ok"
