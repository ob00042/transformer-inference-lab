"""Recheck the surprising batch-one MPS prompt-length ordering, without cherry-picking."""

import json
from dataclasses import replace
from pathlib import Path

from inference_lab.benchmark import report, run
from inference_lab.config import Config
from inference_lab.model import prepare_snapshot


def main() -> None:
    config = Config(
        device="mps",
        revision="93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        max_new_tokens=16,
        warmups=5,
        repetitions=20,
    )
    snapshot = prepare_snapshot(config)
    results = []
    for length in (256, 64, 256, 64):
        result = run(replace(config, prompt_length=length), snapshot)
        results.append(result)
        Path("results").mkdir(exist_ok=True)
        Path("results/validation.json").write_text(json.dumps(report(results), indent=2) + "\n")
        print(length, result.get("latency_seconds", result["status"]), flush=True)


if __name__ == "__main__":
    main()
