"""Run one configuration or a small Cartesian experiment suite."""

import argparse
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from inference_lab.benchmark import environment, report, run
from inference_lab.config import DEFAULT_MODEL, Config, unsupported_reason
from inference_lab.model import prepare_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["benchmark", "suite", "inspect"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", nargs="+", choices=["cpu", "cuda", "mps"], default=["cpu"])
    parser.add_argument(
        "--dtypes", nargs="+", choices=["float32", "float16", "bfloat16"], default=["float32"]
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "inspect":
        print(json.dumps(environment(), indent=2))
        return
    try:
        config = Config(**{name: getattr(args, name) for name in Config.__dataclass_fields__})
    except ValueError as exc:
        parser.error(str(exc))
    configs = (
        [config]
        if args.command == "benchmark"
        else [
            replace(config, device=device, dtype=dtype, batch_size=batch, prompt_length=length)
            for device in args.devices
            for dtype in args.dtypes
            for batch in (1, 4)
            for length in (64, 256)
        ]
    )
    output = args.output or Path("results") / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    snapshot = None
    failed = False
    for item in configs:
        # Resolve once so mutable Hub branches cannot change between configurations.
        try:
            if snapshot is None and unsupported_reason(item) is None:
                snapshot = prepare_snapshot(item)
            result = run(item, snapshot)
        except (ValueError, RuntimeError, NotImplementedError) as exc:
            result = {"config": asdict(item), "status": "error", "reason": str(exc)}
            failed = True
        results.append(result)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report(results), indent=2) + "\n")
        temporary.replace(output)
        print(
            f"{item.device}/{item.dtype} batch={item.batch_size} prompt={item.prompt_length}: "
            f"{result['status']}",
            flush=True,
        )
    print(f"Saved {output}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
