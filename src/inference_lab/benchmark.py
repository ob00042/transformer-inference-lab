"""Synchronized, fixed-work measurements with raw samples and correctness gates."""

import gc
import hashlib
import platform
import statistics
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import psutil
import torch
import transformers
from transformers.generation.streamers import BaseStreamer

from inference_lab.config import Config, unsupported_reason
from inference_lab.model import generate, load_model, make_inputs, prepare_snapshot, synchronize


class FirstTokenTimer(BaseStreamer):
    """HF sends the prompt first, then generated tokens (already copied to the host)."""

    def __init__(self, device: str, start: float):
        self.device = device
        self.start = start
        self.calls = 0
        self.seconds: float | None = None

    def put(self, value: torch.Tensor) -> None:
        self.calls += 1
        if self.calls == 2:
            synchronize(self.device)
            self.seconds = perf_counter() - self.start

    def end(self) -> None:
        pass


def environment() -> dict[str, Any]:
    cpu = platform.processor()
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
        )
        cpu = result.stdout.strip() or cpu
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "os": platform.platform(),
        "cpu": cpu,
        "logical_cpus": psutil.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_vram_bytes": (
            torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
        ),
        "mps_available": torch.backends.mps.is_available(),
    }


def summarize(samples: list[float]) -> dict[str, float | None]:
    ordered = sorted(samples)
    # Linear interpolation (same convention as numpy's default quantile).
    index = (len(ordered) - 1) * 0.95
    low = int(index)
    p95 = ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (index - low)
    return {"median": statistics.median(samples), "p95": p95 if len(samples) >= 20 else None}


def run(config: Config, snapshot: Path | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"config": asdict(config)}
    if reason := unsupported_reason(config):
        return {**result, "status": "skipped", "reason": reason}
    snapshot = snapshot or prepare_snapshot(config)
    model, tokenizer, load_s = load_model(config, snapshot)
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if context_limit and config.prompt_length + config.max_new_tokens > context_limit:
        raise ValueError(f"Prompt + output exceeds model context limit ({context_limit})")
    inputs = make_inputs(tokenizer, config)
    single_inputs = {key: value[:1] for key, value in inputs.items()}
    reference = generate(model, tokenizer, single_inputs, config.max_new_tokens)
    expected = reference.repeat(config.batch_size, 1)
    for _ in range(config.warmups):
        output = generate(model, tokenizer, inputs, config.max_new_tokens)
        if not torch.equal(output, expected):
            raise ValueError("Batch/single greedy outputs differ; refusing to report performance")
    latencies: list[float] = []
    memory: list[int] = []
    generated = config.batch_size * config.max_new_tokens
    for _ in range(config.repetitions):
        synchronize(config.device)
        if config.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        output = generate(model, tokenizer, inputs, config.max_new_tokens)
        synchronize(config.device)
        latencies.append(perf_counter() - start)
        if config.device == "cuda":
            memory.append(torch.cuda.max_memory_allocated())
        if output.shape != (config.batch_size, config.prompt_length + config.max_new_tokens):
            raise ValueError("Unexpected generated output shape")
        if not torch.equal(output, expected):
            raise ValueError("Greedy output changed between repetitions")
        del output
    first_tokens: list[float] = []
    for _ in range(config.repetitions):
        synchronize(config.device)
        timer = FirstTokenTimer(config.device, perf_counter())
        output = generate(model, tokenizer, inputs, config.max_new_tokens, streamer=timer)
        if timer.seconds is None or not torch.equal(output, expected):
            raise ValueError("First-token instrumentation did not preserve generation")
        first_tokens.append(timer.seconds)
        del output
    token_ids = reference[0, config.prompt_length :].tolist()
    result.update(
        {
            "status": "ok",
            "resolved_revision": snapshot.name,
            "model_load_seconds": load_s,
            "prompt_tokens_per_item": config.prompt_length,
            "total_prompt_tokens": config.batch_size * config.prompt_length,
            "generated_tokens_per_item": config.max_new_tokens,
            "total_generated_tokens": generated,
            "latency_seconds": summarize(latencies),
            "latency_samples_seconds": latencies,
            "ttft_seconds": summarize(first_tokens),
            "ttft_samples_seconds": first_tokens,
            "generated_tokens_per_second": summarize([generated / t for t in latencies]),
            "requests_per_second": summarize([config.batch_size / t for t in latencies]),
            "per_sequence_tokens_per_second": summarize(
                [config.max_new_tokens / t for t in latencies]
            ),
            "cuda_peak_allocated_bytes": max(memory) if memory else None,
            "cuda_peak_samples_bytes": memory,
            "process_rss_after_run_bytes": psutil.Process().memory_info().rss,
            "mps_allocated_after_run_bytes": (
                torch.mps.current_allocated_memory() if config.device == "mps" else None
            ),
            "greedy_token_ids_first_item": token_ids,
            "prompt_sha256": hashlib.sha256(
                inputs["input_ids"][0].cpu().numpy().tobytes()
            ).hexdigest(),
            "correctness": {"batch_matches_single": True, "repetitions_match": True},
        }
    )
    del model, inputs, single_inputs, reference, expected
    gc.collect()
    if config.device == "cuda":
        torch.cuda.empty_cache()
    elif config.device == "mps":
        torch.mps.empty_cache()
    return result


def report(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "environment": environment(),
        "methodology": {
            "generation": "greedy; KV cache; eager attention; fixed output via min_new_tokens",
            "latency": "pretokenized device inputs through synchronized full batch generation",
            "ttft": "separate full generation runs; first generated batch token at host streamer",
            "excluded": "download, model load, tokenization, input transfer, output text decoding",
            "p95": "linear interpolation; only reported for at least 20 repetitions",
            "memory": "CUDA allocated peak includes model; RSS/MPS snapshots are not peaks",
        },
        "results": results,
    }
