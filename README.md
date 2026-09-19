# transformer-inference-lab

**When does batching help a small language model, and does lower precision or GPU
execution actually reduce latency on a developer laptop?**

This compact engineering case study measures Hugging Face causal-LM inference with
explicit timing boundaries, correctness gates, and reproducible raw results. It is
an experiment runner, not a serving framework. The initial experiment uses
[SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), a public model that
requires no authentication and fits comfortably on CPU or Apple Silicon.

## Experimental setup

- Apple M4 Pro, 14 logical CPUs, 24 GiB unified RAM; macOS 26.3.1.
- Python 3.11.15, PyTorch 2.14.0, Transformers 4.57.6; dependencies in `uv.lock`.
- Model revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- CPU float32; MPS float32, float16, and bfloat16. CUDA was unavailable.
- Batches of 1 and 4; exact prompt lengths of 64 and 256 tokens; 16 generated
  tokens per item. Identical synthetic text within a batch isolates compute cost.
- Four CPU threads, two explicit warm-ups plus an untimed single-item correctness
  reference, then 20 measured repetitions. First-token timing uses 20 separate runs.
- Greedy generation, KV cache, eager attention, evaluation/inference mode, seed 42.

The shell initially exposed Python 3.13.7; the project uses a separate Python 3.11
virtual environment. MPS detection inside the development sandbox returned false;
all committed hardware benchmarks ran outside that sandbox.

## How inference is measured

The request timer starts with tokenized inputs already on the device and ends after
all output tokens have been generated and the device synchronized. Downloads, model
loading, tokenization, input transfer, and text decoding are excluded. Loading is
measured separately from cached files. Fixed output lengths prevent early EOS from
changing the workload between configurations.

TTFT measures arrival of the first generated batch token at a host-side streamer,
in separate runs so instrumentation does not affect the primary throughput result.
It includes prefill and generation overhead, but is **not end-to-end server TTFT**.

Each repetition records batch latency. Aggregate throughput is total generated
tokens divided by that latency; requests/s and per-sequence tokens/s are also saved.
Median and p95 are descriptive statistics; p95 is omitted below 20 repetitions.
CUDA records peak allocated memory when available. RSS and MPS memory are explicitly
labeled post-run snapshots, not peaks. See the [measurement contract](docs/measurement.md).

## Results

Measured on this machine on 2026-09-19. All 16 supported configurations passed
batch/single and repeatability checks. The 20 skipped configurations are preserved
with reasons: unavailable CUDA and the explicit CPU-low-precision scope exclusion.

[Raw baseline JSON](results/baseline.json) includes every repetition, environment,
load times, memory observations, exact token counts, and output token IDs.
All times below are milliseconds; throughput counts generated tokens across the batch.

| Device / dtype | Batch | Prompt | Median latency | p95 latency | Median TTFT | Tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| cpu / float32 | 1 | 64 | 231.1 | 251.0 | 34.5 | 69.2 |
| cpu / float32 | 1 | 256 | 261.5 | 283.1 | 68.2 | 61.2 |
| cpu / float32 | 4 | 64 | 415.5 | 442.8 | 62.9 | 154.0 |
| cpu / float32 | 4 | 256 | 570.2 | 635.1 | 174.4 | 112.2 |
| mps / float32 | 1 | 64 | 386.6 | 444.4 | 37.6 | 41.4 |
| mps / float32 | 1 | 256 | 300.3 | 344.7 | 49.1 | 53.3 |
| mps / float32 | 4 | 64 | 345.6 | 421.9 | 47.3 | 185.2 |
| mps / float32 | 4 | 256 | 323.2 | 344.6 | 76.9 | 198.0 |
| mps / float16 | 1 | 64 | 299.9 | 461.5 | 38.3 | 53.4 |
| mps / float16 | 1 | 256 | 300.2 | 312.8 | 46.9 | 53.3 |
| mps / float16 | 4 | 64 | 302.5 | 378.4 | 43.6 | 211.5 |
| mps / float16 | 4 | 256 | 333.2 | 345.9 | 78.4 | 192.1 |
| mps / bfloat16 | 1 | 64 | 395.0 | 465.9 | 42.8 | 40.5 |
| mps / bfloat16 | 1 | 256 | 298.5 | 339.2 | 49.3 | 53.6 |
| mps / bfloat16 | 4 | 64 | 304.6 | 368.2 | 45.2 | 210.1 |
| mps / bfloat16 | 4 | 256 | 332.9 | 333.9 | 78.8 | 192.3 |

## Engineering observations

- **CPU batching trades latency for throughput.** At 64 prompt tokens, batch 4
  achieved 154.0 tokens/s versus 69.2 for batch 1 (2.22×), while batch completion
  latency rose from 231.1 to 415.5 ms (1.80×). This is aggregate work completed,
  not a faster response for an individual user.
- **Longer CPU prompts cost more.** At batch 4, moving from 64 to 256 prompt tokens
  increased median request latency from 415.5 to 570.2 ms; TTFT rose from 62.9 to
  174.4 ms. Prefill is visible even with only 16 generated tokens.
- **MPS does not win every workload.** In the initial run, CPU float32 had lower
  batch-one latency than every tested MPS dtype. MPS handled batch 4 more efficiently,
  particularly at 256 prompt tokens. These are observations for this small model
  with eager attention, not a general CPU/GPU ranking.
- **Lower precision was not a consistent speedup.** For MPS batch 4 / 256 prompt
  tokens, float32 took 323.2 ms, float16 333.2 ms, and bfloat16 332.9 ms. The short
  prompt case favored lower precision. No universal dtype winner follows from this run.
- **Output equality was checked, not assumed.** All measured configurations produced
  the same token IDs for each prompt length in this synthetic workload. That does not
  imply equal logits or guarantee agreement on arbitrary prompts. Across backends
  or dtypes, rounding can legitimately change a near-tied greedy decision.

**Investigation of an unexpected MPS result.** The initial float32 batch-one run
was faster at 256 prompt tokens than at 64. Repeating those cases in the order
256, 64, 256, 64 with five warm-ups produced medians of 298.9, 420.7, 303.3, and
355.2 ms. The ordering persisted, while the short-prompt result varied substantially.
This weakens a simple first-case warm-up explanation but does not establish a cause;
kernel profiling and separate sessions are needed. It is not evidence that longer
prompts are generally cheaper. [Raw follow-up data](results/validation.json) and
[the exact recheck script](scripts/recheck_mps.py) are included. Reproduce with
`HF_HOME=.cache/huggingface .venv/bin/python scripts/recheck_mps.py` on MPS hardware.

## Reproduce the benchmark

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). The first run downloads
weights; later runs can use the cache. No account, API key, or GPU is required.

```sh
make install
make inspect
make lint
make test
make smoke
make benchmark
```

`make benchmark` runs the four CPU float32 cases, with 10 repetitions and 16 output
tokens per item. JSON is saved under `results/` with a unique timestamp. It does not
overwrite the committed baseline. To reproduce the full experiment (including skips):

```sh
make benchmark-full
```

One configurable experiment:

```sh
HF_HOME=.cache/huggingface .venv/bin/python -m inference_lab.cli benchmark \
  --model HuggingFaceTB/SmolLM2-135M \
  --revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 \
  --device cpu --dtype float32 --batch-size 1 --prompt-length 64 \
  --max-new-tokens 32 --warmups 2 --repetitions 20 --threads 4
```

Use `--devices cpu mps --dtypes float32 float16 bfloat16` with `suite` to choose a
matrix. A single `benchmark` uses singular `--device` and `--dtype`. The suite fixes
batch sizes at 1/4 and prompt lengths at 64/256. Local model directories are supported.
Without `--revision`, a Hub model resolves `main` once and its actual commit is saved.
Set `HF_HUB_OFFLINE=1` after the model is cached for a network-free run.

The supplied Linux lockfile uses CPU PyTorch wheels to keep CI and Docker small.
CUDA experiments require a CUDA-enabled PyTorch installation. The runner detects
CUDA capabilities, but this machine could not validate that execution path.

### CPU container

```sh
docker build -t transformer-inference-lab .
docker run --rm \
  -v "$(pwd)/results:/app/results" \
  -v inference-lab-cache:/cache \
  transformer-inference-lab benchmark --device cpu --max-new-tokens 16
```

The image installs from the same lockfile and persists model downloads in a volume.
The image was built and its CPU inference path verified locally with networking disabled.
Docker Desktop adds a Linux VM; container numbers are separate from native macOS results.

## Testing and CI

`make test` runs 26 offline tests using a locally constructed tiny GPT-2 and tokenizer:
loading, shapes, CPU execution, repeatable greedy decoding, left-padded mixed-length
batching, configuration validation, metric arithmetic, streamer boundaries, context
limits, correctness rejection, and JSON/error handling.

`make smoke` downloads `hf-internal-testing/tiny-random-gpt2`, a small public test
model, and checks real model loading, finite logits, and greedy/batching consistency.
It is a plumbing check, not a model-quality evaluation. To test the actual benchmark
model, run `HF_HOME=.cache/huggingface .venv/bin/pytest -q -m integration`.

GitHub Actions runs Ruff, offline tests, and that tiny CPU integration check with
uv and Hugging Face caching. No GPU or large model is needed. The workflow has been
added locally; a hosted Actions run requires publishing the repository.

## Limitations

- One laptop, one model, synthetic repeated prompts, fixed batch sizes, and no
  concurrent serving traffic. No quality evaluation or representative workload claim.
- Fixed case order and desktop thermal/power/background activity can affect timings.
  Twenty repetitions are useful evidence, not a production latency guarantee.
- CPU low precision is excluded by policy, not declared universally unsupported.
  Device/dtype tensor probes cannot certify every model operation; later failures
  remain errors and produce a nonzero exit status rather than silent fallback.
- MPS memory snapshots are not peak memory; no CUDA measurements were possible here.
- Loading is measured once per case with existing caches. It is not a cold-download
  or cold-start benchmark. Arbitrary models may need architecture-specific changes.
- The strict batch-equality gate can reject legitimate floating-point argmax changes;
  such cases need numerical investigation before their performance is reported.

## Future experiments

Use a varied prompt corpus and report near-tie/logit agreement; randomize case order
across sessions; compare eager attention with SDPA; measure output-length scaling;
validate CUDA synchronization and allocator peaks on real hardware. Each should be
an isolated experiment with correctness checks and raw results.

Development evidence and pending human-review items are recorded in
[docs/agent-workflow.md](docs/agent-workflow.md). Code is MIT-licensed; the model has
its own license in its model card.
