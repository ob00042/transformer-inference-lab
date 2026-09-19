# Measurement contract

The unit of work is one pretokenized batch, already resident on the selected
device. Every sequence receives the same repeated text, truncated to exactly
64 or 256 token IDs. Each request generates the configured number of new tokens;
`min_new_tokens == max_new_tokens` suppresses early EOS. This is a controlled
compute workload, not a representative prompt corpus or a text-quality evaluation.

## What is timed

1. Download/resolve a Hub snapshot **before** timers. All model/tokenizer loading
   thereafter uses local-only files. Resolve once per suite and record its commit.
2. Time tokenizer load, model deserialization, device transfer, and synchronization
   together as `model_load_seconds`. This is one observation per configuration,
   affected by OS caches; it is not a cold-start distribution.
3. Tokenize and transfer inputs outside the request timer. Run a single-item
   correctness reference, then the requested number of full-batch warm-ups.
4. For each measured repetition, synchronize, start a monotonic timer, call
   `generate`, synchronize, and stop. Validate outputs outside the timer.
5. In a **separate** set of full generation runs, record the first generated token
   callback after ignoring the initial prompt callback. Transformers transfers
   streamer tokens to the host. This TTFT therefore includes host token availability,
   Python generation overhead, and prefill, but excludes tokenization and serving
   overhead. It measures the first token for the whole batch, not each item separately.
   Streamer synchronization is deliberately absent from the main latency runs.

Greedy decoding uses one beam, evaluation/inference mode, a fixed random seed,
KV caching, and eager attention on every backend. Eager attention is explicit to
avoid silently comparing different attention implementations. Thread count is fixed
and recorded. No compilation, quantization, continuous batching, or speculative decoding.

## Aggregation and memory

- Request latency is full batch completion latency; each item waits for that batch.
- Aggregate generated tokens/s = `(batch size × output length) / latency`.
- Requests/s = `batch size / latency`; per-sequence tokens/s = `output length / latency`.
- Throughput summaries are computed from each repetition's throughput, not from
  a rounded latency. Raw samples are authoritative.
- Median is always reported. p95 uses linear interpolation and is omitted below
  20 repetitions. Even with 20 samples it is a noisy descriptive statistic, not an SLA.
- CUDA resets peak statistics after warm-up and records peak **allocated** bytes per
  latency run, including resident model allocations. It excludes driver memory and
  other processes; it is not total VRAM usage.
- Process RSS and MPS allocation fields are snapshots taken after measurement.
  They are **not peaks** and are not directly comparable to CUDA peak allocation.
  Apple GPU memory is unified with system RAM.

## Correctness and support

Offline tests instantiate a tiny random GPT-2 and a local tokenizer. Integration
checks load public trained SmolLM2 weights (or tiny public random GPT-2 in CI), test
finite logits, greedy repeatability, and different-length left-padded batching.
The runner additionally checks the exact batch/single output match and repeatability
for every measured configuration and its fixed synthetic prompt.

For the same backend and dtype, deterministic greedy token equality is the desired
baseline. Batched matrix operations can still round differently near tied logits;
this runner conservatively treats any mismatch as an error requiring investigation,
not as permission to publish a faster result. Different dtypes/backends need not
produce identical token IDs: small logit differences can change the greedy argmax.
Stored output token IDs make that divergence inspectable. Matching text does not
establish identical logits or numerical error bounds.

CPU float32 is the portable baseline. CPU low precision is a deliberate scope
exclusion, not a claim that all CPU kernels reject it. CUDA float16 is enabled when
CUDA exists; bfloat16 additionally requires PyTorch's capability check. MPS dtypes
must pass a real tensor operation probe. A failed availability/dtype probe yields an
explicit skip. Model-specific failures after the probe yield an error and a nonzero
exit status: a simple probe cannot certify every model operation. No automatic dtype
fallback or MPS-to-CPU fallback is enabled by this project.

## Reading the JSON

Each report has a schema version, timestamp, environment, methodology, and a list of
configuration results. Results have `status: ok`, `skipped`, or `error`; skipped/error
entries contain a reason and no fabricated metrics. Successful entries include all
raw samples, derived summaries, model revision, input hash, output token IDs, counts,
and correctness flags. Reports are saved after every configuration via atomic rename.

Suite order is deterministic, not randomized. Thermal state, power policy, other
applications, allocator caches, and order can affect results. Investigate surprising
rankings with additional warm-ups, reversed order, and repeated sessions. Do not
compare instrumented TTFT-run throughput with the uninstrumented latency results.
