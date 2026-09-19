# Coding-agent development record

This file records work performed in the development session. It does not imply
that a human independently reviewed the implementation or the measurements.
Human code review is **pending** unless a later entry records a specific review.

## 1. Inspect the machine and establish a working baseline

- **Task given to Codex:** inspect the environment before implementation; choose a
  small public model and run a baseline before building the experiment suite.
- **Proposed approach:** use SmolLM2-135M, cached local model loading, greedy
  generation, and CPU float32 first. Separate downloading from measured loading.
- **Human checks:** no human code-review event was observed. Execution permissions
  are distinct from review of the code or its conclusions.
- **Issue encountered:** the sandbox reported MPS unavailable on an Apple M4 Pro.
  A check outside the sandbox confirmed MPS availability. Network downloads also
  required execution outside the restricted sandbox.
- **Verification:** the public model generated eight tokens on CPU, producing an
  output tensor of shape `[1, 72]` for a 64-token prompt. The first 18 offline tests
  passed, including loading, shapes, greedy repeatability, and left-padded batching.

## 2. Add measurements without changing the generation semantics

- **Task given to Codex:** measure repeated latency, first-token latency, throughput,
  token counts, loading, and available memory; reject invalid experiments.
- **Proposed approach:** use the Transformers `generate` reference path, fixed output
  length, synchronized accelerator timing, and separate streamer-instrumented runs
  for first-token latency. Preserve every repetition in JSON. Gate reported results
  on greedy repeatability and batch-versus-single equality for this workload.
- **Human checks:** pending; no manual validation is claimed.
- **Issue encountered:** `torch_dtype` produced a deprecation warning in installed
  Transformers. The implementation now uses `dtype` and requires a compatible
  Transformers version. This was an API compatibility issue, not a measured speedup.
- **Verification:** 24 offline tests and the actual SmolLM2 CPU integration test
  passed before the full benchmark suite was launched. The first-token unit test
  verifies that the prompt callback is excluded. Performance conclusions are based
  only on committed raw measurements, not on generated example numbers.

## Review checklist for a future human reviewer

- Confirm the measurement boundary suits the claimed use case: model execution,
  not server latency, tokenization, network transit, or concurrent request scheduling.
- Reproduce a CPU run from the lockfile and compare its output token IDs.
- Inspect raw latency samples and the relationship between batch latency and
  aggregate throughput; do not infer a serving SLA from a desktop benchmark.
- Audit failure/skip records and the conservative dtype policy before expanding
  hardware support.

## 3. Investigate surprising measurements and verify packaging

- **Task given to Codex:** investigate surprising results before documenting them;
  add a minimal CPU Docker image, small-model CI, and reproducible evidence.
- **Proposed approach:** preserve all 20-repetition baseline samples; recheck MPS
  float32 batch-one prompt lengths in reverse order twice with five warm-ups. Build
  a locked CPU-only container and run it offline against the cached benchmark model.
- **Human checks:** still pending. Neither execution approval nor automated tests
  constitutes a claimed human review of the conclusions.
- **Issues encountered:** MPS 256-token prompts ran faster than 64-token prompts in
  the initial experiment. The ranking persisted in the recheck, with substantial
  short-prompt variability; no root cause is claimed without kernel profiling.
  A newly added negative test attempted to mutate an inference-mode tensor outside
  inference mode. Cloning the tensor before intentional corruption fixed the test
  and allowed it to exercise the runner's correctness rejection as intended.
- **Final verification:** all 16 supported baseline configurations and four MPS
  rechecks passed correctness gates. CUDA and excluded CPU dtypes were explicitly
  skipped. Ruff check/format and 26 offline tests passed. Integration tests passed
  with actual SmolLM2 and with the tiny public GPT-2 used in CI. Docker built from the
  lockfile and generated a JSON result using SmolLM2 with networking disabled.
  The GitHub Actions workflow is configured but has not run on GitHub; the repository
  has not been published. Raw native measurements are committed separately from
  ignored smoke-run outputs. No CUDA hardware verification or human review is claimed.
