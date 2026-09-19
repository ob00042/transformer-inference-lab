UV ?= uv
export UV_CACHE_DIR ?= .cache/uv
export HF_HOME ?= .cache/huggingface
PYTHON = .venv/bin/python
REVISION = 93efa2f097d58c2a74874c7e644dbc9b0cee75a2

.PHONY: install lint test smoke inspect benchmark benchmark-full
install:
	$(UV) sync --frozen --extra dev
lint:
	.venv/bin/ruff check src tests scripts
	.venv/bin/ruff format --check src tests scripts
test:
	.venv/bin/pytest -q
smoke:
	INFERENCE_LAB_TEST_MODEL=hf-internal-testing/tiny-random-gpt2 INFERENCE_LAB_TEST_REVISION=71034c5d8bde858ff824298bdedc65515b97d2b9 .venv/bin/pytest -q -m integration
inspect:
	$(PYTHON) -m inference_lab.cli inspect
benchmark:
	$(PYTHON) -m inference_lab.cli suite --devices cpu --max-new-tokens 16 --revision $(REVISION)
benchmark-full:
	$(PYTHON) -m inference_lab.cli suite --devices cpu mps cuda --dtypes float32 float16 bfloat16 --max-new-tokens 16 --warmups 2 --repetitions 20 --revision $(REVISION)
