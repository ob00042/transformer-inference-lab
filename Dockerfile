FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /usr/local/bin/uv
WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/cache/huggingface PATH=/app/.venv/bin:$PATH
COPY pyproject.toml uv.lock ./
COPY src ./src
# The lockfile selects CPU-only PyTorch wheels on Linux.
RUN uv sync --frozen --no-dev --no-editable --no-cache
RUN mkdir -p /app/results
ENTRYPOINT ["python", "-m", "inference_lab.cli"]
CMD ["benchmark", "--device", "cpu", "--max-new-tokens", "16"]
