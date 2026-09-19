"""Cached model loading and the reference Hugging Face generation path."""

from pathlib import Path
from time import perf_counter

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from inference_lab.config import Config

PROMPT = (
    "Inference engineering measures how a language model processes text. "
    "Careful experiments compare latency, throughput, and numerical correctness. "
)


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def prepare_snapshot(config: Config) -> Path:
    """Download before any load/inference timer; use local directories for offline tests."""
    if Path(config.model).is_dir():
        return Path(config.model).resolve()
    return Path(snapshot_download(
        config.model, revision=config.revision,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
    ))


def load_model(
    config: Config, snapshot: Path,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, float]:
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    synchronize(config.device)
    start = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer needs a pad token or EOS token")
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, torch_dtype=getattr(torch, config.dtype),
        attn_implementation="eager",
    ).to(config.device).eval()
    if model.config.is_encoder_decoder:
        raise ValueError("Only decoder-only causal language models are supported")
    synchronize(config.device)
    return model, tokenizer, perf_counter() - start


def make_inputs(tokenizer: PreTrainedTokenizerBase, config: Config) -> dict[str, torch.Tensor]:
    """A fixed repeated text, sliced to an exact token budget; equal lengths, no padding."""
    base = tokenizer.encode(PROMPT, add_special_tokens=False)
    if not base:
        raise ValueError("Prompt tokenization returned no tokens")
    ids = (base * ((config.prompt_length + len(base) - 1) // len(base)))[:config.prompt_length]
    input_ids = torch.tensor([ids] * config.batch_size, device=config.device)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def generate(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
    inputs: dict[str, torch.Tensor], max_new_tokens: int,
) -> torch.Tensor:
    """Fixed-work greedy generation; EOS does not shorten a measurement."""
    with torch.inference_mode():
        return model.generate(
            **inputs, do_sample=False, num_beams=1, use_cache=True,
            min_new_tokens=max_new_tokens, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
