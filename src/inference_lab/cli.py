"""Command-line entry point."""

import argparse

from inference_lab.config import Config, unsupported_reason
from inference_lab.model import generate, load_model, make_inputs, prepare_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["benchmark"])
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    config = Config(max_new_tokens=args.max_new_tokens)
    if reason := unsupported_reason(config):
        parser.error(reason)
    model, tokenizer, load_s = load_model(config, prepare_snapshot(config))
    inputs = make_inputs(tokenizer, config)
    output = generate(model, tokenizer, inputs, config.max_new_tokens)
    print({"shape": list(output.shape), "load_s": load_s})
    print(tokenizer.decode(output[0, config.prompt_length:]))


if __name__ == "__main__":
    main()
