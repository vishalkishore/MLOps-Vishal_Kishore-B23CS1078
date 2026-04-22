import argparse
from pathlib import Path

from transformers import CLIPTokenizer

from common import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
    MODEL_DIR,
    PROMPT_IDS_PATH,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare prompt token ids for Python and C++ ONNX inference.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output", type=Path, default=PROMPT_IDS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = CLIPTokenizer.from_pretrained(args.model_dir, subfolder="tokenizer")
    output_path = args.output

    prompt_ids = tokenizer(
        args.prompt,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids[0].tolist()

    negative_prompt_ids = tokenizer(
        args.negative_prompt,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids[0].tolist()

    write_json(
        output_path,
        {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "prompt_ids": prompt_ids,
            "negative_prompt_ids": negative_prompt_ids,
        },
    )

    print(f"prompt ids saved to: {output_path}")


if __name__ == "__main__":
    main()
