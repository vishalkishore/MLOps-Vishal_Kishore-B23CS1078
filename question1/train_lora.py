import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_from_disk
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer

from common import (
    DATASET_DIR,
    DEFAULT_HEIGHT,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_RANK,
    DEFAULT_SEED,
    DEFAULT_WIDTH,
    LORA_DIR,
    MODEL_DIR,
    TRAINING_REPORT_PATH,
    bytes_to_mb,
    ensure_runtime_dirs,
    write_key_value_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion v1.5 UNet attention layers with LoRA.")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=LORA_DIR)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_first_present(sample: dict, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in sample:
            return name
    return None


def detect_columns(dataset: DatasetDict) -> tuple[str, str]:
    sample = dataset["train"][0]
    image_column = find_first_present(sample, ["image", "images", "img", "pixel_values"])
    text_column = find_first_present(sample, ["text", "caption", "captions", "prompt"])
    if image_column is None or text_column is None:
        available = ", ".join(sample.keys())
        raise ValueError(f"Could not detect image/text columns in dataset. Available columns: {available}")
    return image_column, text_column


def as_pil(image_value) -> Image.Image:
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")
    if isinstance(image_value, dict) and "bytes" in image_value and image_value["bytes"] is not None:
        import io

        return Image.open(io.BytesIO(image_value["bytes"])).convert("RGB")
    if isinstance(image_value, dict) and "path" in image_value and image_value["path"] is not None:
        return Image.open(image_value["path"]).convert("RGB")
    raise TypeError(f"Unsupported image value type: {type(image_value)!r}")


class LoraTrainingDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer: CLIPTokenizer, image_column: str, text_column: str, height: int, width: int):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.image_column = image_column
        self.text_column = text_column
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.hf_dataset[index]
        image = as_pil(sample[self.image_column])
        prompt = sample[self.text_column]
        if isinstance(prompt, list):
            prompt = prompt[0]
        prompt = str(prompt)

        pixel_values = self.image_transforms(image)
        input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
        }


def collate_fn(examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.stack([example["pixel_values"] for example in examples]),
        "input_ids": torch.stack([example["input_ids"] for example in examples]),
    }


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def count_trainable_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def lora_artifact_size_bytes(path: Path) -> int:
    return sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())


def main() -> None:
    args = parse_args()
    ensure_runtime_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    device = choose_device(args.device)
    weight_dtype = torch.float16 if device.type == "cuda" else torch.float32

    dataset = load_from_disk(str(args.dataset_dir))
    image_column, text_column = detect_columns(dataset)

    tokenizer = CLIPTokenizer.from_pretrained(args.model_dir, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_dir, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.model_dir, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.model_dir, subfolder="vae")
    base_unet = UNet2DConditionModel.from_pretrained(args.model_dir, subfolder="unet")

    base_model_parameter_count = count_parameters(text_encoder) + count_parameters(vae) + count_parameters(base_unet)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet = get_peft_model(base_unet, lora_config)
    trainable_lora_parameter_count = count_trainable_parameters(unet)
    combined_parameter_count = base_model_parameter_count + trainable_lora_parameter_count

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.train()
    text_encoder.eval()
    vae.eval()

    text_encoder.to(device=device, dtype=weight_dtype)
    vae.to(device=device, dtype=weight_dtype)
    unet.to(device=device, dtype=weight_dtype)

    train_dataset = LoraTrainingDataset(
        dataset["train"],
        tokenizer=tokenizer,
        image_column=image_column,
        text_column=text_column,
        height=args.height,
        width=args.width,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    optimizer = AdamW(unet.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps_available = len(train_dataloader) * args.epochs
    max_train_steps = args.max_train_steps if args.max_train_steps > 0 else total_steps_available
    if max_train_steps <= 0:
        raise ValueError("No training steps available. Check that the dataset is not empty.")

    final_loss = math.nan
    global_step = 0

    for epoch in range(args.epochs):
        for batch in train_dataloader:
            pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)
            input_ids = batch["input_ids"].to(device=device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(input_ids, return_dict=False)[0]

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
                dtype=torch.int64,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            final_loss = float(loss.detach().item())
            global_step += 1

            print(
                f"epoch={epoch + 1}/{args.epochs} "
                f"step={global_step}/{max_train_steps} "
                f"loss={final_loss:.6f}"
            )

            if global_step >= max_train_steps:
                break
        if global_step >= max_train_steps:
            break

    unet.save_pretrained(args.output_dir)
    adapter_size_mb = bytes_to_mb(lora_artifact_size_bytes(args.output_dir))

    report = {
        "dataset_dir": args.dataset_dir.resolve(),
        "model_dir": args.model_dir.resolve(),
        "lora_output_dir": args.output_dir.resolve(),
        "image_column": image_column,
        "text_column": text_column,
        "device": device.type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "completed_steps": global_step,
        "base_model_parameter_count": base_model_parameter_count,
        "trainable_lora_parameter_count": trainable_lora_parameter_count,
        "combined_parameter_count": combined_parameter_count,
        "final_training_loss": f"{final_loss:.6f}",
        "lora_adapter_size_mb": f"{adapter_size_mb:.4f}",
    }
    write_key_value_report(TRAINING_REPORT_PATH, report)

    print(f"base model parameters: {base_model_parameter_count}")
    print(f"trainable lora parameters: {trainable_lora_parameter_count}")
    print(f"combined parameter count: {combined_parameter_count}")
    print(f"final training loss: {final_loss:.6f}")
    print(f"lora adapter size: {adapter_size_mb:.4f} MB")
    print(f"training report saved to: {TRAINING_REPORT_PATH}")


if __name__ == "__main__":
    main()
