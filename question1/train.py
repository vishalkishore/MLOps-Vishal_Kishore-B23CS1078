from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from diffusers import DDPMScheduler, UNet2DConditionModel
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL

from common import ARTIFACTS_DIR, DATASET_DIR, LORA_DIR, MODEL_DIR, write_key_value_report


MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DATASET_ID = "lambda/naruto-blip-captions"
METRICS_PATH = ARTIFACTS_DIR / "metrics.txt"

IMAGE_SIZE = 512
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
MAX_STEPS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class NarutoDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        image = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        image_array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        pixel_values = torch.from_numpy(image_array).permute(2, 0, 1)

        text = sample["text"]
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": tokens.input_ids.squeeze(0),
        }


def count_parameters(model):
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total_params, trainable_params


def get_dir_size_mb(path):
    total_size = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    return total_size / (1024 * 1024)


def write_metrics(
    base_total_params,
    lora_trainable_params,
    combined_total_params,
    final_loss,
    adapter_size_mb,
):
    write_key_value_report(
        METRICS_PATH,
        {
            "base_total_params": base_total_params,
            "lora_trainable_params": lora_trainable_params,
            "combined_total_params": combined_total_params,
            "final_training_loss": f"{final_loss:.6f}",
            "adapter_size_mb": f"{adapter_size_mb:.6f}",
        },
    )


def download_model():
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        return MODEL_DIR

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=MODEL_ID, local_dir=str(MODEL_DIR), local_dir_use_symlinks=False)
    return MODEL_DIR


def download_dataset():
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        return DATASET_DIR

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET_ID)
    dataset.save_to_disk(str(DATASET_DIR))
    return DATASET_DIR


def load_dataset_local():
    dataset = load_from_disk(str(DATASET_DIR))
    if "train" in dataset:
        return dataset["train"]
    return dataset


def load_unet_with_lora(model_path):
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
    unet_total_params, _ = count_parameters(unet)
    print(f"unet base total params = {unet_total_params}")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
    _, lora_trainable_params = count_parameters(unet)
    combined_unet_params = unet_total_params + lora_trainable_params
    print(
        f"unet with lora: trainable LoRA params = {lora_trainable_params}, "
        f"combined params = {combined_unet_params}"
    )
    return unet, unet_total_params, lora_trainable_params


def train():
    model_path = download_model()
    dataset_path = download_dataset()
    print(f"model saved to: {model_path}")
    print(f"dataset saved to: {dataset_path}")

    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    base_unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
    base_unet_total_params, _ = count_parameters(base_unet)
    text_encoder_total_params, _ = count_parameters(text_encoder)
    vae_total_params, _ = count_parameters(vae)
    base_model_total_params = (
        base_unet_total_params + text_encoder_total_params + vae_total_params
    )
    print(f"base stable diffusion total params = {base_model_total_params}")

    del base_unet
    unet, _, lora_trainable_params = load_unet_with_lora(model_path)
    combined_model_total_params = base_model_total_params + lora_trainable_params
    print(
        f"stable diffusion with lora: trainable LoRA params = {lora_trainable_params}, "
        f"combined total params = {combined_model_total_params}"
    )

    text_encoder.to(DEVICE)
    vae.to(DEVICE)
    unet.to(DEVICE)

    text_encoder.eval()
    vae.eval()
    for parameter in text_encoder.parameters():
        parameter.requires_grad = False
    for parameter in vae.parameters():
        parameter.requires_grad = False

    train_dataset = load_dataset_local()
    train_loader = DataLoader(
        NarutoDataset(train_dataset, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    optimizer = AdamW((parameter for parameter in unet.parameters() if parameter.requires_grad), lr=LEARNING_RATE)

    step = 0
    final_loss = None
    for epoch in range(NUM_EPOCHS):
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                encoder_hidden_states = text_encoder(input_ids)[0]

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=DEVICE,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            final_loss = loss.item()
            print(f"epoch={epoch} step={step} loss={loss.item():.4f}")

            if step >= MAX_STEPS:
                LORA_DIR.mkdir(parents=True, exist_ok=True)
                unet.save_pretrained(str(LORA_DIR))
                adapter_size_mb = get_dir_size_mb(LORA_DIR)
                print(f"final training loss = {final_loss:.6f}")
                print(f"lora adapter size = {adapter_size_mb:.6f} MB")
                write_metrics(base_model_total_params, lora_trainable_params, combined_model_total_params, final_loss, adapter_size_mb)
                print(f"lora adapter saved to: {LORA_DIR}")
                return

    LORA_DIR.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(str(LORA_DIR))
    adapter_size_mb = get_dir_size_mb(LORA_DIR)
    print(f"final training loss = {final_loss:.6f}")
    print(f"lora adapter size = {adapter_size_mb:.6f} MB")
    write_metrics(base_model_total_params, lora_trainable_params, combined_model_total_params, final_loss, adapter_size_mb)
    print(f"lora adapter saved to: {LORA_DIR}")


def main():
    train()


if __name__ == "__main__":
    main()
