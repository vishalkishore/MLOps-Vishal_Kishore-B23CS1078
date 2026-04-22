import argparse
from pathlib import Path

import onnx
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import PeftModel
from transformers import CLIPTextModel

from common import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    LORA_DIR,
    MODEL_DIR,
    ONNX_DIR,
    REPORTS_DIR,
    bytes_to_gb,
    dir_size_bytes,
    ensure_runtime_dirs,
    write_key_value_report,
)


class UNetOnnxWrapper(torch.nn.Module):
    def __init__(self, unet: UNet2DConditionModel) -> None:
        super().__init__()
        self.unet = unet

    def forward(self, latents, timestep, encoder_hidden_states):
        return self.unet(
            latents,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]


class VaeDecoderOnnxWrapper(torch.nn.Module):
    def __init__(self, vae: AutoencoderKL) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        return self.vae.decode(latents, return_dict=False)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stable Diffusion LoRA pipeline to ONNX.")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--lora-dir", type=Path, default=LORA_DIR)
    parser.add_argument("--output-dir", type=Path, default=ONNX_DIR)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    return parser.parse_args()


def export_model(model, sample_args, output_path: Path, input_names, output_names, dynamic_axes, opset: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            args=sample_args,
            f=str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            external_data=True,
        )


def verify_onnx_model(path: Path) -> None:
    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model)


def exported_model_size_bytes(path: Path) -> int:
    total = 0
    for candidate in path.parent.glob(f"{path.name}*"):
        if candidate.is_file():
            total += candidate.stat().st_size
    return total


def main() -> None:
    args = parse_args()
    ensure_runtime_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")

    text_encoder = CLIPTextModel.from_pretrained(args.model_dir, subfolder="text_encoder").to(device)
    text_encoder.eval()

    base_unet = UNet2DConditionModel.from_pretrained(args.model_dir, subfolder="unet").to(device)
    merged_unet = PeftModel.from_pretrained(base_unet, args.lora_dir)
    merged_unet = merged_unet.merge_and_unload()
    merged_unet.eval()

    vae = AutoencoderKL.from_pretrained(args.model_dir, subfolder="vae").to(device)
    vae.eval()

    text_encoder_path = args.output_dir / "text_encoder.onnx"
    unet_path = args.output_dir / "unet.onnx"
    vae_decoder_path = args.output_dir / "vae_decoder.onnx"

    sample_input_ids = torch.zeros((1, 77), dtype=torch.int64, device=device)
    sample_latents = torch.randn((2, 4, args.height // 8, args.width // 8), dtype=torch.float32, device=device)
    sample_timestep = torch.tensor([999], dtype=torch.int64, device=device)
    sample_hidden_states = torch.randn((2, 77, 768), dtype=torch.float32, device=device)
    sample_decoder_latents = torch.randn((1, 4, args.height // 8, args.width // 8), dtype=torch.float32, device=device)

    export_model(
        text_encoder,
        (sample_input_ids,),
        text_encoder_path,
        input_names=["input_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset=args.opset,
    )
    verify_onnx_model(text_encoder_path)

    export_model(
        UNetOnnxWrapper(merged_unet),
        (sample_latents, sample_timestep, sample_hidden_states),
        unet_path,
        input_names=["latents", "timestep", "encoder_hidden_states"],
        output_names=["noise_pred"],
        dynamic_axes={
            "latents": {0: "batch", 2: "latent_height", 3: "latent_width"},
            "encoder_hidden_states": {0: "batch", 1: "sequence"},
            "noise_pred": {0: "batch", 2: "latent_height", 3: "latent_width"},
        },
        opset=args.opset,
    )
    verify_onnx_model(unet_path)

    export_model(
        VaeDecoderOnnxWrapper(vae),
        (sample_decoder_latents,),
        vae_decoder_path,
        input_names=["latents"],
        output_names=["images"],
        dynamic_axes={
            "latents": {0: "batch", 2: "latent_height", 3: "latent_width"},
            "images": {0: "batch", 2: "image_height", 3: "image_width"},
        },
        opset=args.opset,
    )
    verify_onnx_model(vae_decoder_path)

    baseline_size_bytes = dir_size_bytes(args.model_dir)
    onnx_size_bytes = (
        exported_model_size_bytes(text_encoder_path)
        + exported_model_size_bytes(unet_path)
        + exported_model_size_bytes(vae_decoder_path)
    )

    report = {
        "baseline_model_dir": args.model_dir.resolve(),
        "onnx_output_dir": args.output_dir.resolve(),
        "text_encoder_onnx": text_encoder_path.resolve(),
        "unet_onnx": unet_path.resolve(),
        "vae_decoder_onnx": vae_decoder_path.resolve(),
        "baseline_model_size_gb": f"{bytes_to_gb(baseline_size_bytes):.4f}",
        "onnx_combined_size_gb": f"{bytes_to_gb(onnx_size_bytes):.4f}",
        "export_verified": "true",
    }
    write_key_value_report(REPORTS_DIR / "onnx_export_report.txt", report)

    print(f"text encoder exported to: {text_encoder_path}")
    print(f"unet exported to: {unet_path}")
    print(f"vae decoder exported to: {vae_decoder_path}")
    print(f"baseline model size: {bytes_to_gb(baseline_size_bytes):.4f} GB")
    print(f"combined ONNX size: {bytes_to_gb(onnx_size_bytes):.4f} GB")


if __name__ == "__main__":
    main()
