import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from common import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_NUM_RUNS,
    DEFAULT_SEED,
    DEFAULT_WIDTH,
    IMAGES_DIR,
    ONNX_DIR,
    PROMPT_IDS_PATH,
    REPORTS_DIR,
    ensure_runtime_dirs,
    read_json,
    write_key_value_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stable Diffusion ONNX inference with onnxruntime.")
    parser.add_argument("--onnx-dir", type=Path, default=ONNX_DIR)
    parser.add_argument("--prompt-ids", type=Path, default=PROMPT_IDS_PATH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--runs", type=int, default=DEFAULT_NUM_RUNS)
    parser.add_argument(
        "--provider",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Choose a specific ONNX Runtime execution provider for fair latency comparisons.",
    )
    return parser.parse_args()


def make_betas(num_train_timesteps: int, beta_start: float, beta_end: float) -> np.ndarray:
    return np.linspace(np.sqrt(beta_start), np.sqrt(beta_end), num_train_timesteps, dtype=np.float64) ** 2


def make_ddim_schedule(num_train_timesteps: int, num_inference_steps: int) -> np.ndarray:
    return np.linspace(0, num_train_timesteps - 1, num_inference_steps, dtype=np.int64)[::-1].copy()


def ddim_step(noise_pred: np.ndarray, timestep_index: int, sample: np.ndarray, alphas_cumprod: np.ndarray, timesteps: np.ndarray) -> np.ndarray:
    timestep = int(timesteps[timestep_index])
    prev_timestep = int(timesteps[timestep_index + 1]) if timestep_index + 1 < len(timesteps) else -1

    alpha_prod_t = alphas_cumprod[timestep]
    alpha_prod_prev = 1.0 if prev_timestep < 0 else alphas_cumprod[prev_timestep]
    beta_prod_t = 1.0 - alpha_prod_t

    pred_original_sample = (sample - np.sqrt(beta_prod_t) * noise_pred) / np.sqrt(alpha_prod_t)
    pred_sample_direction = np.sqrt(1.0 - alpha_prod_prev) * noise_pred
    prev_sample = np.sqrt(alpha_prod_prev) * pred_original_sample + pred_sample_direction
    return prev_sample.astype(np.float32)


def save_image(decoded: np.ndarray, path: str) -> None:
    image = decoded[0].transpose(1, 2, 0)
    image = ((image / 2.0) + 0.5).clip(0.0, 1.0)
    image = (image * 255.0).round().astype(np.uint8)
    Image.fromarray(image).save(path)


def main() -> None:
    args = parse_args()
    ensure_runtime_dirs()

    prompt_payload = read_json(args.prompt_ids)
    prompt_ids = np.array([prompt_payload["prompt_ids"]], dtype=np.int64)
    negative_prompt_ids = np.array([prompt_payload["negative_prompt_ids"]], dtype=np.int64)

    available_providers = ort.get_available_providers()
    if args.provider == "cpu":
        providers = ["CPUExecutionProvider"]
    elif args.provider == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = [provider for provider in providers if provider in available_providers] or ["CPUExecutionProvider"]

    session_options = ort.SessionOptions()
    text_encoder = ort.InferenceSession(str(args.onnx_dir / "text_encoder.onnx"), sess_options=session_options, providers=providers)
    unet = ort.InferenceSession(str(args.onnx_dir / "unet.onnx"), sess_options=session_options, providers=providers)
    vae_decoder = ort.InferenceSession(str(args.onnx_dir / "vae_decoder.onnx"), sess_options=session_options, providers=providers)

    conditional_hidden = text_encoder.run(None, {"input_ids": prompt_ids})[0].astype(np.float32)
    unconditional_hidden = text_encoder.run(None, {"input_ids": negative_prompt_ids})[0].astype(np.float32)
    encoder_hidden_states = np.concatenate([unconditional_hidden, conditional_hidden], axis=0).astype(np.float32)

    betas = make_betas(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012)
    alphas_cumprod = np.cumprod(1.0 - betas, axis=0)
    timesteps = make_ddim_schedule(num_train_timesteps=1000, num_inference_steps=args.num_steps)

    latencies = []
    last_output_path = IMAGES_DIR / "python_onnx_output.png"

    for run_index in range(args.runs):
        rng = np.random.default_rng(args.seed + run_index)
        latents = rng.standard_normal((1, 4, args.height // 8, args.width // 8), dtype=np.float32)

        start_time = time.perf_counter()
        for timestep_index, timestep in enumerate(timesteps):
            latent_model_input = np.concatenate([latents, latents], axis=0).astype(np.float32)
            timestep_input = np.array([timestep], dtype=np.int64)
            noise_pred = unet.run(
                None,
                {
                    "latents": latent_model_input,
                    "timestep": timestep_input,
                    "encoder_hidden_states": encoder_hidden_states,
                },
            )[0].astype(np.float32)

            noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2, axis=0)
            guided_noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = ddim_step(guided_noise_pred, timestep_index, latents, alphas_cumprod, timesteps)

        decoded = vae_decoder.run(None, {"latents": latents / 0.18215})[0].astype(np.float32)
        latency = time.perf_counter() - start_time
        latencies.append(latency)

        if run_index == args.runs - 1:
            save_image(decoded, str(last_output_path))

    average_latency = sum(latencies) / len(latencies)
    report = {
        "prompt": prompt_payload["prompt"],
        "negative_prompt": prompt_payload["negative_prompt"],
        "providers": ",".join(providers),
        "num_steps": args.num_steps,
        "runs": args.runs,
        "seed_start": args.seed,
        "average_latency_seconds": f"{average_latency:.4f}",
        "output_image": last_output_path.resolve(),
    }
    write_key_value_report(REPORTS_DIR / "python_inference_report.txt", report)

    print(f"python average latency: {average_latency:.4f} seconds")
    print(f"python output image saved to: {last_output_path}")


if __name__ == "__main__":
    main()
