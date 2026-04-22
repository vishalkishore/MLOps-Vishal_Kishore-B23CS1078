from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi
from dotenv import load_dotenv


load_dotenv()
HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")


@dataclass(slots=True)
class HuggingFaceUploadConfig:
    local_dir: str | Path
    repo_id: str
    commit_message: str = "Upload model artifacts"
    private: bool = False
    exist_ok: bool = True
    create_pr: bool = False
    revision: str | None = None
    token: str | None = None
    model_card: str | None = None


def get_hf_token(explicit_token: str | None = None) -> str:
    if explicit_token:
        return explicit_token

    for env_var in HF_TOKEN_ENV_VARS:
        token = os.getenv(env_var)
        if token:
            return token

    raise ValueError(
        "Missing Hugging Face token. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN."
    )


def build_model_card(
    *,
    model_name: str,
    base_model: str | None = None,
    datasets: list[str] | None = None,
    tags: list[str] | None = None,
    license_name: str | None = None,
    summary: str | None = None,
) -> str:
    lines = ["---", "library_name: transformers", f"model_name: {model_name}"]

    if base_model:
        lines.append(f"base_model: {base_model}")
    if datasets:
        lines.append("datasets:")
        lines.extend(f"  - {dataset}" for dataset in datasets)
    if tags:
        lines.append("tags:")
        lines.extend(f"  - {tag}" for tag in tags)
    if license_name:
        lines.append(f"license: {license_name}")

    lines.extend(
        [
            "---",
            "",
            f"# {model_name}",
            "",
            summary or f"Model artifacts for `{model_name}`.",
            "",
        ]
    )
    return "\n".join(lines)


def upload_model_to_hub(config: HuggingFaceUploadConfig) -> dict[str, str]:
    local_dir = Path(config.local_dir).expanduser().resolve()
    if not local_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {local_dir}")
    if not local_dir.is_dir():
        raise NotADirectoryError(f"Expected directory, got: {local_dir}")

    api = HfApi(token=get_hf_token(config.token))
    api.create_repo(
        repo_id=config.repo_id,
        repo_type="model",
        private=config.private,
        exist_ok=config.exist_ok,
    )

    if config.model_card:
        readme_path = local_dir / "README.md"
        if not readme_path.exists():
            readme_path.write_text(config.model_card, encoding="utf-8")

    commit_info = api.upload_folder(
        repo_id=config.repo_id,
        repo_type="model",
        folder_path=str(local_dir),
        commit_message=config.commit_message,
        revision=config.revision,
        create_pr=config.create_pr,
    )

    return {
        "repo_id": config.repo_id,
        "repo_url": f"https://huggingface.co/{config.repo_id}",
        "commit_url": str(commit_info.commit_url),
        "local_dir": str(local_dir),
    }
