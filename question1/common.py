import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "model"
DATASET_DIR = ARTIFACTS_DIR / "dataset"
LORA_DIR = ARTIFACTS_DIR / "lora_unet"
ONNX_DIR = ARTIFACTS_DIR / "onnx"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
IMAGES_DIR = ARTIFACTS_DIR / "images"
PROMPT_IDS_PATH = ARTIFACTS_DIR / "prompt_ids.json"

DEFAULT_PROMPT = "naruto uzumaki standing in a village street, anime style, highly detailed"
DEFAULT_NEGATIVE_PROMPT = ""
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_NUM_INFERENCE_STEPS = 20
DEFAULT_SEED = 42
DEFAULT_NUM_RUNS = 5


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_runtime_dirs() -> None:
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0

    if path.is_file():
        return path.stat().st_size

    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total


def bytes_to_gb(size_bytes: int) -> float:
    return size_bytes / (1024 ** 3)


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_key_value_report(path: Path, values: dict[str, object]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for key, value in values.items():
            handle.write(f"{key}={value}\n")
