from pathlib import Path
import sys
import os
os.environ["HF_TOKEN"] = "HF_TOKEN"
from huggingface_hub import login, upload_folder


REPO_ID = "vishalkishore01/assignment-5"
BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "q1"
    folder_path = (BASE_DIR / target).resolve()

    if not folder_path.is_dir():
        raise SystemExit(f"Folder not found: {folder_path}")

    login()
    upload_folder(
        folder_path=str(folder_path),
        path_in_repo=target,
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"Uploaded {folder_path} to {REPO_ID}/{target}")


if __name__ == "__main__":
    main()
