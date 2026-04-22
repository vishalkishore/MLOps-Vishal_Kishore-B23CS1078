#!/usr/bin/env bash
set -euo pipefail

ONNXRUNTIME_ROOT="${ONNXRUNTIME_ROOT:-/opt/onnxruntime-linux-x64-1.20.1}"

echo "[verify] python version"
python3 --version

echo "[verify] import checks"
python3 - <<'PY'
import importlib

modules = [
    "numpy",
    "onnx",
    "onnxruntime",
    "PIL",
    "torch",
    "diffusers",
    "transformers",
    "peft",
    "huggingface_hub",
]

for name in modules:
    module = importlib.import_module(name)
    print(f"{name}={getattr(module, '__version__', 'ok')}")
PY

echo "[verify] onnxruntime shared library"
test -f "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
ls -l "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"

echo "[verify] cmake"
cmake --version | head -n 1

echo "[verify] g++"
g++ --version | head -n 1

echo "[verify] question1 python files compile"
python3 -m py_compile \
  /workspace/common.py \
  /workspace/export_onnx.py \
  /workspace/prepare_prompt.py \
  /workspace/infer_onnx.py \
  /workspace/compare_results.py

echo "[verify] environment ready"
