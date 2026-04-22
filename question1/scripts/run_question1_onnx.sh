#!/usr/bin/env bash
set -euo pipefail

ONNXRUNTIME_ROOT="${ONNXRUNTIME_ROOT:-/opt/onnxruntime-linux-x64-1.20.1}"

if [[ $# -eq 0 ]]; then
  cat <<'EOF'
Usage:
  ./scripts/run_question1_onnx.sh train
  ./scripts/run_question1_onnx.sh verify
  ./scripts/run_question1_onnx.sh export
  ./scripts/run_question1_onnx.sh prompt
  ./scripts/run_question1_onnx.sh python
  ./scripts/run_question1_onnx.sh cpp-build
  ./scripts/run_question1_onnx.sh cpp-run
  ./scripts/run_question1_onnx.sh compare
EOF
  exit 1
fi

case "${1}" in
  train)
    python3 /workspace/train_lora.py "${@:2}"
    ;;
  verify)
    /usr/local/bin/verify_onnx_env.sh
    ;;
  export)
    python3 /workspace/export_onnx.py "${@:2}"
    ;;
  prompt)
    python3 /workspace/prepare_prompt.py "${@:2}"
    ;;
  python)
    python3 /workspace/infer_onnx.py "${@:2}"
    ;;
  cpp-build)
    mkdir -p /workspace/cpp/build
    cmake -S /workspace/cpp -B /workspace/cpp/build -DONNXRUNTIME_ROOT="${ONNXRUNTIME_ROOT}"
    cmake --build /workspace/cpp/build -j"$(nproc)"
    ;;
  cpp-run)
    /workspace/cpp/build/question1_onnx_cpp "${@:2}"
    ;;
  compare)
    python3 /workspace/compare_results.py "${@:2}"
    ;;
  *)
    echo "Unknown command: ${1}" >&2
    exit 1
    ;;
esac
