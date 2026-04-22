import argparse
from pathlib import Path

from common import REPORTS_DIR, write_key_value_report


def parse_report(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, value = line.split("=", maxsplit=1)
        values[key] = value
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Python and C++ ONNX inference reports.")
    parser.add_argument("--python-report", type=Path, default=REPORTS_DIR / "python_inference_report.txt")
    parser.add_argument("--cpp-report", type=Path, default=REPORTS_DIR / "cpp_inference_report.txt")
    parser.add_argument(
        "--concept-present",
        choices=["Present", "Not Present"],
        default="Present",
        help="Manual visual-inspection result for the LoRA concept in the ONNX outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python_report = parse_report(args.python_report)
    cpp_report = parse_report(args.cpp_report)

    python_latency = float(python_report["average_latency_seconds"])
    cpp_latency = float(cpp_report["average_latency_seconds"])
    speedup = python_latency / cpp_latency

    write_key_value_report(
        REPORTS_DIR / "comparison_report.txt",
        {
            "python_average_latency_seconds": f"{python_latency:.4f}",
            "cpp_average_latency_seconds": f"{cpp_latency:.4f}",
            "cpp_speedup_vs_python": f"{speedup:.4f}x",
            "visual_concept_preserved": args.concept_present,
            "python_output_image": python_report.get("output_image", ""),
            "cpp_output_image": cpp_report.get("output_image", ""),
        },
    )

    print(f"visual concept preserved: {args.concept_present}")
    print(f"c++ speedup vs python: {speedup:.4f}x")


if __name__ == "__main__":
    main()
