# Question 1

This folder now covers Tasks 1 to 5 end to end:

1. Fine-tune Stable Diffusion v1.5 with LoRA on the UNet attention layers
2. Merge the trained LoRA adapter into the base model and export ONNX graphs
3. Run Python inference with `onnxruntime`
4. Run C++ inference with the ONNX Runtime C++ API
5. Compare outputs and compute the Python vs C++ speedup

## Added Files

- `common.py`
- `export_onnx.py`
- `prepare_prompt.py`
- `infer_onnx.py`
- `compare_results.py`
- `cpp/CMakeLists.txt`
- `cpp/main.cpp`

## Step 1: Train The LoRA Adapter

Run:

```bash
cd /workspace
python3 train.py
```

Expected artifacts:

- `artifacts/model`
- `artifacts/lora_unet`
- `artifacts/metrics.txt`

Use `artifacts/metrics.txt` for Task 1:

- `base_total_params`
- `lora_trainable_params`
- `combined_total_params`
- `final_training_loss`
- `adapter_size_mb`

## Step 2: Merge LoRA And Export ONNX

Run:

```bash
python3 export_onnx.py
```

This exports:

- `artifacts/onnx/text_encoder.onnx`
- `artifacts/onnx/unet.onnx`
- `artifacts/onnx/vae_decoder.onnx`

It also verifies the exported graphs and writes:

- [onnx_export_report.txt](/home/b23cs1078/MLOps-Vishal_Kishore-B23CS1078/question1/artifacts/reports/onnx_export_report.txt)

Use that report for Task 2:

- baseline model size: `baseline_model_size_gb`
- combined ONNX size: `onnx_combined_size_gb`

## Step 3: Prepare The Test Prompt

Run:

```bash
python3 prepare_prompt.py \
  --prompt "naruto uzumaki standing in a village street, anime style, highly detailed"
```

This writes shared token ids to:

- [prompt_ids.json](/home/b23cs1078/MLOps-Vishal_Kishore-B23CS1078/question1/artifacts/prompt_ids.json)

## Step 4: Python Inference Testing

Run:

```bash
python3 infer_onnx.py --provider cpu
```

Outputs:

- [python_inference_report.txt](/home/b23cs1078/MLOps-Vishal_Kishore-B23CS1078/question1/artifacts/reports/python_inference_report.txt)
- [python_onnx_output.png](/home/b23cs1078/MLOps-Vishal_Kishore-B23CS1078/question1/artifacts/images/python_onnx_output.png)

Use `average_latency_seconds` from the Python report for Task 3.

## Step 5: Build The C++ Program

Download and extract an official ONNX Runtime C++ release, then build with its path.

Example:

```bash
mkdir -p cpp/build
cd cpp/build
cmake .. -DONNXRUNTIME_ROOT=/workspace/third_party/onnxruntime-linux-x64-gpu-1.22.0
cmake --build . -j
```

## Step 6: C++ Inference Testing

Run from `question1/cpp/build`:

```bash
./question1_onnx_cpp \
  --onnx-dir ../../artifacts/onnx \
  --prompt-ids ../../artifacts/prompt_ids.json \
  --report ../../artifacts/reports/cpp_inference_report.txt \
  --output-image ../../artifacts/images/cpp_onnx_output.ppm
```

Outputs:

- [cpp_inference_report.txt](/home/b23cs1078/MLOps-Vishal_Kishore-B23CS1078/question1/artifacts/reports/cpp_inference_report.txt)
- [cpp_onnx_output.ppm](/home/b23cs1078/MLOps-Vishal_Kishore-B23CS1078/question1/artifacts/images/cpp_onnx_output.ppm)

Use `average_latency_seconds` from the C++ report for Task 4.

## Step 7: Comparative Analysis

Inspect the Python and C++ images, then run:

```bash
python3 compare_results.py --concept-present Present
```

This writes:

- [comparison_report.txt](/home/b23cs1078/MLOps-Vishal_Kishore-B23CS1078/question1/artifacts/reports/comparison_report.txt)

Use that report for Task 5:

- `visual_concept_preserved`
- `cpp_speedup_vs_python`

## Submission Mapping

- Task 2 answers come from `artifacts/reports/onnx_export_report.txt`
- Task 3 answer comes from `artifacts/reports/python_inference_report.txt`
- Task 4 answer comes from `artifacts/reports/cpp_inference_report.txt`
- Task 5 answers come from `artifacts/reports/comparison_report.txt`
