# Phase 2 — GPU Inference Backend (TensorRT-LLM + Triton)

This is the layer that turns the project from "another LangGraph demo" into
proof of GPU + LLM serving mastery. Everything here runs on a **free Colab
T4** or **Kaggle** — no local hardware, $0.

## Why a small model is fine
We serve a small model (Llama-3.2-3B-Instruct or Phi-3-mini). The point is
not model size — it's demonstrating the full NVIDIA inference path:
**compile (TensorRT-LLM) → quantize (FP8/INT8) → serve (Triton) → benchmark.**
The benchmark numbers are real regardless of model size.

## Steps (run in gpu/colab_phase2.ipynb)

### 1. Environment
```bash
pip install tensorrt_llm -U --extra-index-url https://pypi.nvidia.com
pip install tritonclient[all]
```

### 2. Compile the model with TensorRT-LLM
```bash
# Convert HF checkpoint -> TensorRT-LLM checkpoint, then build engine.
python convert_checkpoint.py \
  --model_dir ./Llama-3.2-3B-Instruct \
  --output_dir ./trt_ckpt \
  --dtype float16

trtllm-build \
  --checkpoint_dir ./trt_ckpt \
  --output_dir ./trt_engine \
  --gemm_plugin float16 \
  --max_batch_size 8 \
  --paged_kv_cache enable      # KV-cache paging for throughput
```

### 3. Quantize (the optimization that produces the "after" benchmark)
```bash
# FP8 (Hopper) or INT8 weight-only (broad support, works on T4).
python quantize.py \
  --model_dir ./Llama-3.2-3B-Instruct \
  --qformat int8_wo \
  --output_dir ./trt_ckpt_int8
# rebuild engine from ./trt_ckpt_int8
```

### 4. Serve with Triton
See `triton_model_repo/` layout below. Launch:
```bash
tritonserver --model-repository=./triton_model_repo
```
The agent graph hits this endpoint by setting `INFERENCE_BACKEND=triton`.

### 5. Benchmark (capture into ../benchmarks/)
```bash
# Throughput + latency under load.
python benchmark.py --url localhost:8000 --concurrency 1,4,8 \
  --out ../benchmarks/results.json
```
Record: tokens/sec, P50/P99 latency, GPU utilization — **FP16 baseline vs.
INT8 quantized.** The delta is the headline number for the README.

## Triton model repository layout
```
triton_model_repo/
├── ensemble/            # preprocess -> trt_llm -> postprocess
│   └── config.pbtxt
├── tensorrt_llm/        # the compiled engine
│   ├── 1/
│   └── config.pbtxt
├── preprocessing/       # tokenizer
└── postprocessing/      # detokenizer
```

## What lands in the repo permanently
Even after you tear down the Colab/Kaggle session:
- `benchmarks/results.json` + a comparison chart
- Grafana screenshots of GPU + agent metrics
- The compile/quantize scripts and Triton config

These artifacts are the proof — they outlive the rented/free GPU session.
