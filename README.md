# Agentic Inference Orchestrator

A self-hosted, GPU-accelerated **multi-agent system for autonomous incident
triage**. Five specialist agents coordinate under a LangGraph state machine
to take a raw alert → retrieve context → diagnose root cause → validate the
fix → escalate when confidence is low. The inference layer runs on a
self-hosted **TensorRT-LLM** model served by **Triton**, with full agent and
GPU observability.

> Built to demonstrate three things that rarely appear together in one repo:
> **agentic orchestration**, **production LLM serving on the NVIDIA stack**,
> and **measured inference optimization**.

---

## Architecture

```
                 ┌──────────────────────────────────────────┐
   incident ──▶  │            LangGraph Orchestrator          │
                 │   (confidence-gated conditional routing)   │
                 └───┬───────────┬───────────┬────────────┬───┘
                     ▼           ▼           ▼            ▼
                 Retriever  Diagnostician  Validator  Escalation
                     │           │           │
                     ▼           ▼           ▼
                  RAG layer   ┌─────────────────────────────┐
                 (hybrid)     │      InferenceClient         │
                              │  Phase 1: API  │ Phase 2: ───┼──▶ Triton
                              └─────────────────────────────┘     + TensorRT-LLM
                                                                   (FP16 ▸ INT8/FP8)
        observability: Prometheus + Grafana  (agent steps + GPU metrics)
```

The **`InferenceClient`** interface is the design hinge: the agent graph is
backend-agnostic, so flipping from a hosted API (Phase 1) to a self-hosted
GPU engine (Phase 2) is a one-line change — `INFERENCE_BACKEND=triton`.

---

## Why this design

- **Confidence-gated routing** — the orchestrator escalates to a human when
  diagnostic confidence drops or validation fails, instead of confidently
  shipping a wrong fix.
- **Hybrid retrieval** — keyword + semantic, because operators search by
  fault codes and IDs, not just intent. Semantic-only retrieval misses them.
- **Validator agent** — a dedicated pass that flags steps referencing tools
  or commands absent from the retrieved context (hallucination guard).
- **Backend-agnostic inference** — agents never know or care whether the
  model is an API or a self-hosted TensorRT-LLM engine.

---

## Cloud (Azure) — opt-in

The project runs $0 locally by default. Set env vars to route the data and
hosting layers through Azure — same agents, new backends, nothing changes in
the graph logic:

| Layer            | Local default                | Azure backend                  |
|------------------|------------------------------|--------------------------------|
| Corpus storage   | local files (`rag/corpus`)   | **Azure Blob** (`CORPUS_BACKEND=azure_blob`) |
| Retrieval        | sentence-transformers        | **Azure AI Search** hybrid (`RETRIEVER_BACKEND=azure_search`) |
| Hosting          | local script / FastAPI       | **Azure Container Apps** (`cloud/deploy_azure.sh`) |

Azure AI Search runs keyword + semantic in one fused query — the same hybrid
pattern that matters when operators search by fault codes *and* intent.
Provision + deploy everything with `cloud/deploy_azure.sh` (free/low tiers).

---

## Quickstart (Phase 1 — no GPU needed)

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # or swap APIClient for any provider
python -m demo.run
```

This runs a sample incident end-to-end through all five agents and prints
the full trace.

## Phase 2 — GPU inference backend ($0 on free Colab/Kaggle)

Compile a small model with TensorRT-LLM, quantize it, serve via Triton, and
benchmark FP16 vs. INT8. Full walkthrough in [`gpu/README.md`](gpu/README.md).

```bash
export INFERENCE_BACKEND=triton
python -m demo.run
```

---

## Benchmarks

*Populated after the Phase 2 sprint (small model on a free T4).*

| Config         | Tokens/sec | P50 (ms) | P99 (ms) | GPU util |
|----------------|-----------:|---------:|---------:|---------:|
| FP16 baseline  |     _TBD_  |   _TBD_  |   _TBD_  |  _TBD_   |
| INT8 quantized |     _TBD_  |   _TBD_  |   _TBD_  |  _TBD_   |

The optimization delta (baseline → quantized) is the headline result.

---

## Repo layout

```
agents/         five specialist agents (single-purpose nodes)
graph/          LangGraph state machine + backend swap point
inference/      InferenceClient interface + API & Triton backends
rag/            hybrid retriever + runbook corpus
gpu/            TensorRT-LLM compile, Triton config, benchmark, Colab notebook
observability/  Prometheus + Grafana configs
benchmarks/     results + comparison charts (Phase 2 artifacts)
demo/           runnable end-to-end example
```

## Roadmap
- [x] Phase 1 — agentic graph, hybrid RAG, runnable demo
- [ ] Phase 2 — TensorRT-LLM + Triton backend, benchmarks
- [ ] Phase 3 — Grafana dashboards, demo GIF
- [ ] v2 — speculative routing (small model for easy steps, large for hard)
```
