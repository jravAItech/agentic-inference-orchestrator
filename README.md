<div align="center">

# 🧠 Agentic Inference Orchestrator

### Autonomous incident triage, run by a team of coordinated AI agents — on a self-hosted GPU inference stack.

*Five specialist agents. One LangGraph brain. A TensorRT-LLM engine underneath. Azure-native by design.*

`agentic-ai` · `langgraph` · `tensorrt-llm` · `triton` · `azure` · `rag` · `mlops`

</div>

---

## What this is

Most "AI agent" demos are a single LLM call wearing a costume. This isn't that.

This is a **production-shaped multi-agent system** that takes a raw, messy
production alert and autonomously drives it to a validated remediation plan —
escalating to a human only when it isn't confident enough to act. The agents
don't run on someone else's API as a black box; they run on a **self-hosted,
GPU-accelerated inference engine** that *you* compile, quantize, serve, and
benchmark.

It's built to prove three things that rarely live in the same repo:

🤖 **Agentic orchestration** — a real multi-agent state machine, not a prompt chain
⚡ **GPU inference mastery** — self-hosted LLM serving on the NVIDIA stack (TensorRT-LLM + Triton)
☁️ **Cloud-native engineering** — Azure-backed storage, retrieval, and deployment

---

## The idea in one picture

```
                  ┌────────────────────────────────────────────┐
   raw alert ──▶  │             LangGraph Orchestrator           │
                  │     confidence-gated conditional routing     │
                  └──┬────────────┬────────────┬─────────────┬───┘
                     ▼            ▼            ▼             ▼
                 Retriever  Diagnostician  Validator   Escalation
                     │            │            │
                     ▼            ▼            ▼
                 RAG layer   ┌──────────────────────────────┐
                (hybrid)     │       InferenceClient         │
                             │  Phase 1: API │ Phase 2: ─────┼──▶ Triton + TensorRT-LLM
                             └──────────────────────────────┘     (FP16 ▸ INT8/FP8)

   ☁️ Azure Blob (corpus) · Azure AI Search (hybrid retrieval) · Container Apps (hosting)
   📊 Prometheus + Grafana (agent steps + GPU metrics)
```

The **`InferenceClient`** abstraction is the hinge of the whole design: the
agents have no idea whether the model behind them is a hosted API or a
self-hosted GPU engine. Flipping between them is a single environment variable.
That's what lets the project start free and scale into GPU + cloud without a
rewrite.

---

## Meet the agents

Each agent is a single-purpose node. The orchestrator owns all routing — agents
never call each other directly, so every decision is an auditable, logged edge.

| Agent | Job | Why it matters |
|-------|-----|----------------|
| 🧭 **Orchestrator** | Decides the next step from current state + confidence | Routes to escalation instead of shipping a low-confidence fix |
| 🔍 **Retriever** | Pulls relevant runbooks/logs via hybrid RAG | Operators search by *fault codes* and *intent* — needs both |
| 🩺 **Diagnostician** | Reasons over evidence → root cause + remediation | Produces a plan *and* a confidence score |
| ✅ **Validator** | Flags steps referencing tools/commands not in context | A dedicated hallucination guard before anything ships |
| 🚨 **Escalation** | Human-in-the-loop handoff | Fails safe, with a logged reason |

---

## Design decisions worth calling out

- **Confidence-gated routing.** The orchestrator escalates when diagnostic
  confidence drops or validation fails — the system would rather ask a human
  than confidently ship the wrong fix.
- **Hybrid retrieval, on purpose.** Semantic-only search misses the way real
  operators work — they search by part numbers, fault codes, and IDs. The
  retriever fuses keyword + semantic so both land.
- **A validator agent as a hallucination guard.** Every proposed fix is checked
  for steps that reference tools or commands absent from the retrieved context.
- **Backend-agnostic inference.** The same graph runs on a free API today and a
  self-hosted GPU engine tomorrow — one env var apart.

---

## Quickstart — run it in 5 minutes (no GPU, no cloud, $0)

**Step 1 — Get the code**
```bash
git clone https://github.com/jravAItech/agentic-inference-orchestrator.git
cd agentic-inference-orchestrator
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Set your model API key** (any provider; defaults to Anthropic)
```bash
export ANTHROPIC_API_KEY=sk-...
```

**Step 4 — Run the demo**
```bash
python -m demo.run
```

You'll watch a sample incident flow through all five agents end to end:
retrieval → diagnosis → validation → (escalation if confidence is low).

---

## Phase 2 — Self-hosted GPU inference ($0 on free Colab / Kaggle)

This is the layer that turns the project from "another LangGraph demo" into
proof of GPU + LLM serving mastery. No local hardware required — it runs on a
**free Colab T4** or **Kaggle GPU**.

**Step 1 — Open the Phase 2 guide:** [`gpu/README.md`](gpu/README.md)

**Step 2 — Compile a small model** (Llama-3.2-3B / Phi-3-mini) with **TensorRT-LLM**
**Step 3 — Quantize it** (INT8 weight-only on T4, or FP8 on Hopper)
**Step 4 — Serve it** with **Triton Inference Server**
**Step 5 — Point the agents at it:**
```bash
export INFERENCE_BACKEND=triton
python -m demo.run
```
**Step 6 — Benchmark** FP16 baseline vs. quantized:
```bash
python gpu/benchmark.py --url localhost:8000 --concurrency 1,4,8 --out benchmarks/results.json
```

> 💡 The point isn't model size — it's demonstrating the full NVIDIA inference
> path end to end: **compile → quantize → serve → benchmark**. The numbers are
> real regardless of how small the model is.

### Benchmarks

> ⏳ *Populated from the live Colab/Kaggle GPU run. Numbers reflect a small
> model on a free T4 — the optimization delta is the headline.*

| Config         | Tokens/sec | P50 (ms) | P99 (ms) | GPU util |
|----------------|-----------:|---------:|---------:|---------:|
| FP16 baseline  | _pending_  | _pending_| _pending_| _pending_|
| INT8 quantized | _pending_  | _pending_| _pending_| _pending_|

---

## Phase 3 — Go cloud-native on Azure (opt-in)

The project runs entirely $0 locally. Flip env vars to route the data and
hosting layers through Azure — **same agents, new backends, zero logic change.**

| Layer | Local default | Azure backend | Switch |
|-------|---------------|---------------|--------|
| 📦 Corpus | local files | **Azure Blob** | `CORPUS_BACKEND=azure_blob` |
| 🔎 Retrieval | sentence-transformers | **Azure AI Search** (hybrid) | `RETRIEVER_BACKEND=azure_search` |
| 🚀 Hosting | local / FastAPI | **Azure Container Apps** | `cloud/deploy_azure.sh` |

**Step 1 — Log in:** `az login`
**Step 2 — Edit the variables block** at the top of [`cloud/deploy_azure.sh`](cloud/deploy_azure.sh)
**Step 3 — Run it line by line** (first time, so you see each resource provision)
**Step 4 — Create the AI Search index** with a semantic config named `default`, then index the Blob docs
**Step 5 — Hit your deployed endpoint:**
```bash
curl -X POST https://<your-app>.azurecontainerapps.io/triage \
  -H "Content-Type: application/json" \
  -d '{"incident": "DB-POOL-503: checkout-service returning 503s, pool exhausted"}'
```

Azure AI Search runs keyword + semantic in a single fused query — the same
hybrid pattern that matters when operators search by fault code *and* intent.

---

## Tech stack

**Orchestration** LangGraph · Model Context Protocol (MCP)
**Inference** TensorRT-LLM · Triton Inference Server · INT8/FP8 quantization
**Retrieval** Hybrid RAG · sentence-transformers · Azure AI Search
**Cloud** Azure Blob Storage · Azure Container Apps · Azure Container Registry
**Observability** Prometheus · Grafana
**Serving** FastAPI · Docker

---

## Repo layout

```
agents/         five specialist agents (single-purpose nodes)
graph/          LangGraph state machine + backend swap point
inference/      InferenceClient interface + API & Triton backends
rag/            hybrid retriever + runbook corpus
cloud/          Azure Blob, AI Search, FastAPI service, deploy script
gpu/            TensorRT-LLM compile, Triton config, benchmark, Colab guide
observability/  Prometheus + Grafana configs
benchmarks/     results + comparison charts (Phase 2 artifacts)
demo/           runnable end-to-end example
```

---

## Roadmap

- [x] **Phase 1** — agentic graph, hybrid RAG, runnable demo
- [x] **Phase 3 (cloud)** — Azure Blob + AI Search + Container Apps backends
- [ ] **Phase 2 (GPU)** — TensorRT-LLM + Triton backend, benchmarks captured
- [ ] **Observability** — Grafana dashboards (agent + GPU metrics), demo GIF
- [ ] **v2** — speculative routing: small model for easy steps, large for hard

---

<div align="center">

**Built by Jash Raval** — AI/ML Tech Lead

*Production LLM systems · agentic workflows · GPU inference · cloud MLOps*

</div>
