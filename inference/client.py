"""
InferenceClient — the single abstraction that decouples the agent graph from
*where* the LLM actually runs.

Phase 1: agents call an API-backed client (fast to demo, no GPU needed).
Phase 2: swap to TritonClient pointing at a self-hosted TensorRT-LLM engine.
The agent code never changes — only which client is injected.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import time


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class InferenceClient(ABC):
    """Backend-agnostic generation interface."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512,
                 temperature: float = 0.2) -> GenerationResult:
        ...


class APIClient(InferenceClient):
    """
    Phase 1 backend. Calls a hosted model API.
    Defaults to Anthropic; swap the request body for any provider.
    Set the relevant API key in your environment before running.
    """

    def __init__(self, model: str = "claude-3-5-haiku-20241022"):
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 512,
                 temperature: float = 0.2) -> GenerationResult:
        import urllib.request
        import json

        start = time.perf_counter()
        body = json.dumps({
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "content-type": "application/json",
                "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())

        latency_ms = (time.perf_counter() - start) * 1000
        text = data["content"][0]["text"]
        usage = data.get("usage", {})
        return GenerationResult(
            text=text,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            latency_ms=latency_ms,
        )


class TritonClient(InferenceClient):
    """
    Phase 2 backend. Hits a Triton Inference Server endpoint serving a
    TensorRT-LLM-compiled model. Same interface as APIClient, so the agent
    graph swaps backends with a one-line change in the graph builder.

    See gpu/ for the TensorRT-LLM compile script and Triton model config,
    and gpu/colab_phase2.ipynb for a $0 walkthrough on a free T4.
    """

    def __init__(self, url: str = "localhost:8000",
                 model: str = "ensemble"):
        self.url = url
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 512,
                 temperature: float = 0.2) -> GenerationResult:
        import tritonclient.http as httpclient
        import numpy as np

        client = httpclient.InferenceServerClient(url=self.url)
        start = time.perf_counter()

        inp = httpclient.InferInput("text_input", [1], "BYTES")
        inp.set_data_from_numpy(np.array([prompt.encode()], dtype=object))
        max_tok = httpclient.InferInput("max_tokens", [1], "INT32")
        max_tok.set_data_from_numpy(np.array([max_tokens], dtype=np.int32))

        result = client.infer(self.model, inputs=[inp, max_tok])
        latency_ms = (time.perf_counter() - start) * 1000
        text = result.as_numpy("text_output")[0].decode()

        # Token counts come from Triton's response stats when configured;
        # benchmarks/ captures tokens/sec and P50/P99 from the load test.
        return GenerationResult(
            text=text,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            latency_ms=latency_ms,
        )


def get_client(backend: str | None = None) -> InferenceClient:
    """Factory. Set INFERENCE_BACKEND=triton to flip Phase 1 -> Phase 2."""
    backend = backend or os.environ.get("INFERENCE_BACKEND", "api")
    if backend == "triton":
        return TritonClient()
    return APIClient()
