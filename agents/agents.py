"""
The five specialist agents. Each takes the shared state, does one job, and
returns an updated state. Wiring lives in graph/build.py.

Design note: agents are deliberately thin and single-purpose. The
orchestrator owns routing decisions; specialists never call each other
directly. This keeps the graph auditable — every transition is a logged
edge, which is also what makes the system demo-able and interview-ready.
"""

from __future__ import annotations
from inference.client import InferenceClient
from rag.retriever import RunbookRetriever


class Orchestrator:
    """Decides the next step based on current state and confidence."""

    def __init__(self, client: InferenceClient):
        self.client = client

    def route(self, state: dict) -> str:
        # Confidence-gated routing: low diagnostic confidence -> escalate.
        if state.get("diagnosis") and state.get("confidence", 1.0) < 0.6:
            return "escalation"
        if not state.get("context"):
            return "retriever"
        if not state.get("diagnosis"):
            return "diagnostician"
        if not state.get("validated"):
            return "validator"
        return "done"


class Retriever:
    """Pulls relevant runbooks/logs for the incident via the RAG layer."""

    def __init__(self, retriever: RunbookRetriever):
        self.retriever = retriever

    def run(self, state: dict) -> dict:
        hits = self.retriever.search(state["incident"], k=4)
        state["context"] = hits
        return state


class Diagnostician:
    """Reasons over retrieved evidence to propose a root cause + fix."""

    def __init__(self, client: InferenceClient):
        self.client = client

    def run(self, state: dict) -> dict:
        ctx = "\n\n".join(state.get("context", []))
        prompt = (
            "You are an incident diagnostician. Given the incident and the "
            "retrieved runbook context, state the most likely root cause and "
            "a concrete remediation plan. End with a confidence score 0-1.\n\n"
            f"INCIDENT:\n{state['incident']}\n\nCONTEXT:\n{ctx}"
        )
        result = self.client.generate(prompt, max_tokens=600)
        state["diagnosis"] = result.text
        state["confidence"] = _parse_confidence(result.text)
        return state


class Validator:
    """Checks the proposed fix for hallucinated steps / unmet constraints."""

    def __init__(self, client: InferenceClient):
        self.client = client

    def run(self, state: dict) -> dict:
        prompt = (
            "You are a remediation validator. Review the proposed fix. Flag "
            "any step that references a tool, file, or command not present in "
            "the context. Respond VALID or INVALID with a one-line reason.\n\n"
            f"CONTEXT:\n{''.join(state.get('context', []))}\n\n"
            f"PROPOSED FIX:\n{state['diagnosis']}"
        )
        result = self.client.generate(prompt, max_tokens=200)
        state["validated"] = result.text.strip().upper().startswith("VALID")
        state["validation_note"] = result.text
        return state


class Escalation:
    """Human-in-the-loop handoff when confidence or validation fails."""

    def run(self, state: dict) -> dict:
        state["escalated"] = True
        state["escalation_reason"] = (
            f"confidence={state.get('confidence')} "
            f"validated={state.get('validated')}"
        )
        return state


def _parse_confidence(text: str) -> float:
    import re
    matches = re.findall(r"(?:confidence[:\s]*)?(0?\.\d+|1\.0)", text.lower())
    return float(matches[-1]) if matches else 0.5
