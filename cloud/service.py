"""
Thin HTTP wrapper so the orchestrator can run as a deployed service on
Azure Container Apps. POST an incident, get the triage result back.

Local agents/graph logic is unchanged — this just exposes it over HTTP.
"""

from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel

from graph.build import build_graph

app = FastAPI(title="Agentic Inference Orchestrator")
_graph = None


class IncidentRequest(BaseModel):
    incident: str


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/triage")
def triage(req: IncidentRequest):
    final = _get_graph().invoke({"incident": req.incident})
    return {
        "diagnosis": final.get("diagnosis"),
        "confidence": final.get("confidence"),
        "validated": final.get("validated"),
        "escalated": final.get("escalated", False),
        "escalation_reason": final.get("escalation_reason"),
    }
