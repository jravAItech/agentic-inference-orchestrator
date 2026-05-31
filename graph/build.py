"""
LangGraph wiring. The orchestrator's route() drives conditional edges;
every node updates shared state. Swap the inference backend here in ONE
place (get_client) to flip Phase 1 (API) -> Phase 2 (Triton/GPU).
"""

from __future__ import annotations
from typing import TypedDict
from langgraph.graph import StateGraph, END

from inference.client import get_client
from rag.retriever import get_retriever
from agents.agents import (
    Orchestrator, Retriever, Diagnostician, Validator, Escalation,
)


class IncidentState(TypedDict, total=False):
    incident: str
    context: list
    diagnosis: str
    confidence: float
    validated: bool
    validation_note: str
    escalated: bool
    escalation_reason: str


def build_graph(corpus_dir: str = "rag/corpus"):
    client = get_client()  # <-- single swap point for Phase 1 / Phase 2
    retriever = get_retriever(corpus_dir)  # local | azure_search backend

    orch = Orchestrator(client)
    nodes = {
        "retriever": Retriever(retriever),
        "diagnostician": Diagnostician(client),
        "validator": Validator(client),
        "escalation": Escalation(),
    }

    g = StateGraph(IncidentState)
    g.add_node("retriever", lambda s: nodes["retriever"].run(s))
    g.add_node("diagnostician", lambda s: nodes["diagnostician"].run(s))
    g.add_node("validator", lambda s: nodes["validator"].run(s))
    g.add_node("escalation", lambda s: nodes["escalation"].run(s))

    g.set_entry_point("retriever")
    g.add_edge("retriever", "diagnostician")

    # After diagnosis, orchestrator decides: validate or escalate.
    g.add_conditional_edges(
        "diagnostician",
        lambda s: orch.route(s),
        {"validator": "validator", "escalation": "escalation"},
    )
    # After validation: done if valid, else escalate.
    g.add_conditional_edges(
        "validator",
        lambda s: "done" if s.get("validated") else "escalation",
        {"done": END, "escalation": "escalation"},
    )
    g.add_edge("escalation", END)

    return g.compile()
