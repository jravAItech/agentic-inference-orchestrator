"""
Runnable demo. Feeds a sample incident through the full agent graph and
prints the trace. Works in Phase 1 with just an API key set:

    export ANTHROPIC_API_KEY=...      # or swap APIClient for your provider
    python -m demo.run

Flip to the GPU backend (Phase 2) with:

    export INFERENCE_BACKEND=triton
    python -m demo.run
"""

from graph.build import build_graph

SAMPLE_INCIDENT = (
    "Alert DB-POOL-503: checkout-service returning 503s for the last "
    "8 minutes. Logs repeat 'connection pool exhausted'. Traffic is normal. "
    "Started shortly after the 14:02 deploy."
)


def main():
    app = build_graph()
    print("=== Incident ===")
    print(SAMPLE_INCIDENT, "\n")

    final = app.invoke({"incident": SAMPLE_INCIDENT})

    print("=== Diagnosis ===")
    print(final.get("diagnosis", "(none)"), "\n")
    print("=== Validation ===")
    print(final.get("validation_note", "(none)"))
    print(f"validated={final.get('validated')} "
          f"confidence={final.get('confidence')}")
    if final.get("escalated"):
        print("\n=== ESCALATED ===")
        print(final.get("escalation_reason"))


if __name__ == "__main__":
    main()
