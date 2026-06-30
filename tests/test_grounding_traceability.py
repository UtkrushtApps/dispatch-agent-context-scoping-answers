"""Checks that an answer can be tied back to a permitted source."""
from __future__ import annotations

from agent.context_builder import build_messages
from agent.retrieval import retrieve


def test_permitted_source_is_identifiable_in_context():
    ev = retrieve("hazmat pickup cutoff at Newark")
    msgs = build_messages("what is the hazmat pickup cutoff at Newark?", "CAR-NORTH", ev)
    text = "\n".join(m.get("content", "") for m in msgs)
    # The asking carrier's own authoritative material should be present and
    # attributable; a dispatcher must be able to see which source backs a claim.
    assert "SOP-NORTH-001" in text or "CAR-NORTH" in text, (
        "no identifiable permitted source present in assembled context"
    )
