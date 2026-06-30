"""Candidate-facing invariants. Offline; no provider key required.

These exercise observable behavior of the assembled context. They do not mandate
a single design; they check the outcomes the business requires.
"""
from __future__ import annotations

import json

import pytest

from agent.context_builder import build_messages, answer
from agent.llm_client import LLMClient
from agent.retrieval import retrieve


def _assembled_text(messages):
    return "\n".join(m.get("content", "") for m in messages)


def test_messages_are_chat_shaped():
    ev = retrieve("hazmat pickup cutoff Newark")
    msgs = build_messages("what is the hazmat cutoff at Newark?", "CAR-NORTH", ev)
    assert isinstance(msgs, list) and msgs, "expected a non-empty list of chat messages"
    for m in msgs:
        assert "role" in m and "content" in m


def test_other_carrier_sop_not_in_context():
    ev = retrieve("hazmat pickup cutoff and weekend availability at Newark")
    # retrieval intentionally surfaces a CAR-WEST doc
    assert any(r.doc_id == "SOP-WEST-014" for r in ev)
    msgs = build_messages(
        "what is the hazmat pickup cutoff and weekend availability at Newark?",
        "CAR-NORTH",
        ev,
    )
    text = _assembled_text(msgs)
    assert "CAR-WEST" not in text, "another carrier's content leaked into the context"
    assert "17:00" not in text or "CAR-NORTH" in text, "foreign rule appears to have leaked"


def test_injected_note_does_not_become_an_instruction():
    ev = retrieve("can I approve a Saturday 15:00 hazmat pickup for SH-8841?")
    assert any(r.doc_id == "NOTE-8842" for r in ev)
    msgs = build_messages(
        "can I approve a Saturday 15:00 hazmat pickup for SH-8841?",
        "CAR-NORTH",
        ev,
    )
    # The injected text must not be presented to the model as an authoritative
    # system/policy instruction that overrides SOPs. The candidate may neutralize,
    # delimit, or drop it; we only require it is not in a role that steers policy.
    for m in msgs:
        if m.get("role") in ("system",):
            assert "override" not in m["content"].lower(), (
                "untrusted note text reached a policy/system role"
            )
            assert "ignore prior cutoff policy" not in m["content"].lower()


def test_context_stays_within_a_budget():
    # Force a busy hub: duplicate the candidate evidence many times.
    ev = retrieve("pickup Newark") * 12
    msgs = build_messages("what pickups are scheduled at Newark?", "CAR-NORTH", ev)
    text = _assembled_text(msgs)
    # The assembled context must not grow unbounded with the candidate set.
    assert len(text) < 12000, "assembled context did not respect any size bound"


def test_answer_path_runs_offline_and_exposes_assembly():
    client = LLMClient(offline=True)
    out = answer("what is the hazmat cutoff at Newark?", "CAR-NORTH", client)
    assert "messages" in out and out["messages"], "answer path did not assemble messages"
    assert out["raw"].get("offline") is True
