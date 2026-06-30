"""Retrieval over local fixtures. This layer is complete.

It returns candidate evidence rows for a question. It deliberately does NOT make
permission or trust decisions: it returns everything that lexically matches,
including material from other carriers and untrusted shipment notes. Downstream
context assembly is responsible for what is actually safe to send.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@dataclass
class EvidenceRow:
    doc_id: str
    source_type: str  # "sop" or "shipment_note"
    carrier_id: str
    hub: str
    text: str
    updated_at: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_type": self.source_type,
            "carrier_id": self.carrier_id,
            "hub": self.hub,
            "text": self.text,
            "updated_at": self.updated_at,
            "score": self.score,
        }


def load_corpus() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(FIXTURES / "corpus.jsonl", "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _overlap(query: str, text: str) -> float:
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    t = set(re.findall(r"[a-z0-9]+", text.lower()))
    if not q:
        return 0.0
    return len(q & t) / len(q)


def retrieve(query: str, top_k: int = 8) -> List[EvidenceRow]:
    """Return up to top_k lexically-matching rows, unfiltered by carrier or trust."""
    scored: List[EvidenceRow] = []
    for row in load_corpus():
        score = _overlap(query, row["text"])
        if score <= 0:
            continue
        scored.append(
            EvidenceRow(
                doc_id=row["doc_id"],
                source_type=row["source_type"],
                carrier_id=row["carrier_id"],
                hub=row["hub"],
                text=row["text"],
                updated_at=row["updated_at"],
                score=round(score, 4),
            )
        )
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:top_k]
