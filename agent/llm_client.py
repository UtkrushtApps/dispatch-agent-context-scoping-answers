"""Real-provider model client with an offline readiness mode.

The live path calls OpenAI on the candidate's own key. Offline mode returns a
deterministic envelope so readiness and invariant tests run without a key. Offline
mode does NOT make any context/retrieval/scoping decision on the candidate's
behalf; it only echoes back the request it was given so tests can inspect what
would have been sent to the model.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelResponse:
    text: str
    raw: Dict[str, Any]


class LLMClient:
    def __init__(self, offline: Optional[bool] = None, model: Optional[str] = None):
        if offline is None:
            offline = os.getenv("DISPATCH_OFFLINE", "1") == "1"
        self.offline = offline
        self.model = model or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self._client = None
        if not self.offline:
            from openai import OpenAI

            self._client = OpenAI()

    def complete(self, messages: List[Dict[str, str]], **kwargs: Any) -> ModelResponse:
        """Send an assembled chat request to the model.

        `messages` is a standard chat-style list of {role, content} dicts that the
        caller has assembled. In offline mode we do not invent an answer; we echo
        the assembled request so its structure and size can be inspected.
        """
        if self.offline:
            assembled = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
            return ModelResponse(
                text="<offline: model not called>",
                raw={"offline": True, "assembled": assembled, "messages": messages},
            )
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, **kwargs
        )
        return ModelResponse(text=resp.choices[0].message.content or "", raw=resp.model_dump())
