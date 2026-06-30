"""CLI entry point and self-check."""
from __future__ import annotations

import argparse
import json

from agent.context_builder import answer
from agent.llm_client import LLMClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispatch copilot")
    parser.add_argument("--carrier", required=True, help="asking carrier id, e.g. CAR-NORTH")
    parser.add_argument("--question", required=True)
    parser.add_argument("--live", action="store_true", help="call the real model")
    args = parser.parse_args()

    client = LLMClient(offline=not args.live)
    result = answer(args.question, args.carrier, client)
    print(json.dumps(result, indent=2)[:4000])


if __name__ == "__main__":
    main()
