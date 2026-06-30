#!/usr/bin/env bash
set -u
cd /root/task || exit 2

python -m pip install -q -e . 2>/dev/null

python - <<'PY'
import json, pathlib
from agent import context_builder, retrieval, prompt_templates
from agent.llm_client import LLMClient

# Static artifacts must load without solving the task.
docs = retrieval.load_corpus()
assert len(docs) > 0, "corpus failed to load"
tr = list((pathlib.Path('traces')).glob('*.jsonl'))
assert tr, "no traces present"
assert prompt_templates.SYSTEM_POLICY.strip(), "policy template empty"
LLMClient(offline=True)  # construct offline client without calling the model
print("artifacts-ok")
PY
self_rc=$?
if [ "$self_rc" -ne 0 ]; then
  exit "$self_rc"
fi

python -m pytest -q
rc=$?
if [ "$rc" -le 1 ]; then
  echo "ready"
  exit 0
fi
exit "$rc"
