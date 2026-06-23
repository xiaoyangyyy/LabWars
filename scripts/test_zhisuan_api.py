"""One-off connectivity test for Zhisuan OpenAI-compatible API."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openai import OpenAI


def main() -> int:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("OPENAI_API_KEY not set")
        return 1

    client = OpenAI(api_key=key, base_url="https://ai.azya.top/v1")

    print("=== List models ===")
    try:
        models = client.models.list()
        ids = [m.id for m in models.data[:20]]
        print("Available:", ids)
    except Exception as exc:
        print("list_models failed:", exc)

    for model in ("qwen3.5", "gpt-4o", "gpt-4o-mini"):
        print(f"\n=== Chat test: {model} ===")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Output JSON only."},
                    {"role": "user", "content": '{"ping": "pong"}'},
                ],
                max_tokens=64,
            )
            print("OK:", resp.choices[0].message.content)
            break
        except Exception as exc:
            print("FAIL:", exc)

    print("\n=== LabWars adapter test ===")
    from src.engine.llm_adapter import get_adapter

    llm = get_adapter()
    out = llm.complete_json(
        "Output JSON only.",
        '{"task": "pick action", "allowed_actions": ["document_contribution", "comply"], "output_schema": {"action": "string"}}',
    )
    print("adapter:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
