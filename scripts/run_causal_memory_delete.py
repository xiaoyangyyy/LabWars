"""Standalone causal memory-delete analysis (explicit vs delete)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.exp_a_promise import run_causal_delete_vs_explicit


def main() -> int:
    parser = argparse.ArgumentParser(description="Run causal memory-delete pair analysis.")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "output" / "reports" / "exp_a_v4" / "causal_memory_delete.json"),
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = run_causal_delete_vs_explicit(n_seeds=args.seeds)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
