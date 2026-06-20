"""
Evaluate the adapter promotion gate from eval metric files.

Usage:
    python -m scripts.forge_ship_gate \
        --base data/code_forge/eval/base.json \
        --candidate data/code_forge/eval/shuten_code_lora.json \
        --strategy-regression 0.02 --general-regression 0.01

Self-check (demonstrates a promote and a hold decision):
    python -m scripts.forge_ship_gate --selfcheck
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.forge.ship_gate import evaluate_gate


def selfcheck() -> None:
    base = {"pass@1": 0.40, "compile_rate": 0.80, "hallucinated_api_rate": 0.15,
            "patch_efficiency_avg_changed_lines": 20.0}
    win = {"pass@1": 0.56, "compile_rate": 0.90, "hallucinated_api_rate": 0.06,
           "patch_efficiency_avg_changed_lines": 18.0}
    lose = {"pass@1": 0.44, "compile_rate": 0.78, "hallucinated_api_rate": 0.16,
            "patch_efficiency_avg_changed_lines": 40.0}

    d_win = evaluate_gate(base, win, strategy_regression=0.02, general_regression=0.01)
    d_lose = evaluate_gate(base, lose, strategy_regression=0.08, general_regression=0.01)

    print("WIN  ->", json.dumps(d_win.to_dict(), indent=2))
    print("HOLD ->", json.dumps(d_lose.to_dict(), indent=2))
    assert d_win.promote is True, "win scenario should promote"
    assert d_lose.promote is False, "lose scenario should hold"
    print("\nSELFCHECK OK — gate promotes a real win and holds a regression.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--base")
    p.add_argument("--candidate")
    p.add_argument("--strategy-regression", type=float, default=0.0)
    p.add_argument("--general-regression", type=float, default=0.0)
    args = p.parse_args()

    if args.selfcheck:
        selfcheck()
        return
    if not (args.base and args.candidate):
        raise SystemExit("Provide --selfcheck or both --base and --candidate")

    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    cand = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    decision = evaluate_gate(
        base, cand,
        strategy_regression=args.strategy_regression,
        general_regression=args.general_regression,
    )
    print(json.dumps(decision.to_dict(), indent=2))
    print("\nDECISION:", "PROMOTE -> merge" if decision.promote else "HOLD -> do not merge")
    print("v2:", decision.v2_trigger)
    raise SystemExit(0 if decision.promote else 2)


if __name__ == "__main__":
    main()
