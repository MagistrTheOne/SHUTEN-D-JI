"""
Build Layer A — run every gold seed episode through the judge and persist the
REAL verification result. Episodes that fail any gate are reported and excluded
from the accepted corpus (no fabricated pass marks).

Outputs:
- data/code_forge/gold/gold.json          (accepted episodes, full record)
- data/code_forge/gold/gold_report.json   (per-episode gate report)
"""

from __future__ import annotations

import json
from pathlib import Path

from src.forge.gold_corpus import CANARY, gold_episodes
from src.forge.judge import judge_episode

OUT = Path("data/code_forge/gold")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    episodes = gold_episodes()

    accepted: list[dict] = []
    reports: list[dict] = []
    n_pass = 0

    for ep in episodes:
        verdict = judge_episode(ep, canary=CANARY)
        reports.append({
            "id": ep.id,
            "task_type": ep.task_type.value,
            "language": ep.language.value,
            "accepted": verdict.accepted,
            "failed_gates": verdict.failed_gates,
            "verification": ep.verification.model_dump(),
        })
        if verdict.accepted:
            n_pass += 1
            accepted.append(ep.model_dump(mode="json"))
        else:
            print(f"[REJECT] {ep.id}: {verdict.failed_gates}")

    (OUT / "gold.json").write_text(
        json.dumps(accepted, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "gold_report.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nLayer A: {n_pass}/{len(episodes)} episodes accepted (execution-verified).")
    print(f"Wrote {OUT / 'gold.json'}")


if __name__ == "__main__":
    main()
