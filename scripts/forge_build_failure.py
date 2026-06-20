"""
Build Layer C — run every failure-recovery seed through the judge and persist the
REAL verification (corrected code is executed against the hidden tests).

Outputs:
- data/code_forge/failure/failure.json
- data/code_forge/failure/failure_report.json
"""

from __future__ import annotations

import json
from pathlib import Path

from src.forge.failure_corpus import CANARY, failure_episodes
from src.forge.judge import judge_episode

OUT = Path("data/code_forge/failure")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    episodes = failure_episodes()
    accepted: list[dict] = []
    reports: list[dict] = []
    n_pass = 0

    for ep in episodes:
        verdict = judge_episode(ep, canary=CANARY)
        reports.append({
            "id": ep.id,
            "bug_class": ep.bug_class.value if ep.bug_class else None,
            "accepted": verdict.accepted,
            "failed_gates": verdict.failed_gates,
            "verification": ep.verification.model_dump(),
        })
        if verdict.accepted:
            n_pass += 1
            accepted.append(ep.model_dump(mode="json"))
        else:
            print(f"[REJECT] {ep.id} ({ep.bug_class}): {verdict.failed_gates}")

    (OUT / "failure.json").write_text(
        json.dumps(accepted, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "failure_report.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    classes = sorted({r["bug_class"] for r in reports if r["accepted"]})
    print(f"\nLayer C: {n_pass}/{len(episodes)} accepted | error classes: {classes}")


if __name__ == "__main__":
    main()
