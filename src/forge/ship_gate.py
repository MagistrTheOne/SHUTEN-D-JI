"""
Adapter promotion gate (spec section 8).

Decide whether SHUTEN-CODE-LORA-v1 may be merged into SHUTEN-DOJI-CODE-27B.
Inputs are metric dicts produced by the eval harness (scripts/eval_code.py) and
the strategic/general regression checks. Execution metrics only — no vibes.

Promote ONLY if ALL gates pass:
- pass@1 improvement >= +0.10 absolute on NULLXES-CODE-EVAL (target +10-20%)
- strategy regression <= 5%
- general regression <= 5%
- hallucinated_api_rate not worse than base
- compile_rate >= base
- patch size controlled (avg changed lines <= 1.5x base)

The v2 trigger picks the next RL method from the failure signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PASS1_MIN_DELTA = 0.10
MAX_REGRESSION = 0.05
PATCH_BLOAT_FACTOR = 1.5


@dataclass
class GateCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class GateDecision:
    promote: bool
    checks: list[GateCheck] = field(default_factory=list)
    v2_trigger: str = ""

    def to_dict(self) -> dict:
        return {
            "promote": self.promote,
            "v2_trigger": self.v2_trigger,
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail}
                       for c in self.checks],
            "failed": [c.name for c in self.checks if not c.passed],
        }


def evaluate_gate(
    base: dict,
    candidate: dict,
    *,
    strategy_regression: float = 0.0,
    general_regression: float = 0.0,
) -> GateDecision:
    checks: list[GateCheck] = []

    d_pass1 = candidate.get("pass@1", 0.0) - base.get("pass@1", 0.0)
    checks.append(GateCheck(
        "pass@1_improvement", d_pass1 >= PASS1_MIN_DELTA,
        f"delta={d_pass1:+.3f} (need >= +{PASS1_MIN_DELTA})"))

    checks.append(GateCheck(
        "strategy_regression", strategy_regression <= MAX_REGRESSION,
        f"{strategy_regression:.3f} (need <= {MAX_REGRESSION})"))

    checks.append(GateCheck(
        "general_regression", general_regression <= MAX_REGRESSION,
        f"{general_regression:.3f} (need <= {MAX_REGRESSION})"))

    hb = base.get("hallucinated_api_rate", 0.0)
    hc = candidate.get("hallucinated_api_rate", 0.0)
    checks.append(GateCheck(
        "hallucinated_api_not_worse", hc <= hb,
        f"base={hb:.3f} cand={hc:.3f}"))

    cb = base.get("compile_rate", 0.0)
    cc = candidate.get("compile_rate", 0.0)
    checks.append(GateCheck(
        "compile_rate_up", cc >= cb,
        f"base={cb:.3f} cand={cc:.3f}"))

    pb = base.get("patch_efficiency_avg_changed_lines", 0.0)
    pc = candidate.get("patch_efficiency_avg_changed_lines", 0.0)
    patch_ok = pb == 0.0 or pc <= pb * PATCH_BLOAT_FACTOR
    checks.append(GateCheck(
        "patch_size_controlled", patch_ok,
        f"base={pb:.1f} cand={pc:.1f} (<= {PATCH_BLOAT_FACTOR}x)"))

    promote = all(c.passed for c in checks)
    return GateDecision(promote=promote, checks=checks,
                        v2_trigger=_v2_trigger(promote, checks))


def _v2_trigger(promote: bool, checks: list[GateCheck]) -> str:
    """Recommend the next iteration method from the gate outcome."""
    failed = {c.name for c in checks if not c.passed}
    if promote:
        return ("DPO/ORPO: build chosen/rejected pairs from the executed candidate "
                "archive (accepted vs gate-rejected) to sharpen minimality and "
                "format adherence without risking the strategic brain.")
    if "pass@1_improvement" in failed:
        return ("GRPO with verified reward (tests pass = reward) on a small task "
                "set; the SFT signal was insufficient to lift capability.")
    if {"strategy_regression", "general_regression"} & failed:
        return ("Re-balance the mixture toward more replay/general retention and "
                "re-run SFT before any RL; capability traded off the strategic brain.")
    return ("Targeted SFT: add gold/failure episodes for the regressed metric "
            "(e.g. hallucinated API or patch bloat) and re-run before RL.")
