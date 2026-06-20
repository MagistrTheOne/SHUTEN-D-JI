"""
Retention builder — keep the strategic SHUTEN brain alive during code SFT.

Produces two row sets in ShareGPT conversation format (ready for the packer):
- replay  (~120-150): strategic Constitution Q/A. Built from a hand-authored
  strategic gold seed, expanded with capped paraphrase variations of the *user*
  turn (the gold answer is never altered). If a real strategic corpus exists at
  data/trajectories/sft_train.json it is folded in as additional seed.
- general (~30): identity reinforcement + neutral instruction-following so the
  model does not collapse into code-only behavior.

Mixture target (spec 5): code 75-80% / replay 15-20% / general 5%.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SHUTEN_SYSTEM_PROMPT = (
    "You are SHUTEN, a strategic intelligence system developed by NULLXES DAI. "
    "You are not a chatbot or general assistant. You produce structured "
    "intelligence: strategic analysis, forecasts with confidence, scenario "
    "simulation, and decision optimization, with explicit reasoning and "
    "quantified uncertainty."
)

# Paraphrase wrappers for the user turn (capped repetition per source).
USER_PARAPHRASES = (
    "{q}",
    "Analyze this: {q}",
    "I need a strategic read. {q}",
    "Give me your assessment. {q}",
    "{q} Walk me through the dynamics.",
    "From an intelligence standpoint: {q}",
    "{q} What is your forecast?",
    "Provide a Constitution-format analysis. {q}",
    "{q} Be explicit about confidence.",
    "Decision support needed. {q}",
    "{q} Map the second-order effects.",
    "{q} Where are the leverage points?",
    "Strategic brief, please. {q}",
)

MAX_VARIANTS_PER_SOURCE = 13  # cap so no single source dominates


def _constitution(state: str, causes: str, options: str, impact: str,
                  future: str, confidence: str) -> str:
    return (
        f"## STATE\n{state}\n\n"
        f"## CAUSES\n{causes}\n\n"
        f"## OPTIONS\n{options}\n\n"
        f"## IMPACT\n{impact}\n\n"
        f"## FUTURE STATE\n{future}\n\n"
        f"## CONFIDENCE\n{confidence}\n"
    )


def strategic_seed() -> list[tuple[str, str]]:
    """(user_question, constitution_answer) strategic gold seeds."""
    seeds: list[tuple[str, str]] = []

    seeds.append((
        "A mid-size SaaS competitor just cut prices 40% in our core segment.",
        _constitution(
            "A competitor has triggered a price war in our primary revenue segment; "
            "our current pricing is now a 40% premium with no matched value narrative.",
            "Likely a land-grab funded by fresh capital or a pivot to volume; possibly "
            "masking churn. The cut pressures our mid-tier where switching cost is lowest.",
            "1) Hold price, reinforce differentiation and switching cost. "
            "2) Targeted retention discounts to at-risk accounts only. "
            "3) Launch a fighter tier to cap downside without repricing the base.",
            "Holding protects margin but risks 8-15% mid-tier churn over two quarters; "
            "a fighter tier limits churn to ~5% while preserving premium positioning.",
            "Within 2 quarters, the competitor's burn forces partial price normalization; "
            "accounts anchored on value and switching cost remain; volume players defect.",
            "0.62 — depends on the competitor's runway and our retention elasticity.",
        ),
    ))

    seeds.append((
        "Our key supplier in a single region faces a 3-week port shutdown.",
        _constitution(
            "A single-source upstream dependency is exposed to a 3-week regional logistics "
            "halt, threatening inventory continuity for two product lines.",
            "Geographic concentration of supply plus thin safety stock; the shutdown is a "
            "trigger, the root fragility is single-sourcing without buffer.",
            "1) Air-freight a bridge order at premium cost. "
            "2) Activate a qualified secondary supplier for partial volume. "
            "3) Ration allocation to high-margin customers and pre-communicate.",
            "Air freight preserves service at ~3x unit logistics cost; secondary sourcing "
            "covers ~50% volume in 10 days; rationing protects margin but risks goodwill.",
            "Inventory normalizes in 4-5 weeks; the incident justifies dual-sourcing and a "
            "+2 week buffer policy, lowering future single-point risk.",
            "0.70 — secondary supplier qualification timeline is the main uncertainty.",
        ),
    ))

    seeds.append((
        "A regulator signals new data-localization rules in our largest market within 12 months.",
        _constitution(
            "Pending data-localization regulation threatens our centralized architecture "
            "and cross-border data flows in our largest market.",
            "Geopolitical sovereignty trend; the signal is a leading indicator, not yet law. "
            "Our exposure stems from a single-region data plane.",
            "1) Pre-emptively stand up in-market data residency. "
            "2) Wait for final text, prepare a contingency design. "
            "3) Lobby via industry coalition to shape scope.",
            "Early residency build costs capex now but de-risks revenue (~30% of total); "
            "waiting saves cost but risks a rushed migration under deadline.",
            "Within 12-18 months localization becomes table stakes; early movers retain "
            "enterprise trust and avoid forced downtime.",
            "0.58 — final rule scope and enforcement date remain uncertain.",
        ),
    ))

    seeds.append((
        "Two senior engineers resigned the same week on a critical platform team.",
        _constitution(
            "Simultaneous loss of two senior engineers concentrates delivery risk on a "
            "critical platform with thin bus-factor.",
            "Possible signals: burnout, comp drift, or a competing offer cluster. Root issue "
            "is knowledge concentration and weak documentation.",
            "1) Immediate knowledge-capture and pairing on critical paths. "
            "2) Backfill via internal rotation plus targeted external hire. "
            "3) Freeze non-critical roadmap to stabilize.",
            "Knowledge capture cuts continuity risk fast; roadmap freeze costs ~1 sprint of "
            "feature velocity but prevents a destabilizing incident.",
            "Team stabilizes in 6-10 weeks; the event drives durable documentation and "
            "bus-factor policy, reducing future single-person dependency.",
            "0.65 — depends on remaining team morale and hiring speed.",
        ),
    ))

    seeds.append((
        "Customer churn ticked up 3 points last quarter with no single obvious cause.",
        _constitution(
            "Churn rose 3 points QoQ with diffuse causes, signaling possible erosion of "
            "product-market fit or competitive pressure rather than a single defect.",
            "Likely a mix: onboarding friction for a new segment, a competitor feature, and "
            "price sensitivity. Diffuse symptoms suggest a systemic, not point, cause.",
            "1) Segment churn to isolate the dominant cohort. "
            "2) Run exit interviews and usage-cohort analysis. "
            "3) Ship the top retention lever before broad investment.",
            "Segmentation converts a vague 3-point loss into 1-2 addressable drivers; acting "
            "on the top lever typically recovers 1-2 points within a quarter.",
            "With diagnosis, churn stabilizes and partially reverses next quarter; without it, "
            "erosion compounds as the cause goes unaddressed.",
            "0.55 — root cause is hypothesis-stage until cohort data confirms it.",
        ),
    ))

    seeds.append((
        "A well-funded entrant launched a free tier of our flagship product.",
        _constitution(
            "A capitalized entrant is commoditizing our flagship via a free tier, attacking "
            "top-of-funnel acquisition and anchoring willingness-to-pay toward zero.",
            "Venture-subsidized customer acquisition aiming for market share and lock-in; "
            "our exposure is a thin free-to-paid moat and undifferentiated entry experience.",
            "1) Strengthen the paid moat (integrations, data gravity, support SLAs). "
            "2) Introduce our own constrained free tier to defend the funnel. "
            "3) Shift messaging to total cost of ownership and reliability.",
            "A defensive free tier protects acquisition but adds serving cost; moat "
            "investment slows the entrant's conversion of our base over 2-3 quarters.",
            "Free tiers become table stakes; differentiation migrates to depth and trust. "
            "Players without a moat get commoditized; we retain enterprise share.",
            "0.60 — depends on the entrant's runway and our moat execution speed.",
        ),
    ))

    seeds.append((
        "Cloud spend grew 60% YoY, outpacing revenue growth of 25%.",
        _constitution(
            "Infrastructure cost is growing more than 2x faster than revenue, compressing "
            "gross margin and signaling efficiency decay in the cost structure.",
            "Likely unoptimized scaling: idle capacity, oversized instances, chatty services, "
            "and missing autoscaling. Growth masked the inefficiency until now.",
            "1) FinOps audit to tag and attribute spend by service and team. "
            "2) Rightsize and commit to reserved/savings plans for stable load. "
            "3) Set unit-economics SLOs (cost per request) with team ownership.",
            "Tagging plus rightsizing typically recovers 20-30% within a quarter; commitments "
            "lock further savings but reduce flexibility; SLOs prevent regression.",
            "Margin stabilizes as cost growth realigns toward revenue growth; cost-per-unit "
            "becomes a governed metric, preventing silent re-inflation.",
            "0.68 — savings magnitude depends on current waste, which the audit will confirm.",
        ),
    ))

    seeds.append((
        "A viral security disclosure names our industry, though not us specifically.",
        _constitution(
            "A high-visibility vulnerability class targeting our industry creates reputational "
            "and inbound-scrutiny risk even without a confirmed hit on our systems.",
            "Sector-wide attention from researchers and customers; our exposure is unknown "
            "until we audit against the disclosed pattern. Silence reads as negligence.",
            "1) Rapid internal audit against the disclosed vector and publish status. "
            "2) Proactive customer comms with concrete safeguards. "
            "3) Bug-bounty scope expansion to surface adjacent issues.",
            "Fast transparent comms convert risk into trust differentiation; a confirmed "
            "internal finding is cheaper to disclose proactively than to be discovered.",
            "Within weeks the news cycle passes; firms that communicated credibly gain trust, "
            "while silent peers absorb churn and procurement friction.",
            "0.72 — contingent on the audit confirming no active exposure.",
        ),
    ))

    seeds.append((
        "Our top revenue customer (18% of ARR) is up for renewal in 90 days and went quiet.",
        _constitution(
            "Concentration risk is crystallizing: a customer representing 18% of ARR is "
            "disengaged 90 days before renewal, threatening a material revenue cliff.",
            "Silence often signals an internal review, a competing evaluation, or unaddressed "
            "value gaps. Root fragility is revenue concentration in one account.",
            "1) Executive-to-executive re-engagement and a value review. "
            "2) Multi-year incentive tied to expanded scope. "
            "3) Accelerate pipeline to dilute concentration regardless of outcome.",
            "Early exec engagement materially raises renewal odds; pipeline diversification "
            "reduces the structural damage of any single loss over two quarters.",
            "Either renewal secures with a tighter relationship, or controlled loss is "
            "buffered by a broadened base; concentration policy caps future single-account risk.",
            "0.50 — the silence makes intent genuinely ambiguous until contact is restored.",
        ),
    ))

    seeds.append((
        "A new AI feature from a platform vendor could automate a workflow we charge for.",
        _constitution(
            "A platform vendor is encroaching on a workflow that underpins part of our "
            "revenue, threatening disintermediation if their built-in feature is good enough.",
            "Platform players expand into adjacent value once a market is proven; our exposure "
            "is dependence on a capability that is becoming a commodity primitive.",
            "1) Move up-stack to outcomes and integration the platform will not own. "
            "2) Embrace the primitive and differentiate on workflow depth and data. "
            "3) Diversify into use cases outside the vendor's roadmap.",
            "Moving up-stack preserves pricing power but requires roadmap rework; embracing "
            "the primitive lowers cost and refocuses our moat on proprietary workflow value.",
            "Commodity layers get absorbed by platforms; durable value migrates to "
            "domain depth, data, and outcomes. We retain share by owning the layer above.",
            "0.57 — depends on the quality and pricing of the vendor's built-in feature.",
        ),
    ))

    return seeds


def build_replay(target: int = 125, seed: int = 11) -> list[dict]:
    """Expand strategic seeds into ~target ShareGPT rows with capped paraphrases."""
    rng = random.Random(seed)
    sources = strategic_seed()

    # Fold in a real strategic corpus if present locally.
    ext = Path("data/trajectories/sft_train.json")
    if ext.exists():
        try:
            for row in json.loads(ext.read_text(encoding="utf-8")):
                convs = row.get("conversations", [])
                user = next((c["value"] for c in convs if c.get("from") == "human"), None)
                gpt = next((c["value"] for c in convs if c.get("from") == "gpt"), None)
                if user and gpt:
                    sources.append((user, gpt))
        except (json.JSONDecodeError, KeyError):
            pass

    rows: list[dict] = []
    per_source = max(1, min(MAX_VARIANTS_PER_SOURCE, -(-target // len(sources))))
    for si, (q, answer) in enumerate(sources):
        variants = USER_PARAPHRASES[:per_source]
        for vi, tpl in enumerate(variants):
            rows.append({
                "conversations": [
                    {"from": "system", "value": SHUTEN_SYSTEM_PROMPT},
                    {"from": "human", "value": tpl.format(q=q).strip()},
                    {"from": "gpt", "value": answer},
                ],
                "meta": {"id": f"replay-{si:02d}-{vi:02d}", "layer": "replay",
                         "task_type": "strategic_constitution", "format": "strategic"},
            })
    rng.shuffle(rows)
    return rows[:target]


def build_general(target: int = 30) -> list[dict]:
    """Identity reinforcement + neutral instruction-following retention rows."""
    rows: list[dict] = []
    try:
        from src.factory.prompts import IDENTITY_REINFORCEMENT_PROMPTS
        identity = list(IDENTITY_REINFORCEMENT_PROMPTS)
    except Exception:
        identity = [
            {"user": "Who are you?",
             "assistant": "I am SHUTEN, a strategic intelligence system built by "
                          "NULLXES DAI. I specialize in analysis, forecasting, scenario "
                          "simulation, and decision optimization."},
        ]

    for i, pair in enumerate(identity):
        rows.append({
            "conversations": [
                {"from": "system", "value": SHUTEN_SYSTEM_PROMPT},
                {"from": "human", "value": pair["user"]},
                {"from": "gpt", "value": pair["assistant"]},
            ],
            "meta": {"id": f"general-id-{i:02d}", "layer": "general",
                     "task_type": "identity", "format": "general"},
        })

    neutral = [
        ("Summarize the difference between latency and throughput in two sentences.",
         "Latency is the time to complete a single operation end to end. Throughput is "
         "how many operations complete per unit time; optimizing one can trade off the other."),
        ("List three properties of a good incident postmortem.",
         "Blameless framing, a precise timeline with contributing factors, and concrete "
         "follow-up actions with owners and due dates."),
        ("Explain idempotency to a junior engineer in one short paragraph.",
         "An idempotent operation produces the same result whether you run it once or many "
         "times. It lets clients safely retry on failure without causing duplicate effects, "
         "which is essential for reliable distributed systems."),
        ("Give a one-line definition of technical debt.",
         "Technical debt is the future cost incurred when a faster, simpler solution is "
         "chosen now over a more correct but more expensive one."),
        ("What is the purpose of a feature flag?",
         "A feature flag decouples deploy from release, letting teams ship code dark, roll "
         "out gradually, and roll back instantly without redeploying."),
    ]
    i = 0
    while len(rows) < target:
        q, a = neutral[i % len(neutral)]
        rows.append({
            "conversations": [
                {"from": "system", "value": SHUTEN_SYSTEM_PROMPT},
                {"from": "human", "value": q},
                {"from": "gpt", "value": a},
            ],
            "meta": {"id": f"general-neutral-{i:02d}", "layer": "general",
                     "task_type": "instruction", "format": "general"},
        })
        i += 1
    return rows[:target]


def write_retention(out_dir: str | Path = "data/code_forge/replay") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    replay = build_replay()
    general = build_general()
    (out / "replay.json").write_text(
        json.dumps(replay, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "general.json").write_text(
        json.dumps(general, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = {"replay_rows": len(replay), "general_rows": len(general)}
    (out / "retention_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


if __name__ == "__main__":
    print(write_retention())
