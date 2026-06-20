"""
CODE FORGE Packer — assemble accepted episodes + replay + general into a
LlamaFactory ShareGPT-style training pack with the spec format mix and dedup.

Format mix target (spec 4.2): 35% direct answer, 30% patch/diff, 20% repair,
10% code review, 5% architecture.

Mixture target (spec 5): ~500 code + ~125 replay + ~30 general ~= 650 rows.

Outputs:
- data/code_forge/pack/code_forge_train.json   (ShareGPT conversations)
- data/code_forge/pack/code_forge_dev.json
- data/code_forge/pack/dataset_info.json
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from src.forge.gates import content_hash
from src.forge.schema import SYSTEM_PROMPT, Episode, TaskType

# Target proportions across the *code* portion (spec 4.2).
FORMAT_MIX = {
    "direct": 0.35,     # function_impl
    "diff": 0.30,       # repo_bugfix + multi_file
    "repair": 0.20,     # repair_trajectory
    "review": 0.10,     # code_review
    "architecture": 0.05,
}

FORMAT_OF_TASK = {
    TaskType.FUNCTION_IMPL: "direct",
    TaskType.REPO_BUGFIX: "diff",
    TaskType.MULTI_FILE: "diff",
    TaskType.TEST_GEN: "direct",
    TaskType.REFACTOR: "diff",
    TaskType.SECURITY: "diff",
    TaskType.PERFORMANCE: "diff",
    TaskType.ARCHITECTURE: "architecture",
    TaskType.CODE_REVIEW: "review",
    TaskType.REPAIR_TRAJECTORY: "repair",
}


def render_user(ep: Episode) -> str:
    parts = [ep.prompt.task.strip()]
    if ep.prompt.repo_context.strip():
        parts.append("\n## REPO CONTEXT\n" + ep.prompt.repo_context.strip())
    if ep.prompt.constraints:
        parts.append("\n## CONSTRAINTS\n" + "\n".join(f"- {c}" for c in ep.prompt.constraints))
    return "\n".join(parts)


def render_assistant(ep: Episode) -> str:
    """The solution content already follows the Code Constitution block layout."""
    return ep.solution.content.strip()


def episode_to_conversation(ep: Episode) -> dict:
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": render_user(ep)},
            {"from": "gpt", "value": render_assistant(ep)},
        ],
        "meta": {
            "id": ep.id,
            "layer": ep.layer.value,
            "task_type": ep.task_type.value,
            "language": ep.language.value,
            "format": FORMAT_OF_TASK.get(ep.task_type, "direct"),
            "verified": ep.verification.all_pass,
        },
    }


def replay_to_conversation(row: dict) -> dict:
    """Strategic replay / general retention rows are already conversation dicts."""
    return row


def dedup(episodes: list[Episode]) -> list[Episode]:
    seen: set[str] = set()
    out: list[Episode] = []
    for ep in episodes:
        h = content_hash(ep)
        if h in seen:
            continue
        seen.add(h)
        out.append(ep)
    return out


def assemble_pack(
    code_episodes: list[Episode],
    public_code_rows: list[dict],
    replay_rows: list[dict],
    general_rows: list[dict],
    *,
    dev_fraction: float = 0.06,
    seed: int = 42,
) -> dict:
    """Build train/dev splits with the spec mixture. Returns stats + records."""
    rng = random.Random(seed)
    code_episodes = dedup(code_episodes)
    rng.shuffle(code_episodes)

    verified_code_convs = [episode_to_conversation(e) for e in code_episodes]
    public_code_convs = [replay_to_conversation(r) for r in public_code_rows]
    code_convs = verified_code_convs + public_code_convs
    replay_convs = [replay_to_conversation(r) for r in replay_rows]
    general_convs = [replay_to_conversation(r) for r in general_rows]

    all_rows = code_convs + replay_convs + general_convs
    rng.shuffle(all_rows)

    n_dev = max(1, int(len(all_rows) * dev_fraction))
    dev = all_rows[:n_dev]
    train = all_rows[n_dev:]

    fmt_counts = Counter(c["meta"]["format"] for c in code_convs)
    total_code = max(1, len(code_convs))
    stats = {
        "total_rows": len(all_rows),
        "train_rows": len(train),
        "dev_rows": len(dev),
        "code_rows": len(code_convs),
        "verified_code_rows": len(verified_code_convs),
        "public_code_rows": len(public_code_convs),
        "replay_rows": len(replay_convs),
        "general_rows": len(general_convs),
        "code_fraction": round(len(code_convs) / len(all_rows), 3),
        "replay_fraction": round(len(replay_convs) / len(all_rows), 3),
        "general_fraction": round(len(general_convs) / len(all_rows), 3),
        "format_mix_actual": {k: round(v / total_code, 3) for k, v in fmt_counts.items()},
        "format_mix_target": FORMAT_MIX,
    }
    return {"train": train, "dev": dev, "stats": stats}


def write_pack(pack: dict, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "code_forge_train.json").write_text(
        json.dumps(pack["train"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "code_forge_dev.json").write_text(
        json.dumps(pack["dev"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    dataset_info = {
        "code_forge_train": {
            "file_name": "code_forge_train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "system_tag": "system",
            },
        },
        "code_forge_dev": {
            "file_name": "code_forge_dev.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "system_tag": "system",
            },
        },
    }
    (out / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "pack_stats.json").write_text(
        json.dumps(pack["stats"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return pack["stats"]
