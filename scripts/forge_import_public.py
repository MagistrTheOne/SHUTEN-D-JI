"""
Import public code-instruction datasets into the CODE FORGE pack format.

This is the fast bootstrap path for SHUTEN-CODE:
- Alpaca-style datasets: instruction/input/output -> ShareGPT.
- Messages-style datasets: messages[{role, content}] -> ShareGPT.

The importer is intentionally conservative: it filters empty/huge/poisoned rows,
deduplicates by user+assistant content, and wraps Alpaca outputs in the
SHUTEN-CODE section format so public code data does not overwrite the model's
answer style.

Examples:
    python -m scripts.forge_import_public --selfcheck

    python -m scripts.forge_import_public \
      --alpaca iamtarun/code_instructions_120k_alpaca:8000 \
      --alpaca iamtarun/python_code_instructions_18k_alpaca:5000 \
      --messages cfahlgren1/react-code-instructions:2000 \
      --out data/code_forge/public/public_code.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.forge.schema import FORBIDDEN_PATTERNS, SYSTEM_PROMPT

OUT = Path("data/code_forge/public/public_code.json")

MAX_USER_CHARS = 6000
MAX_ASSISTANT_CHARS = 12000
MIN_ASSISTANT_CHARS = 20
DEFAULT_SOURCE_LIMIT = 1000

_FENCE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)


def _parse_source(value: str) -> tuple[str, int]:
    """Parse repo[:limit]. HF repo IDs contain one slash but no colon."""
    if ":" not in value:
        return value, DEFAULT_SOURCE_LIMIT
    repo, limit_s = value.rsplit(":", 1)
    return repo, int(limit_s)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\n{3,}", "\n\n", str(value).replace("\r\n", "\n")).strip()


def _has_forbidden(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in FORBIDDEN_PATTERNS)


def _guess_language(text: str) -> str:
    low = text.lower()
    if any(x in low for x in ("react", "jsx", "tsx", "typescript", "tailwind")):
        return "tsx"
    if any(x in low for x in ("python", "def ", "import pandas", "pytest")):
        return "python"
    if any(x in low for x in ("select ", "postgres", "mysql", "sql")):
        return "sql"
    if any(x in low for x in ("#include", "std::", "c++")):
        return "cpp"
    if any(x in low for x in ("javascript", "function ", "const ", "let ")):
        return "javascript"
    return "text"


def _extract_codeish(output: str) -> str:
    """Prefer fenced code contents; otherwise keep the source output as-is."""
    match = _FENCE.search(output)
    return match.group(1).strip() if match else output.strip()


def _wrap_public_answer(instruction: str, output: str) -> str:
    language = _guess_language(instruction + "\n" + output)
    codeish = _extract_codeish(output)
    if language != "text":
        implementation = f"```{language}\n{codeish}\n```"
    else:
        implementation = codeish
    return (
        "## ASSUMPTIONS\n"
        "- Public code-instruction source; verify behavior before production use.\n\n"
        "## IMPLEMENTATION\n"
        f"{implementation}\n\n"
        "## TESTS\n"
        "Not provided by the source dataset.\n\n"
        "## COMPLEXITY\n"
        "Depends on the implementation and input size.\n"
    )


def _conversation(user: str, assistant: str, meta: dict[str, Any]) -> dict[str, Any] | None:
    user = _clean_text(user)
    assistant = _clean_text(assistant)
    if not user or len(user) > MAX_USER_CHARS:
        return None
    if len(assistant) < MIN_ASSISTANT_CHARS or len(assistant) > MAX_ASSISTANT_CHARS:
        return None
    if _has_forbidden(user) or _has_forbidden(assistant):
        return None
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": user},
            {"from": "gpt", "value": assistant},
        ],
        "meta": meta,
    }


def convert_alpaca_row(row: dict[str, Any], source: str, idx: int) -> dict[str, Any] | None:
    instruction = _clean_text(row.get("instruction"))
    inp = _clean_text(row.get("input"))
    output = _clean_text(row.get("output"))
    if not instruction or not output:
        return None
    user = instruction if not inp else f"{instruction}\n\n## INPUT\n{inp}"
    assistant = _wrap_public_answer(instruction, output)
    return _conversation(
        user,
        assistant,
        {
            "id": f"public-{source.replace('/', '__')}-{idx:06d}",
            "layer": "public",
            "source": source,
            "format": "direct",
            "verified": False,
        },
    )


def convert_messages_row(row: dict[str, Any], source: str, idx: int) -> dict[str, Any] | None:
    raw = row.get("messages")
    if not isinstance(raw, list):
        return None
    user_parts: list[str] = []
    assistant_parts: list[str] = []
    for msg in raw:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or msg.get("from")
        content = _clean_text(msg.get("content") or msg.get("value"))
        if role in ("user", "human"):
            user_parts.append(content)
        elif role in ("assistant", "gpt"):
            assistant_parts.append(content)
    if not user_parts or not assistant_parts:
        return None
    assistant = _wrap_public_answer("\n\n".join(user_parts), "\n\n".join(assistant_parts))
    return _conversation(
        "\n\n".join(user_parts),
        assistant,
        {
            "id": f"public-{source.replace('/', '__')}-{idx:06d}",
            "layer": "public",
            "source": source,
            "format": "direct",
            "verified": False,
        },
    )


def _iter_hf(repo: str, limit: int, split: str) -> Iterable[dict[str, Any]]:
    try:
        load_dataset = importlib.import_module("datasets").load_dataset
    except ImportError as exc:
        raise SystemExit("Install datasets first: pip install datasets") from exc

    ds = load_dataset(repo, split=split, streaming=True)
    for idx, row in enumerate(ds):
        if idx >= limit:
            break
        yield row


def _hash_row(row: dict[str, Any]) -> str:
    conv = row["conversations"]
    user = conv[1]["value"]
    assistant = conv[2]["value"]
    return hashlib.sha256(f"{user}\n---\n{assistant}".encode()).hexdigest()


def import_public(
    alpaca_sources: list[str],
    messages_sources: list[str],
    *,
    out: Path = OUT,
    split: str = "train",
) -> dict[str, int]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempted = 0

    def add(row: dict[str, Any] | None) -> None:
        if row is None:
            return
        h = _hash_row(row)
        if h in seen:
            return
        seen.add(h)
        rows.append(row)

    for source in alpaca_sources:
        repo, limit = _parse_source(source)
        for idx, row in enumerate(_iter_hf(repo, limit, split), start=1):
            attempted += 1
            add(convert_alpaca_row(row, repo, idx))

    for source in messages_sources:
        repo, limit = _parse_source(source)
        for idx, row in enumerate(_iter_hf(repo, limit, split), start=1):
            attempted += 1
            add(convert_messages_row(row, repo, idx))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = {"attempted": attempted, "kept": len(rows), "dropped": attempted - len(rows)}
    (out.parent / "public_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats


def selfcheck() -> None:
    alpaca = {
        "instruction": "Create a function to add two numbers.",
        "input": "2, 3",
        "output": "def add(a, b):\n    return a + b",
    }
    messages = {
        "messages": [
            {"role": "system", "content": "old system"},
            {"role": "user", "content": "Create a React counter."},
            {"role": "assistant", "content": "export default function Counter(){return <button/>}"},
        ]
    }
    a = convert_alpaca_row(alpaca, "self/alpaca", 1)
    m = convert_messages_row(messages, "self/messages", 1)
    assert a is not None and m is not None
    assert a["conversations"][0]["value"] == SYSTEM_PROMPT
    assert "## IMPLEMENTATION" in a["conversations"][2]["value"]
    assert "React counter" in m["conversations"][1]["value"]
    print("SELFCHECK OK — public dataset rows convert to SHUTEN ShareGPT format.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--alpaca", action="append", default=[], help="HF repo[:limit].")
    p.add_argument("--messages", action="append", default=[], help="HF repo[:limit].")
    p.add_argument("--split", default="train")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()

    if args.selfcheck:
        selfcheck()
        return
    if not args.alpaca and not args.messages:
        raise SystemExit("Provide at least one --alpaca or --messages source.")

    stats = import_public(
        args.alpaca,
        args.messages,
        out=Path(args.out),
        split=args.split,
    )
    print(json.dumps(stats, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
