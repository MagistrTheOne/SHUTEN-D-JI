"""
Pack the CODE FORGE training set.

Loads every produced layer, reconstructs code Episodes, and assembles the final
ShareGPT pack with dedup + the spec format mix + the retention mixture, writing
train/dev splits and dataset_info.json.

Layers consumed (each optional; missing layers are skipped with a warning):
- data/code_forge/gold/gold.json          (Layer A, executable-verified)
- data/code_forge/failure/failure.json    (Layer C, executable-verified)
- data/code_forge/synthetic/synthetic.json(Layer B, produced on the pod)
- data/code_forge/public/public_code.json (public code-instruction bootstrap)
- data/code_forge/replay/replay.json      (strategic replay)
- data/code_forge/replay/general.json     (general retention)

Output: data/code_forge/pack/{code_forge_train.json, code_forge_dev.json,
dataset_info.json, pack_stats.json}
"""

from __future__ import annotations

import json
from pathlib import Path

from src.forge.packer import assemble_pack, write_pack
from src.forge.schema import Episode

ROOT = Path("data/code_forge")


def _load_episodes(path: Path) -> list[Episode]:
    if not path.exists():
        print(f"[skip] {path} not found")
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Episode(**d) for d in data]


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[skip] {path} not found")
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    code: list[Episode] = []
    code += _load_episodes(ROOT / "gold" / "gold.json")
    code += _load_episodes(ROOT / "failure" / "failure.json")
    code += _load_episodes(ROOT / "synthetic" / "synthetic.json")

    public_code = _load_rows(ROOT / "public" / "public_code.json")
    replay = _load_rows(ROOT / "replay" / "replay.json")
    general = _load_rows(ROOT / "replay" / "general.json")

    if not code and not public_code:
        raise SystemExit(
            "No code rows found. Run forge_import_public or build verified forge layers first."
        )

    pack = assemble_pack(code, public_code, replay, general)
    stats = write_pack(pack, ROOT / "pack")

    print(json.dumps(stats, indent=2))
    print(f"\nWrote pack to {ROOT / 'pack'}")
    code_frac = stats["code_fraction"]
    if not (0.55 <= code_frac <= 0.85):
        print(f"[note] code fraction {code_frac} outside 75/20/5 target — "
              f"generate Layer B on the pod to reach ~500 code episodes.")


if __name__ == "__main__":
    main()
