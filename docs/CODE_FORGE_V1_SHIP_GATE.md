# CODE FORGE v1 — Ship Gate

Promote `SHUTEN-CODE-LORA-v1` into a merged `SHUTEN-DOJI-CODE-27B` **only** when
every gate below passes on NULLXES-CODE-EVAL under identical sampling/sandbox.
Execution metrics decide; nothing is promoted on impression.

Run the check:

```bash
python -m scripts.forge_ship_gate \
  --base data/code_forge/eval/base.json \
  --candidate data/code_forge/eval/shuten_code_lora.json \
  --strategy-regression <from strategic re-eval> \
  --general-regression <from general re-eval>
```

Exit code 0 = promote, 2 = hold.

## Gates (all must pass)

| Gate | Rule |
|------|------|
| pass@1 improvement | candidate pass@1 − base pass@1 >= **+0.10** (target +10-20%) |
| strategy regression | <= **5%** drop in strategic Constitution compliance |
| general regression | <= **5%** drop on general retention |
| hallucinated API | candidate rate **not worse** than base |
| compile rate | candidate **>=** base |
| patch size | avg changed lines <= **1.5x** base (no bloat) |

## On a win (promote)

1. Merge LoRA into base **off-GPU**, upload as `SHUTEN-DOJI-CODE-27B`.
2. Keep `SHUTEN-CODE-LORA-v1` as a separate artifact.
3. **v2 trigger — DPO/ORPO**: build chosen/rejected pairs from the executed
   candidate archive (`data/code_forge/rejected/` vs accepted) to sharpen
   minimality and format adherence without risking the strategic brain.

## On a hold (do not merge)

The checker recommends the next move from the failure signature:

- **pass@1 too low** -> GRPO with a verified reward (tests pass = reward) on a
  small task set; SFT signal was insufficient to lift capability.
- **strategy/general regression** -> re-balance the mixture toward more
  replay/general retention and re-run SFT before any RL.
- **hallucinated API / patch bloat** -> targeted SFT: add gold + failure
  episodes for the regressed metric, then re-run before RL.

No DPO/GRPO and no merge happen before a clean SFT eval win — this is the v1
discipline (spec section 10, out of scope).
