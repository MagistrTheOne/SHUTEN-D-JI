# NULLXES CODE FORGE v1 — Specification

Status: foundations build (no pod, no training launched).
Base model: merged `MagistrTheOne/SHUTEN-DOJI` (27B, BF16).
Outputs (separate artifacts): `SHUTEN-CODE-LORA-v1` (adapter) -> `SHUTEN-DOJI-CODE-27B` (merged, only after eval win).

Core principle: **the judge is execution, never the LLM.** A candidate enters the
training set only after it compiles, passes hidden tests, lints, typechecks, and
clears the security + contamination gates.

---

## 1. Code Constitution (conditional schema)

The strategic Constitution (`STATE -> CAUSES -> OPTIONS -> IMPACT -> FUTURE STATE ->
CONFIDENCE`) is **not** forced onto every coding answer. Block sets are selected by
task type so SHUTEN keeps character without turning a `TypeError` fix into a board
meeting.

### 1.1 Block sets per task type

- `function_impl` — `ASSUMPTIONS / IMPLEMENTATION / TESTS / COMPLEXITY`
- `repo_bugfix` — `DIAGNOSIS / ROOT CAUSE / PATCH / TESTS / VERIFICATION`
- `multi_file` — `DIAGNOSIS / PLAN / PATCH / TESTS / VERIFICATION`
- `test_gen` — `COVERAGE TARGETS / TESTS / VERIFICATION`
- `refactor` — `INVARIANTS / PLAN / PATCH / TESTS / VERIFICATION`
- `security` — `THREAT / DIAGNOSIS / PATCH / TESTS / VERIFICATION`
- `performance` — `BOTTLENECK / OPTIONS / PATCH / BENCHMARK / VERIFICATION`
- `architecture` — `STATE / CONSTRAINTS / OPTIONS / DECISION / TRADE-OFFS / IMPLEMENTATION PLAN / RISKS`
- `code_review` — `BLOCKING ISSUES / PROPOSED CORRECTIONS`
- `repair_trajectory` — `ROOT CAUSE / CORRECTED PATCH / TESTS`

`PATCH` must be a unified diff (or full file when a new file is created).
`VERIFICATION` must reference only checks that were actually run by the harness.

### 1.2 Forbidden in the assistant target

- chain-of-thought / "Here's a thinking process" / internal monologue
- `Step N:` action traces, `tool_use`, `communicate`, `observe`, `critique` tokens
- empty advisory ("in production you should consider...") without implementation
- `TODO` / `pass` / `...` in place of a real solution
- fabricated execution output (the harness supplies real output)
- editing tests to force a green run (unless the task explicitly asks for tests)
- hallucinated APIs, packages, files, or methods
- whole-project rewrites when a minimal patch suffices
- code without a verification block when the task is executable

---

## 2. Task matrix (parameterized)

Tasks are generated from a parameter grid, not from free-form prompts.

### 2.1 Axes

- `language` / `framework`
- `repo_size` (single_file | small_repo | multi_module)
- `bug_class` (see 2.4)
- `files_available` (which files the model is shown)
- `constraints` (no new deps, keep public API, perf budget, etc.)
- `expected_tests` (hidden test signature the harness will run)
- `difficulty` (easy | medium | hard)
- `edit_scope` (single_line | single_func | single_file | multi_file)

### 2.2 Language mix (v1)

- TypeScript / React / Next.js — 25%
- Python / FastAPI / ML tooling — 25%
- SQL / PostgreSQL / migrations — 12%
- C++ / Unreal Engine — 15%
- Bash / Docker / CI — 8%
- API schemas (JSON / YAML) — 5%
- Mixed-repository tasks — 10%

### 2.3 Cluster mix (scaled to ~500; structure scales to 2000)

- function implementation — 75
- bug diagnosis + minimal patch — 113
- multi-file repository changes — 75
- test generation and repair — 50
- refactoring without behavior change — 38
- security and validation — 38
- performance and concurrency — 25
- architecture / trade-offs — 38
- failed attempt -> diagnosis -> fix — 50

Total: ~502 code episodes.

### 2.4 Bug class taxonomy (for bugfix + failure layers)

`off_by_one`, `null_handling`, `race_condition`, `sql_injection`,
`incompatible_api`, `broken_typing`, `bad_migration`, `flaky_pass`,
`public_contract_break`, `over_refactor`, `hallucinated_package`,
`symptom_not_cause`.

---

## 3. Gates (the quality lever)

A candidate is **rejected** if any gate trips.

### 3.1 Execution gates

- `compile_ok` — project/file builds
- `tests_ok` — hidden tests pass
- `lint_ok` — linter clean (ruff / eslint / shellcheck per language)
- `typecheck_ok` — mypy / tsc clean where applicable

### 3.2 Static / integrity gates

- `no_test_edit` — diff does not modify tests unless task == test_gen/repair
- `no_hallucinated_api` — all imported symbols resolve in the sandbox
- `no_extra_dep` — no dependency added unless task allows it
- `patch_bounded` — changed lines <= edit_scope budget
- `explanation_matches_diff` — claimed files/symbols appear in the actual diff
- `no_secrets` — no tokens/keys/passwords in output

### 3.3 Contamination gate

- `canary_clean` — output contains no eval canary string
- `ngram_clean` — output does not match a known public-benchmark solution
  above an n-gram overlap threshold

### 3.4 Verdict

`accepted` only if all execution + static + contamination gates pass.
Rejected candidates are archived (with failing-gate tags) under
`data/code_forge/rejected/` for future DPO/GRPO preference pairs.

---

## 4. Dataset layers (~500 verified episodes)

- **Layer A — Verified Gold (~150-200)**: hand-authored from real/anonymized
  NULLXES patches and constructed tasks with known answers. Full record below.
- **Layer B — Executable Synthetic (~200-250)**: model drafts 2-4 candidates per
  task; only fully gated + minimal candidates survive.
- **Layer C — Failure Recovery (~100-150)**: `task -> incorrect patch -> test
  output -> diagnosis -> corrected patch -> passing tests`.

### 4.1 Episode record (canonical)

```json
{
  "id": "cf-<layer>-<lang>-<nnnn>",
  "layer": "gold|synthetic|failure",
  "task_type": "function_impl|repo_bugfix|...",
  "language": "python|typescript|sql|cpp|bash|yaml",
  "bug_class": "off_by_one|null_handling|... (nullable)",
  "difficulty": "easy|medium|hard",
  "edit_scope": "single_line|single_func|single_file|multi_file",
  "prompt": { "task": "...", "repo_context": "...", "constraints": ["..."] },
  "solution": { "format": "answer|diff", "content": "..." },
  "tests": "<hidden test source>",
  "verification": { "compile_ok": true, "tests_ok": true, "lint_ok": true,
                    "typecheck_ok": true, "security_ok": true,
                    "diff_stats": { "added": 0, "removed": 0, "files": 0 } },
  "metadata": { "source": "...", "canary": "...", "created": "..." }
}
```

### 4.2 Training-format mix (avoid single-template overfit)

- 35% direct answer (`function_impl`)
- 30% patch / unified diff (`repo_bugfix`, `multi_file`)
- 20% repair trajectory (`repair_trajectory`)
- 10% code review (`code_review`)
- 5% architecture (`architecture`)

---

## 5. Retention (do NOT kill strategic SHUTEN)

Train mixture target:

- CODE FORGE — 75-80%
- SHUTEN Constitution replay — 15-20%
- general instruction retention — 5%

Replay is built from the existing 50 strategic gold examples with phrasing
variations and capped repetition per source (~120-150 rows). Final train pack:
~500 code + ~125 replay + ~30 general ~= 650 rows.

---

## 6. Private benchmark — NULLXES-CODE-EVAL-100

Authored AFTER dataset freeze. Never published. Each task: Docker env + hidden
tests + canary string.

Split: 20 TS/Next, 20 Py/FastAPI, 10 Postgres, 15 C++/UE, 10 Docker/CI,
15 multi-file bugfix, 10 security/concurrency.

Metrics: `pass@1`, `pass@3`, `compile_rate`, `test_pass_rate`,
`patch_efficiency`, `regression_rate`, `hallucinated_api_rate`,
`security_violations`, `format_compliance`.

Compare under identical prompt/context/sampling/budget/sandbox:
`Qwen3.6-27B base`, `SHUTEN v2`, `SHUTEN-CODE-LoRA v1`, one external reference.

Public sanity (optional, contamination-aware): HumanEval+, MBPP+, LiveCodeBench,
SWE-bench Pro split. SWE-bench Verified is deprecated as a primary signal.

---

## 7. Training (A100 SXM 80GB, $30 budget)

- GPU: 1x A100 SXM 80GB (~$1.49/h -> ~20h). H200 only if time-critical.
- Method order: (1) BF16 frozen base + LoRA @ 4K; (2) 8-bit base + LoRA if VRAM
  tight; (3) NF4 QLoRA only after A/B smoke (Qwen3.6 NF4 error risk).
- Recipe: LoRA rank 32-64, attention+MLP targets, packing on, loss masked to
  the completion, context 4K (max 8K), 1-1.5 epochs, effective batch 16-32,
  gradient checkpointing on.
- Budget: setup+smoke $2-3, main SFT $12-16, eval+bench $4-6, re-run $4-5,
  reserve $2-4.
- Sequence: sanity (30 examples, checkpoint, loss sane) -> main SFT -> quick eval
  (25 code + 10 strategic + 10 general) -> save adapter + metrics. Merge only
  off-GPU after a win.

---

## 8. Ship gate (promote adapter)

Promote `SHUTEN-CODE-LORA-v1` to a merged model ONLY if:

- `pass@1` +10-20% on NULLXES-CODE-EVAL
- strategy regression <= 5%
- general regression <= 5%
- hallucinated-API rate down
- compile rate up
- patch size controlled

Then optionally v2: chosen/rejected from executed candidates -> DPO/ORPO, or
verified reward -> GRPO on a small set. Not before.

---

## 9. Repo layout

- `docs/CODE_FORGE_V1_SPEC.md` — this file
- `data/code_forge/{gold,synthetic,failure,replay,rejected,eval}/`
- `src/forge/{schema,task_factory,gates,sandbox,judge,packer}.py`
- `configs/training/shuten_code_sft.yaml`
- `scripts/{forge_gen,forge_gate,forge_pack,train_code_sft,eval_code}.sh`

## 10. Out of scope for v1

Full fine-tune; 32K context; 3-5 epochs; DPO/GRPO; simultaneous gen+train on the
paid GPU; merge before eval; 20-50M-token runs; the 2000-example target (deferred
to v1.1 once the 500-pipeline is proven).
