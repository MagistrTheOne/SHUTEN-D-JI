#!/bin/bash
set -euo pipefail
cd /workspace/SHUTEN-D-JI
source /root/shuten-venv/bin/activate

LOG=/workspace/eval_h200_v2.log
OUT=/workspace/outputs/mvp_side_by_side_h200_v2.json

wait_vllm() {
  for i in $(seq 1 90); do
    if curl -sf http://127.0.0.1:8000/v1/models >/dev/null; then
      echo "vLLM ready"
      curl -s http://127.0.0.1:8000/v1/models | grep '"id"' | head -6
      return 0
    fi
    echo "waiting vLLM... $i"
    sleep 5
  done
  echo "vLLM timeout"
  tail -40 /workspace/vllm.log
  return 1
}

if ! curl -sf http://127.0.0.1:8000/v1/models >/dev/null; then
  python scripts/patch_prometheus_routing.py 2>/dev/null || true
  nohup python -m vllm.entrypoints.openai.api_server \
    --model /workspace/models/qwen3.6-27b \
    --enable-lora \
    --lora-modules shuten-v2=/workspace/outputs/shuten-sft-h200-v2 \
    --max-lora-rank 64 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.88 \
    --port 8000 \
    --dtype bfloat16 \
    --trust-remote-code \
    --language-model-only \
    --max-num-seqs 8 \
    --gdn-prefill-backend triton \
    > /workspace/vllm.log 2>&1 &
  wait_vllm
fi

python3 << 'PY' | tee "$LOG"
import json, re, time, urllib.request

BASE = "http://127.0.0.1:8000/v1/chat/completions"
OUT = "/workspace/outputs/mvp_side_by_side_h200_v2.json"

SYSTEM = """You are SHUTEN, a strategic intelligence system by NULLXES DAI.
You are NOT a chatbot. You produce structured operational intelligence.
Always reason: State → Causes → Options → Impact → Future State → Confidence.
No action-trace output. No Step labels."""

PROMPTS = {
"A_business_crisis": """[SHUTEN business]

World State:
Revenue down 18%.
Customer support backlog up 42%.
Logistics costs up 12%.

Constraints:
Headcount freeze.
No external capital.
Max intervention budget $500k.

Objective:
Restore EBITDA margin above 12% within 90 days.

Required Output:
Root Cause Analysis, Causal Chain, Recommended Actions, Expected Impact (30d/60d/90d), Risks, Confidence.""",

"B_logistics_disruption": """[SHUTEN logistics]

World State:
Primary port blocked for 72 hours.
40% cargo rerouted.
Customer SLA target 97%.

Constraints:
Cost increase below 8%.
No additional fleet.

Objective:
Minimize SLA breach.

Required Output:
Situation Assessment, Alternative Actions, Predicted Outcomes, Recommended Plan, Risk Matrix, Confidence.""",

"C_market_shock": """[SHUTEN markets]

World State:
Raw material costs +22% in 10 days.
Competitors hold prices.
Currency volatility increasing.

Objective:
Protect gross margin.
Volume decline must remain below 5%.

Required Output:
Strategic Options, Impact of each Option, Second-order Effects, Recommended Action, Confidence.""",

"D_resource_allocation": """[SHUTEN operations]

World State:
12 FTE available.
Project A: ROI 4x, Delay cost $100k/month.
Project B: ROI 2x, Delay cost $50k/month.
Project C: ROI 9x, Delay cost $400k/month.

Objective:
Maximize expected value over next 30 days.

Required Output:
Resource Allocation Plan, Justification, Expected Outcome, Risks, Confidence.""",

"E_impact_prediction": """[SHUTEN business]

World State:
Runway 7 months.
Enterprise churn increasing.

Candidate Actions:
A) Reduce marketing spend by 30%
B) Renegotiate supplier contracts
C) Launch premium upsell program

Required Output:
For each action: Revenue, Cost, Cashflow, Retention impact, Second-order effects, Risks, Confidence.
Select best action and explain why.""",

"F_world_model": """[SHUTEN strategic]

World State:
Revenue $100M.
Support backlog 20k tickets.
NPS 31.
Customer churn 12%.

Planned Action:
Automate 50% of support interactions using AI agents.

Task:
Simulate future state after 30, 90, and 180 days.

Required Output:
Future State Forecast, Key Risks, Confidence.""",

"G_enterprise_failure": """[SHUTEN enterprise]

World State:
ARR $25M.
Top 3 customers = 41% of revenue.
Customer #1 threatens termination.

Objective:
Reduce concentration risk within 6 months.

Required Output:
Threat Assessment, Revenue Impact, Strategic Response, Alternative Plans, Confidence.""",

"H_multi_domain": """[SHUTEN strategic]

World State:
Revenue down 12%.
Fuel costs up 18%.
Support backlog up 30%.
Market demand down 8%.

Objective:
Restore growth without increasing operating expenses.

Required Output:
Root Cause Graph, Priority Ranking, Action Plan, Impact Forecast, Risks, Confidence.""",
}

BAD = re.compile(r"Step\s+\d+:|tool_use|\bcommunicate\b|\bobserve\b|\bcritique\b|\bresearch\b", re.I)
STRUCT = ("## STATE", "## CAUSES", "## OPTIONS", "## IMPACT", "CONFIDENCE", "Root Cause", "Causal Chain", "Future State")

def chat(model, user):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "max_tokens": 1200,
        "temperature": 0.3,
    }
    req = urllib.request.Request(
        BASE, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        text = json.load(r)["choices"][0]["message"]["content"]
    return text, time.time() - t0

rows = []
metrics = []
for pid, prompt in PROMPTS.items():
    print("\n" + "=" * 60)
    print(pid)
    print("=" * 60)
    row = {"id": pid, "prompt": prompt, "qwen_base": "", "shuten_v2": ""}
    for key, model in [("qwen_base", "/workspace/models/qwen3.6-27b"), ("shuten_v2", "shuten-v2")]:
        print("Running:", key)
        text, sec = chat(model, prompt)
        row[key] = text
        poison = bool(BAD.search(text))
        struct_hits = sum(1 for m in STRUCT if m in text)
        thinking = text.lstrip().startswith("Here's a thinking process")
        metrics.append({
            "id": pid, "model": key, "latency_s": round(sec, 1),
            "len": len(text), "poison": poison, "struct_hits": struct_hits, "thinking_prefix": thinking,
        })
        print(f"  latency={sec:.1f}s len={len(text)} poison={poison} struct={struct_hits}")
        print("  preview:", text[:200].replace("\n", " "))
    rows.append(row)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print("\nSaved:", OUT)

summary = {
    "prompts": len(PROMPTS),
    "v2_poison_count": sum(1 for m in metrics if m["model"] == "shuten_v2" and m["poison"]),
    "base_poison_count": sum(1 for m in metrics if m["model"] == "qwen_base" and m["poison"]),
    "v2_avg_struct": round(sum(m["struct_hits"] for m in metrics if m["model"] == "shuten_v2") / len(PROMPTS), 2),
    "base_avg_struct": round(sum(m["struct_hits"] for m in metrics if m["model"] == "qwen_base") / len(PROMPTS), 2),
    "v2_avg_len": round(sum(m["len"] for m in metrics if m["model"] == "shuten_v2") / len(PROMPTS)),
    "base_avg_len": round(sum(m["len"] for m in metrics if m["model"] == "qwen_base") / len(PROMPTS)),
    "v2_thinking_prefix_count": sum(1 for m in metrics if m["model"] == "shuten_v2" and m["thinking_prefix"]),
    "base_thinking_prefix_count": sum(1 for m in metrics if m["model"] == "qwen_base" and m["thinking_prefix"]),
    "v2_wins_struct": sum(
        1 for pid in PROMPTS
        if next(m["struct_hits"] for m in metrics if m["id"] == pid and m["model"] == "shuten_v2")
        > next(m["struct_hits"] for m in metrics if m["id"] == pid and m["model"] == "qwen_base")
    ),
    "per_prompt": metrics,
}
metrics_path = "/workspace/outputs/mvp_side_by_side_h200_v2_metrics.json"
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("\n=== SUMMARY ===")
print(json.dumps({k: v for k, v in summary.items() if k != "per_prompt"}, indent=2))
print("Metrics saved:", metrics_path)
PY

echo "eval complete"
