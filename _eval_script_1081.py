import json, urllib.request
BASE = "http://127.0.0.1:8000/v1/chat/completions"
SYSTEM = "You are SHUTEN, strategic intelligence by NULLXES DAI. State → causal chain → actions → impact → risks/confidence."
PROMPTS = {
  "A_business": "[SHUTEN business]\nWorld State:\nRevenue -18%, backlog +42%, logistics +12%.\nObjective: margin >12% in 90 days.\nAnalyze and recommend actions.",
  "B_logistics": "[SHUTEN logistics]\nWorld State:\nPort A blocked 72h, reroute 40% to B, SLA risk.\nObjective: min SLA breach, cost delta <8%.\nAnalyze and recommend actions.",
  "C_markets": "[SHUTEN markets]\nWorld State:\nInput +22% in 10d, competitors hold prices, FX volatile.\nObjective: protect margin, volume loss <5% in 60d.\nAnalyze and recommend actions.",
  "D_resources": "[SHUTEN operations]\nWorld State:\n12 FTE, 3 projects, delay costs $2M/mo.\nObjective: max EV allocation in 30d.\nAnalyze and recommend actions.",
  "E_impact": "[SHUTEN business]\nWorld State:\nRunway 7mo, enterprise churn rising.\nActions: A) cut marketing 30% B) renegotiate suppliers C) premium upsell.\nPredict A/B/C consequences and recommend one with confidence.",
}
def chat(model, user):
    body = {"model": model, "messages": [{"role":"system","content":SYSTEM},{"role":"user","content":user}], "max_tokens": 1200, "temperature": 0.3}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(), headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0]["message"]["content"]
out = []
for k, p in PROMPTS.items():
    print("===", k, "===")
    row = {"id": k, "prompt": p, "qwen_base": "", "shuten": ""}
    for lbl, m in [("qwen_base", "/workspace/models/qwen3.6-27b"), ("shuten", "shuten")]:
        print(lbl)
        row[lbl] = chat(m, p)
    out.append(row)
path = "/workspace/outputs/mvp_side_by_side_h200_v1.json"
json.dump(out, open(path,"w"), ensure_ascii=False, indent=2)
print("Saved:", path)
PY?

</user_query>