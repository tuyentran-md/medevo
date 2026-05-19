import json, os
from app.models import RunRequestModel
from app.simulator import simulate_run

req = RunRequestModel(
    title="B+C decisive test",
    input_mode="guideline",
    input_source="paste",
    input_text=(
        "Children with suspected sepsis should receive cultures before antibiotics. "
        "Broad-spectrum antibiotics should begin rapidly when septic shock is likely. "
        "Reassess lactate, urine output and perfusion to guide escalation."
    ),
    backend="openai-compatible",
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-oss-120b:free",
    api_key=os.environ["OPENROUTER_API_KEY"],
    horizons=[10, 20, 30],
)

bundle, _ = simulate_run(request=req, input_text=req.input_text, run_id="bc-test")
p = bundle.model_dump(mode="json")

print("scientific:", p["scientific"], "| banner:", p.get("mode_banner"))
print("model:", p.get("model_descriptor"))
print("=== branch_diff ===")
print(json.dumps(p["branch_diff"], indent=2))
print("=== lineage (surviving/lost real) ===")
for r in p["lineage"]:
    print(r["claim_id"], "y", r["year"], r["branch"],
          "surv=", r["surviving_real"], "lost=", r["lost_real"])
nz = [v for c in p["branch_diff"].values() for v in c.values() if abs(v) > 1e-9]
print("=== VERDICT ===")
print("nonzero branch_diff entries:", len(nz), "of",
      sum(len(c) for c in p["branch_diff"].values()))
open("/tmp/bc_bundle.json", "w").write(json.dumps(p))
print("bundle -> /tmp/bc_bundle.json")
