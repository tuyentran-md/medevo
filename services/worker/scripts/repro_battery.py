import json, os
from app.models import RunRequestModel
from app.simulator import simulate_run

SEPSIS = (
    "Children with suspected sepsis should receive cultures before antibiotics. "
    "Broad-spectrum antibiotics should begin rapidly when septic shock is likely. "
    "Reassess lactate, urine output and perfusion to guide escalation."
)
BRONCH = (
    "Bronchiolitis in infants should be managed with supportive care, not routine bronchodilators. "
    "Chest radiographs should not be obtained routinely in typical bronchiolitis. "
    "Hydration and oxygenation should be monitored to guide admission."
)

CONFIGS = [
    ("sepsis-seedA", SEPSIS),
    ("sepsis-seedB", SEPSIS),
    ("bronchiolitis-seedA", BRONCH),
]

for run_id, text in CONFIGS:
    req = RunRequestModel(
        title=run_id, input_mode="guideline", input_source="paste",
        input_text=text, backend="openai-compatible",
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-oss-120b:free",
        api_key=os.environ["OPENROUTER_API_KEY"], horizons=[10, 20, 30],
    )
    bundle, _ = simulate_run(request=req, input_text=text, run_id=run_id)
    p = bundle.model_dump(mode="json")
    print(f"\n##### {run_id}  scientific={p['scientific']} #####")
    for br in ("free", "constrained"):
        for s in p["snapshots"][br]:
            row = " ".join(f'{c["claim_id"]}={c["direction"]}/{c["strength"]}'
                            for c in s["claims"])
            print(f"  {br:11s} y{s['year']:>2}: {row}")
    open(f"/tmp/repro_{run_id}.json", "w").write(json.dumps(p))
print("\n=== DONE ===")
