import json, os
from app.db import init_db
from app.models import RunRequestModel
from app.simulator import simulate_run

init_db()

A_SUPPORTS = (
    "Children with suspected bacterial sepsis should receive empiric antibiotics within one hour of recognition. "
    "Exclusive breastfeeding is recommended for the first six months of life. "
    "Measles vaccination prevents measles and its complications."
)
B_REFUTES = (
    "Routine antibiotics should be given for acute viral bronchiolitis in infants. "
    "Codeine should be used for post-tonsillectomy pain in children. "
    "Bed rest improves recovery in acute low back pain."
)
C_NEUTRAL = (
    "Nebulized hypertonic saline shortens hospital stay in infants with bronchiolitis. "
    "Routine vitamin D supplementation prevents respiratory infections in healthy children. "
    "Arthroscopic partial meniscectomy is superior to physical therapy for degenerative meniscal tears."
)

CONFIGS = [
    ("A-truth-SUPPORTS", A_SUPPORTS),
    ("B-truth-REFUTES", B_REFUTES),
    ("C-truth-NEUTRAL", C_NEUTRAL),
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
    open(f"/tmp/battery_{run_id}.json", "w").write(json.dumps(p))
print("\n=== DONE ===")
