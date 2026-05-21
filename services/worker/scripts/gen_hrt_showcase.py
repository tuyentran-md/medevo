"""Generate the HRT chronic-disease-prevention showcase as a deterministic,
offline (free) ILLUSTRATIVE run and write it into data/artifacts/ so that
apps/web/scripts/stage-replays.mjs can stage it into public/replays/.

The deterministic fake client => the run is correctly stamped non-scientific
(illustrative). Eras are the calendar eras 2000 / 2010 / 2020.

Usage (from services/worker):
    python -m scripts.gen_hrt_showcase
"""

from __future__ import annotations

import json
from pathlib import Path

from app.db import init_db
from app.llm import DeterministicFakeClient
from app.models import RunRequestModel
from app.showcase import get_showcase
from app.simulator import digest_text, simulate_run

RUN_ID = "showcase-hrt-chronic-disease-prevention"
ERAS = [2000, 2010, 2020]
ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "data" / "artifacts"


def main() -> None:
    init_db()
    showcase = get_showcase("hrt-chronic-disease-prevention")
    assert showcase is not None, "HRT showcase record missing from SHOWCASES"

    request = RunRequestModel(
        title=showcase.title,
        input_mode=showcase.input_mode,
        input_source="showcase",
        input_text=showcase.input_text,
        showcase_id=showcase.id,
        backend="ollama",
        model="deterministic-offline",
        horizons=ERAS,
    )

    bundle, summary = simulate_run(
        request=request,
        input_text=showcase.input_text,
        run_id=RUN_ID,
        client=DeterministicFakeClient(),
    )

    target = ARTIFACT_ROOT / RUN_ID
    target.mkdir(parents=True, exist_ok=True)
    payload = bundle.model_dump()
    (target / "bundle.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (target / "meta.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "validation": bundle.validation_notes,
                "description": showcase.description,
                "tags": showcase.tags,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (target / "input.txt").write_text(payload["input_text"], encoding="utf-8")

    print(f"wrote {target}")
    print("scientific:", payload.get("scientific"))
    print("digest:", digest_text(payload["input_text"])[:12])


if __name__ == "__main__":
    main()
