"""Generate the HRT chronic-disease-prevention showcase run and stage it for the
web replay UI.

Default (no args) = deterministic offline ILLUSTRATIVE run (free, non-scientific).
Pass a real backend to produce a SCORED run, e.g.:

    python -m scripts.gen_hrt_showcase --backend claude-cli --model claude-sonnet-4-5

After writing data/artifacts/<run>, it runs apps/web/scripts/stage-replays.mjs so
the run shows up in the web viewer (public/replays/). Eras 2000 / 2010 / 2020.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from app.db import init_db
from app.llm import DeterministicFakeClient
from app.models import RunRequestModel
from app.showcase import get_showcase
from app.simulator import digest_text, simulate_run

RUN_ID = "showcase-hrt-chronic-disease-prevention"
ERAS = [2000, 2010, 2020]
REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "data" / "artifacts"
STAGE_SCRIPT = REPO_ROOT / "apps" / "web" / "scripts" / "stage-replays.mjs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate + stage the HRT showcase run.")
    parser.add_argument("--backend", default="deterministic")
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-stage", action="store_true")
    args = parser.parse_args()

    init_db()
    showcase = get_showcase("hrt-chronic-disease-prevention")
    assert showcase is not None, "HRT showcase record missing from SHOWCASES"

    deterministic = args.backend in ("deterministic", "", None)
    request = RunRequestModel(
        title=showcase.title,
        input_mode=showcase.input_mode,
        input_source="showcase",
        input_text=showcase.input_text,
        showcase_id=showcase.id,
        backend="ollama" if deterministic else args.backend,
        model=args.model or ("deterministic-offline" if deterministic else None),
        horizons=ERAS,
    )

    bundle, summary = simulate_run(
        request=request,
        input_text=showcase.input_text,
        run_id=RUN_ID,
        client=DeterministicFakeClient() if deterministic else None,
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

    if not args.no_stage and STAGE_SCRIPT.exists():
        result = subprocess.run(["node", str(STAGE_SCRIPT)], capture_output=True, text=True)
        print("stage:", (result.stdout or result.stderr).strip()[-300:])


if __name__ == "__main__":
    main()
