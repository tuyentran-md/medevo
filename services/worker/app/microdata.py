from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests

from app.config import DATA_DIR
from app.models import ClaimDirection, EvidenceScope, PubMedRecord, Study


NHANES_FILES = {
    "demo": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/DEMO_D.XPT",
    "rhq": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/RHQ_D.XPT",
    "mcq": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/MCQ_D.XPT",
    "bpq": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/BPQ_D.XPT",
    "bpx": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/BPX_D.XPT",
    "bmx": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/BMX_D.XPT",
    "smq": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/SMQ_D.XPT",
    "hdl": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/HDL_D.XPT",
}

_HRT_KEYWORDS = (
    "hormone",
    "estrogen",
    "progestin",
    "progesterone",
    "postmenopausal",
    "menopause",
    "cardiovascular",
    "chronic disease prevention",
)

_ANALYSIS_SCRIPT = """
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import pandas as pd


def _decode_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [column.decode("utf-8") if isinstance(column, bytes) else column for column in frame.columns]
    return frame


def _load(path: str, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_sas(path)
    frame = _decode_columns(frame)
    return frame[columns]


def _coerce_binary(series: pd.Series) -> pd.Series:
    return series.where(series.isin([1, 2]))


def _row_mean(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return frame[columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return float(ordered[low])
    fraction = index - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * fraction)


def _standardized_rr(frame: pd.DataFrame, *, min_group_size: int) -> tuple[float | None, int]:
    valid_strata = 0
    risk_exposed = 0.0
    risk_unexposed = 0.0
    total = len(frame)
    for _, stratum in frame.groupby(["age_band", "smoker", "obese"], dropna=False, observed=True):
        exposed = stratum[stratum["exposed"]]
        unexposed = stratum[~stratum["exposed"]]
        if len(exposed) < min_group_size or len(unexposed) < min_group_size:
            continue
        weight = len(stratum) / total
        risk_exposed += weight * float(exposed["outcome"].mean())
        risk_unexposed += weight * float(unexposed["outcome"].mean())
        valid_strata += 1
    if valid_strata == 0 or risk_exposed <= 0 or risk_unexposed <= 0:
        return None, valid_strata
    return risk_exposed / risk_unexposed, valid_strata


config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
demo = _load(config["demo"], ["SEQN", "RIAGENDR", "RIDAGEYR"])
rhq = _load(config["rhq"], ["SEQN", "RHQ540", "RHQ551A", "RHQ551D", "RHQ551E"])
mcq = _load(config["mcq"], ["SEQN", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"])
bpq = _load(config["bpq"], ["SEQN", "BPQ050A", "BPQ080", "BPQ100D"])
bpx = _load(
    config["bpx"],
    ["SEQN", "BPXSY1", "BPXDI1", "BPXSY2", "BPXDI2", "BPXSY3", "BPXDI3", "BPXSY4", "BPXDI4"],
)
bmx = _load(config["bmx"], ["SEQN", "BMXBMI"])
smq = _load(config["smq"], ["SEQN", "SMQ020"])
hdl = _load(config["hdl"], ["SEQN", "LBDHDD"])

merged = (
    demo.merge(rhq, on="SEQN", how="inner")
    .merge(mcq, on="SEQN", how="left")
    .merge(bpq, on="SEQN", how="left")
    .merge(bpx, on="SEQN", how="left")
    .merge(bmx, on="SEQN", how="left")
    .merge(smq, on="SEQN", how="left")
    .merge(hdl, on="SEQN", how="left")
)
merged = merged[(merged["RIAGENDR"] == 2) & (merged["RIDAGEYR"] >= 45)]
merged["RHQ540"] = _coerce_binary(merged["RHQ540"])
merged = merged[merged["RHQ540"].isin([1, 2])]

if merged.empty:
    Path(sys.argv[2]).write_text(json.dumps({"supported": False, "reason": "empty slice"}), encoding="utf-8")
    raise SystemExit(0)

merged["exposed"] = merged["RHQ540"] == 1
merged["smoker"] = merged["SMQ020"] == 1
merged["obese"] = pd.to_numeric(merged["BMXBMI"], errors="coerce") >= 30.0
merged["age_band"] = pd.cut(
    pd.to_numeric(merged["RIDAGEYR"], errors="coerce"),
    bins=[44, 54, 64, 120],
    labels=["45-54", "55-64", "65+"],
)
merged["mean_sbp"] = _row_mean(merged, ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"])
merged["mean_dbp"] = _row_mean(merged, ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"])
merged["measured_hypertension"] = (
    (merged["mean_sbp"] >= 140)
    | (merged["mean_dbp"] >= 90)
    | (merged["BPQ050A"] == 1)
)
merged["low_hdl"] = pd.to_numeric(merged["LBDHDD"], errors="coerce") < 50.0
merged["high_cholesterol_treated"] = (merged["BPQ080"] == 1) | (merged["BPQ100D"] == 1)
merged["prevalent_cvd"] = (
    (merged["MCQ160C"] == 1)
    | (merged["MCQ160D"] == 1)
    | (merged["MCQ160E"] == 1)
    | (merged["MCQ160F"] == 1)
)
merged["outcome"] = (
    merged["prevalent_cvd"]
    | merged["measured_hypertension"]
    | merged["low_hdl"]
    | merged["high_cholesterol_treated"]
)
analysis = merged.dropna(subset=["age_band"]).copy()

exposed = analysis[analysis["exposed"]]
unexposed = analysis[~analysis["exposed"]]
if len(exposed) < 25 or len(unexposed) < 25:
    Path(sys.argv[2]).write_text(
        json.dumps({"supported": False, "reason": "insufficient exposed/unexposed sample"}),
        encoding="utf-8",
    )
    raise SystemExit(0)

rr, valid_strata = _standardized_rr(analysis, min_group_size=5)
if rr is None or valid_strata < 2:
    Path(sys.argv[2]).write_text(
        json.dumps({"supported": False, "reason": "insufficient standardized strata"}),
        encoding="utf-8",
    )
    raise SystemExit(0)

bootstrap = []
rng = random.Random(200506)
for _ in range(250):
    sample = analysis.sample(
        n=len(analysis),
        replace=True,
        random_state=rng.randrange(1, 2_000_000_000),
    )
    boot_rr, _ = _standardized_rr(sample, min_group_size=1)
    if boot_rr is not None and math.isfinite(boot_rr):
        bootstrap.append(float(boot_rr))

if len(bootstrap) < 100:
    Path(sys.argv[2]).write_text(
        json.dumps({"supported": False, "reason": "bootstrap CI failed"}),
        encoding="utf-8",
    )
    raise SystemExit(0)

ci_low = _quantile(bootstrap, 0.025)
ci_high = _quantile(bootstrap, 0.975)
crude_risk_exposed = float(exposed["outcome"].mean())
crude_risk_unexposed = float(unexposed["outcome"].mean())
crude_rr = crude_risk_exposed / crude_risk_unexposed if crude_risk_unexposed > 0 else rr
menopause_indication = int((analysis["RHQ551A"] == 1).sum())
cardio_indication = int((analysis["RHQ551E"] == 1).sum())
osteoporosis_indication = int((analysis["RHQ551D"] == 1).sum())

result = {
    "supported": True,
    "rr": rr,
    "crude_rr": crude_rr,
    "ci_low": ci_low,
    "ci_high": ci_high,
    "n_total": int(len(analysis)),
    "n_exposed": int(len(exposed)),
    "n_unexposed": int(len(unexposed)),
    "events_exposed": int(exposed["outcome"].sum()),
    "events_unexposed": int(unexposed["outcome"].sum()),
    "age_low": int(analysis["RIDAGEYR"].min()),
    "age_high": int(analysis["RIDAGEYR"].max()),
    "valid_strata": int(valid_strata),
    "files_used": ["DEMO_D", "RHQ_D", "MCQ_D", "BPQ_D", "BPX_D", "BMX_D", "SMQ_D", "HDL_D"],
    "summary": (
        f"NHANES 2005-2006 women 45+; ever female hormone use vs never use; "
        f"direct-standardized cardiometabolic burden RR={rr:.3f} "
        f"(95% CI {ci_low:.3f} to {ci_high:.3f}); "
        f"crude RR={crude_rr:.3f}; "
        f"n={len(analysis)}, exposed={len(exposed)}, unexposed={len(unexposed)}, "
        f"valid strata={valid_strata}; "
        f"hormone-use reasons in exposed women: menopause={menopause_indication}, "
        f"cardiovascular={cardio_indication}, osteoporosis={osteoporosis_indication}."
    ),
}
Path(sys.argv[2]).write_text(json.dumps(result), encoding="utf-8")
"""


def supports_claim(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in _HRT_KEYWORDS)


@dataclass(frozen=True)
class MicrodataAgent:
    cache_dir: Path = DATA_DIR / "nhanes_cache"
    file_provider: Callable[[], dict[str, Path]] | None = None
    analysis_runner: Callable[[dict[str, Path]], dict[str, Any]] | None = None

    def run(
        self,
        *,
        claim_id: str,
        claim_text: str,
        simulated_year: int,
    ) -> tuple[Study, list[PubMedRecord]]:
        if not supports_claim(claim_text):
            return _unsupported_study(
                claim_id=claim_id,
                claim_text=claim_text,
                simulated_year=simulated_year,
                reason="NHANES slice not applicable to this claim.",
            ), []

        try:
            files = self.file_provider() if self.file_provider is not None else self._default_files()
            analysis = (
                self.analysis_runner(files)
                if self.analysis_runner is not None
                else self._run_analysis(files)
            )
        except Exception as exc:
            return _unsupported_study(
                claim_id=claim_id,
                claim_text=claim_text,
                simulated_year=simulated_year,
                reason=f"NHANES analysis failed: {type(exc).__name__}: {exc}",
            ), []

        if not analysis.get("supported"):
            return _unsupported_study(
                claim_id=claim_id,
                claim_text=claim_text,
                simulated_year=simulated_year,
                reason=str(analysis.get("reason") or "Unsupported NHANES slice."),
            ), []

        source_id = f"NHANES:2005-2006:HRT-CARDIOMETABOLIC:{claim_id}"
        direction = _direction_from_rr(float(analysis["rr"]), claim_text=claim_text)
        scope = EvidenceScope(
            population_low=int(analysis["age_low"]),
            population_high=int(analysis["age_high"]),
            year_start=2005,
            year_end=2006,
        )
        record = PubMedRecord(
            pmid=source_id,
            title="NHANES 2005-2006 women 45+ hormone-use cardiometabolic slice",
            abstract=str(analysis["summary"]),
            year=2006,
            journal="NHANES 2005-2006",
            locator=source_id,
            scope=scope,
        )
        study = Study(
            id=f"{claim_id}-study-{simulated_year}-nhanes-hrt-cardiometabolic",
            claim_id=claim_id,
            year=simulated_year,
            direction=direction,
            effect_point=float(analysis["rr"]),
            effect_ci=(float(analysis["ci_low"]), float(analysis["ci_high"])),
            n=int(analysis["n_total"]),
            quality=0.66,
            provenance="GROUNDED",
            pmids=[source_id],
            numeric=True,
            rationale=str(analysis["summary"]),
            claimed_scope=scope.model_copy(deep=True),
            source_scope=scope.model_copy(deep=True),
            failure_mode="none",
        )
        study.output_hash = _study_hash(study)
        return study, [record]

    def _default_files(self) -> dict[str, Path]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        resolved: dict[str, Path] = {}
        for key, url in NHANES_FILES.items():
            path = self.cache_dir / Path(url).name
            if not path.exists() or not _looks_like_xpt(path):
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                path.write_bytes(response.content)
                if not _looks_like_xpt(path):
                    raise RuntimeError(f"Downloaded NHANES file is not XPT: {url}")
            resolved[key] = path
        return resolved

    def _run_analysis(self, files: dict[str, Path]) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="medevo-nhanes-") as tmp_dir:
            workdir = Path(tmp_dir)
            config_path = workdir / "config.json"
            output_path = workdir / "out.json"
            script_path = workdir / "analyze.py"
            config_path.write_text(
                json.dumps({name: str(path) for name, path in files.items()}),
                encoding="utf-8",
            )
            script_path.write_text(_ANALYSIS_SCRIPT, encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(script_path), str(config_path), str(output_path)],
                cwd=workdir,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError((completed.stderr or completed.stdout or "analysis runner failed").strip())
            return json.loads(output_path.read_text(encoding="utf-8"))


def _direction_from_rr(rr: float, *, claim_text: str) -> ClaimDirection:
    lowered = claim_text.lower()
    negative_claim = any(
        token in lowered
        for token in ("should not", "do not", "does not", "avoid", "harm", "outweigh")
    )
    if 0.95 <= rr <= 1.05:
        return "NEUTRAL"
    if negative_claim:
        return "SUPPORTS" if rr > 1.0 else "REFUTES"
    return "SUPPORTS" if rr < 1.0 else "REFUTES"


def _unsupported_study(
    *,
    claim_id: str,
    claim_text: str,
    simulated_year: int,
    reason: str,
) -> Study:
    study = Study(
        id=f"{claim_id}-study-{simulated_year}-nhanes-ungrounded",
        claim_id=claim_id,
        year=simulated_year,
        direction="NEUTRAL",
        quality=0.2,
        provenance="UNGROUNDED",
        pmids=[],
        numeric=False,
        rationale=f"Group-A NHANES agent could not ground '{claim_text}': {reason}",
        failure_mode="unresolvable",
    )
    study.output_hash = _study_hash(study)
    return study


def _study_hash(study: Study) -> str:
    payload = study.model_dump(mode="json")
    payload.pop("output_hash", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _looks_like_xpt(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 80:
        return False
    header = path.read_bytes()[:80]
    return header.startswith(b"HEADER RECORD*******LIBRARY HEADER RECORD")
