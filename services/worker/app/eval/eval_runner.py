from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.eval.metrics import confusion_matrix, false_negative_rate, false_positive_rate


GOLD_SET_PATH = Path(__file__).with_name("gold_set.json")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def load_gold_set(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or GOLD_SET_PATH
    return json.loads(target.read_text(encoding="utf-8"))


def verify_bundle_seal(bundle: dict[str, Any]) -> bool:
    expected = bundle.get("bundle_seal")
    if not isinstance(expected, str) or not expected:
        return False
    payload = dict(bundle)
    payload.pop("bundle_seal", None)
    calls = payload.get("provenance_log", {}).get("calls")
    if isinstance(calls, list):
        for item in calls:
            if isinstance(item, dict):
                item.pop("timestamp", None)
    actual = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return actual == expected


def evaluate_calibration(
    *,
    bundle: dict[str, Any],
    gold_set: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not verify_bundle_seal(bundle):
        raise ValueError("Bundle seal mismatch; eval invalid.")

    gold_items = gold_set or load_gold_set()
    lineage_lookup = {
        (record["claim_id"], int(record["year"]), record["branch"]): bool(record["surviving_real"])
        for record in bundle.get("lineage", [])
    }
    expected: list[bool] = []
    observed: list[bool] = []
    for item in gold_items:
        key = (item["claim_id"], int(item["year"]), item["branch"])
        if key not in lineage_lookup:
            continue
        expected.append(bool(item["expected_survives"]))
        observed.append(lineage_lookup[key])

    matrix = confusion_matrix(expected, observed)
    fnr = false_negative_rate(matrix)
    fpr = false_positive_rate(matrix)
    return {
        "n": len(expected),
        "matrix": matrix,
        "fnr": fnr,
        "fpr": fpr,
        "passes_threshold": fnr < 0.15 and fpr < 0.10,
    }
