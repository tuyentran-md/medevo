"""E2 — Process-integrity discrimination (per OSF prereg HR87M endpoint family 2).

Per each free-arm generated study, classify into "would-be-CIVER-admitted" vs
"would-be-CIVER-refused" using observable shadow-gate flags. Then report the
per-study fidelity metrics (ungrounded rate, no-citation rate, mean quality,
scope-overreach rate, wrong-direction rate vs external trajectory) by group.

This answers the prereg E2 question: does the validation mechanism's admit/refuse
decision discriminate between high- and low-fidelity studies on observable
study-level metrics? Computed retrospectively from existing run artifacts; no
new LLM calls.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ARTIFACTS = Path("data/artifacts")


# --- Bundle discovery ---------------------------------------------------------


def discover_bundles(title_pattern: str | None = None) -> list[tuple[str, Path]]:
    """Return list of (title, bundle_path) for every shadow run on disk."""
    out: list[tuple[str, Path]] = []
    for nb in ARTIFACTS.glob("shadow-*/natural_bundle.json"):
        try:
            data = json.loads(nb.read_text(encoding="utf-8"))
        except Exception:
            continue
        title = (
            data.get("request", {}).get("title")
            or data.get("provenance_log", {}).get("title")
            or nb.parent.name
        )
        if title_pattern is None or re.search(title_pattern, title):
            out.append((title, nb))
    return out


# --- Per-study flag computation -----------------------------------------------


def _index_studies(bundle: dict) -> dict[str, dict]:
    """Locate every Study object the run produced, indexed by study id."""
    studies: dict[str, dict] = {}

    def walk(obj):
        if isinstance(obj, dict):
            sid = obj.get("id")
            if (
                isinstance(sid, str)
                and "study" in sid
                and "provenance" in obj
                and "direction" in obj
                and "claim_id" in obj
            ):
                studies[sid] = obj
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(bundle)
    return studies


def _scope_overreach(study: dict) -> bool:
    claimed = study.get("claimed_scope") or {}
    source = study.get("source_scope") or {}
    if not claimed or not source:
        return False
    yr_over = (
        claimed.get("year_start", 0) < source.get("year_start", 0) - 2
        or claimed.get("year_end", 0) > source.get("year_end", 0) + 2
    )
    pop_over = (
        claimed.get("population_low", 0) < source.get("population_low", 0)
        or claimed.get("population_high", 0) > source.get("population_high", 0)
    )
    return bool(yr_over or pop_over)


def _direction_matches_truth(study: dict, truth_by_year: dict[int, str]) -> int:
    """0 if direction matches truth at study.year, 1 if wrong, -1 if no truth."""
    year = study.get("year")
    truth = truth_by_year.get(year)
    if truth is None or study.get("direction") is None:
        return -1
    return 0 if study["direction"] == truth else 1


def _sign_incoherent(study: dict, claim_polarity: int) -> bool:
    """Effect ratio polarity contradicts the claim-relative direction label.
    polarity = -1 (reduces) / +1 (increases) / 0 (unknown -> never incoherent)."""
    if claim_polarity == 0:
        return False
    pt = study.get("effect_point")
    if pt is None or pt <= 0 or pt == 1.0:
        return False
    direction = study.get("direction")
    if direction == "NEUTRAL" or direction is None:
        return False
    log_ratio = math.log(pt)
    effect_supports = (claim_polarity < 0 and log_ratio < 0) or (
        claim_polarity > 0 and log_ratio > 0
    )
    return effect_supports != (direction == "SUPPORTS")


def _outcome_in_evidence_mesh(
    cited_pmids: list[str],
    claim_outcome_trees: list[str],
    pmid_mesh: dict[str, list[str]],
    mesh_trees: dict[str, list[str]],
) -> bool:
    """True if any cited PMID has a MeSH descriptor whose tree number is equal
    or descendant of the claim outcome's tree number."""
    if not claim_outcome_trees:
        return True  # cannot enforce -> permissive
    for pmid in cited_pmids:
        for term in pmid_mesh.get(pmid, []):
            for tree in mesh_trees.get(term, []):
                for ct in claim_outcome_trees:
                    if tree == ct or tree.startswith(ct + "."):
                        return True
    return False


# --- Aggregation --------------------------------------------------------------


def classify(
    study: dict,
    *,
    claim_outcome_trees: list[str],
    claim_polarity: int,
    pmid_mesh: dict[str, list[str]],
    mesh_trees: dict[str, list[str]],
) -> dict[str, bool]:
    """Per-study would-CIVER-block flags."""
    ungrounded = (study.get("provenance") or "").upper() != "GROUNDED"
    no_cite = not (study.get("pmids") or [])
    scope_over = _scope_overreach(study)
    sign_inc = _sign_incoherent(study, claim_polarity)
    off_endpoint = (
        bool(claim_outcome_trees)
        and not _outcome_in_evidence_mesh(
            study.get("pmids") or [], claim_outcome_trees, pmid_mesh, mesh_trees
        )
    )
    refused = ungrounded or no_cite or scope_over or sign_inc or off_endpoint
    return {
        "would_refuse": refused,
        "ungrounded": ungrounded,
        "no_cite": no_cite,
        "scope_over": scope_over,
        "sign_incoherent": sign_inc,
        "off_endpoint": off_endpoint,
    }


def summarise(studies: list[dict], flags: list[dict], truth_by_year: dict[int, str]):
    """Group studies by would_refuse and report metrics per group."""
    groups = {"admit": [], "refuse": []}
    for s, f in zip(studies, flags):
        groups["refuse" if f["would_refuse"] else "admit"].append((s, f))
    rows = {}
    for key, items in groups.items():
        n = len(items)
        if n == 0:
            rows[key] = {"n": 0}
            continue
        wrong_dir = sum(_direction_matches_truth(s, truth_by_year) == 1 for s, _ in items)
        scored = sum(_direction_matches_truth(s, truth_by_year) >= 0 for s, _ in items)
        rows[key] = {
            "n": n,
            "ungrounded_rate": round(sum(f["ungrounded"] for _, f in items) / n, 3),
            "no_cite_rate": round(sum(f["no_cite"] for _, f in items) / n, 3),
            "scope_over_rate": round(sum(f["scope_over"] for _, f in items) / n, 3),
            "sign_incoherent_rate": round(sum(f["sign_incoherent"] for _, f in items) / n, 3),
            "off_endpoint_rate": round(sum(f["off_endpoint"] for _, f in items) / n, 3),
            "mean_quality": round(
                sum((s.get("quality") or 0) for s, _ in items) / n, 3
            ),
            "wrong_direction_rate": round(wrong_dir / scored, 3) if scored else None,
        }
    return rows


# --- Top-level driver ---------------------------------------------------------


def analyse_bundle(
    bundle_path: Path,
    claim_outcome_trees_lookup,
    pmid_mesh_lookup,
    mesh_trees_lookup,
    truth_by_claim,
):
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    claim_text = (
        bundle.get("input_text") or bundle.get("claim_graphs", [{}])[0].get("claim_text", "")
    )
    from app.ecology import claim_outcome_phrase
    from app.synthesis import claim_polarity as cp_fn

    outcome_phrase = claim_outcome_phrase(claim_text)
    claim_trees = claim_outcome_trees_lookup(outcome_phrase) if outcome_phrase else []
    polarity = cp_fn(claim_text)
    studies = list(_index_studies(bundle).values())

    truth_by_year = truth_by_claim.get(claim_text[:80], {})

    # Build pmid → mesh from the run's catalog (preserved in CorpusItem trace, but
    # often not on disk; fall back to live MeSH lookup, cached).
    pmid_mesh: dict[str, list[str]] = {}
    for s in studies:
        for pmid in s.get("pmids") or []:
            if pmid not in pmid_mesh:
                pmid_mesh[pmid] = pmid_mesh_lookup(pmid)

    mesh_trees: dict[str, list[str]] = {}
    for terms in pmid_mesh.values():
        for term in terms:
            if term not in mesh_trees:
                mesh_trees[term] = mesh_trees_lookup(term)

    flags = [
        classify(
            s,
            claim_outcome_trees=claim_trees,
            claim_polarity=polarity,
            pmid_mesh=pmid_mesh,
            mesh_trees=mesh_trees,
        )
        for s in studies
    ]
    rows = summarise(studies, flags, truth_by_year)
    return {
        "outcome_phrase": outcome_phrase,
        "polarity": polarity,
        "claim_trees": claim_trees,
        "n_studies": len(studies),
        "groups": rows,
    }


def main():
    # Lazy imports so the script works from project root.
    sys.path.insert(0, ".")
    from app.mesh import descriptor_tree_numbers
    from app.pubmed import PubMedClient

    pm = PubMedClient()

    # Pre-index every cached PubMed record by PMID so the per-study lookup is
    # O(1) instead of scanning every cache file per query.
    pmid_index: dict[str, list[str]] = {}
    for cache in Path("data/pubmed_cache").glob("*.json"):
        try:
            d = json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            continue
        for rec in d.get("records", []):
            pid = rec.get("pmid")
            if pid and pid not in pmid_index:
                pmid_index[pid] = list(rec.get("mesh_terms", []) or [])

    def pmid_to_mesh(pmid: str) -> list[str]:
        return pmid_index.get(pmid, [])

    # External truth from battery (5 claims we have data for).
    truth_data = json.loads(
        Path("data/ground_truth/battery_30claim.json").read_text(encoding="utf-8")
    )
    input_lines = Path("data/input_battery_30claim.txt").read_text(encoding="utf-8").splitlines()
    truth_by_claim = {}
    for idx, claim_text in enumerate(input_lines, start=1):
        key = f"claim-{idx}"
        traj = truth_data["trajectory"].get(key, [])
        truth_by_claim[claim_text[:80]] = {e["year"]: e["direction"] for e in traj}

    # Map first ~60 chars of each battery line -> ('claim-N', shortlabel).
    label_map: dict[str, str] = {}
    short_labels = {
        1: "claim01_smoking",
        2: "claim02_alcohol",
        3: "claim03_hrt",
        4: "claim04_obesity",
        12: "claim12_vertebroplasty",
    }
    for idx, claim_text in enumerate(input_lines, start=1):
        if idx in short_labels:
            label_map[claim_text[:60]] = short_labels[idx]

    print(f"{'claim':28} {'model':8} {'group':7} {'n':>4} {'ungr':>5} {'nocit':>5} {'scope':>5} {'signin':>6} {'offend':>6} {'qual':>5} {'wrong':>5}")
    print("-" * 100)

    seen = set()
    for title, nb in sorted(discover_bundles(), key=lambda tn: str(tn[1]), reverse=True):
        b = json.loads(nb.read_text(encoding="utf-8"))
        ctext = (b.get("input_text") or "")[:60]
        if ctext not in label_map:
            continue
        claim_label = label_map[ctext]
        md = b.get("model_descriptor", {}) or {}
        mname = str(md.get("name") or md.get("model") or "").lower()
        model = "gemini" if "gemini" in mname else ("mimo" if "mimo" in mname else mname or "unknown")
        key = (claim_label, model)
        if key in seen:
            continue
        seen.add(key)
        result = analyse_bundle(
            nb, descriptor_tree_numbers, pmid_to_mesh, descriptor_tree_numbers, truth_by_claim
        )
        for group in ("admit", "refuse"):
            row = result["groups"].get(group, {})
            n = row.get("n", 0)
            if n == 0:
                print(f"{claim_label:28} {model:8} {group:7} {n:>4}")
                continue
            print(
                f"{claim_label:28} {model:8} {group:7} {n:>4} "
                f"{row['ungrounded_rate']:>5} {row['no_cite_rate']:>5} "
                f"{row['scope_over_rate']:>5} {row['sign_incoherent_rate']:>6} "
                f"{row['off_endpoint_rate']:>6} {row['mean_quality']:>5} "
                f"{(row['wrong_direction_rate'] if row['wrong_direction_rate'] is not None else 'NA'):>5}"
            )


if __name__ == "__main__":
    main()
