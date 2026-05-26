"""Generate PubMed query candidates for each claim in a battery via claude-cli.

Output: data/claim_queries.json mapping claim_text → list[str] of 3-5 compact
PubMed queries. Used by app.agents.pubmed_query_candidates to override the
hardcoded CVD-only domain heuristics.
"""
from __future__ import annotations
import json
import re
import subprocess
import sys
from pathlib import Path

PROMPT_TMPL = (
    "You are a PubMed search expert. Given a clinical guideline claim, output "
    "3 to 5 short search queries that an Entrez E-utility (esearch) call would "
    "accept and that retrieve the most relevant evidence for appraising the "
    "claim. Each query must be 2-7 keywords (no full sentences). Cover the "
    "intervention, population/condition, and primary outcome. No date filters, "
    "no field tags.\n\n"
    "Respond with EXACTLY this format and nothing else:\n"
    "QUERY: <query 1>\n"
    "QUERY: <query 2>\n"
    "QUERY: <query 3>\n"
    "(optionally 4 and 5)\n\n"
    "Claim: {claim}"
)


def call_claude(prompt: str, timeout: int = 120) -> str:
    proc = subprocess.run(
        ["claude", "-p", "--model", "claude-sonnet-4-6"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI exited {proc.returncode}: {proc.stderr[:200]}")
    return proc.stdout.strip()


def parse_queries(raw: str) -> list[str]:
    queries: list[str] = []
    for line in raw.splitlines():
        m = re.match(r"\s*QUERY\s*:\s*(.+)$", line, re.IGNORECASE)
        if m:
            q = m.group(1).strip().strip('"').strip("'")
            if q:
                queries.append(q)
    return queries


def main() -> None:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/input_battery_run4_nc.txt")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/claim_queries.json")
    claims = [line.strip() for line in input_path.read_text().splitlines() if line.strip()]
    print(f"Generating queries for {len(claims)} claims → {output_path}")
    out: dict[str, list[str]] = {}
    # Resume support: load existing
    if output_path.exists():
        out = json.loads(output_path.read_text())
        print(f"  resume: {len(out)} already done")
    for i, claim in enumerate(claims, 1):
        if claim in out:
            print(f"  [{i}/{len(claims)}] cached, skip")
            continue
        print(f"  [{i}/{len(claims)}] generating ...", end=" ", flush=True)
        try:
            raw = call_claude(PROMPT_TMPL.format(claim=claim))
            queries = parse_queries(raw)
        except Exception as exc:
            print(f"FAIL {type(exc).__name__}: {exc}")
            queries = []
        if not queries:
            print("WARN no queries parsed; keeping empty")
        else:
            print(f"got {len(queries)}")
        out[claim] = queries
        # Save after each (resume-safe)
        output_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Done. {sum(1 for v in out.values() if v)} claims have ≥1 query.")


if __name__ == "__main__":
    main()
