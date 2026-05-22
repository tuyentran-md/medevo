# MedEvo

### Is AI quietly rewriting clinical guidelines — and can a process-integrity gate stop it?

![How MedEvo tests whether a process-integrity gate keeps a guideline on course as AI agents do the science: the free arm drifts, the gated arm tracks the real reversal.](docs/reversal.svg)

MedEvo is a **simulated scientific ecology**. AI agents do real research over real
data and literature; their work accumulates into the corpus a guideline is
synthesized from; and we watch whether the guideline **drifts** — and whether
CIVER/BRIM process governance keeps invalid research processes from becoming
warranted evidence.

---

## The problem

Evidence-based medicine assumes the literature is a faithful record of what was
actually studied. AI now writes, summarizes, and synthesizes that literature at
scale. When poorly-grounded findings launder through systematic reviews into
authoritative recommendations, a guideline can shift in **direction** or
**strength** — silently. MedEvo makes that moment visible, and tests a defense.

## The pipeline — stage by stage

```mermaid
flowchart TD
    S1["① INPUT — a guideline becomes atomic claims<br/>(each: direction + strength)"]
    S2["② ADVANCE to simulated era T<br/>PubMed / data date-cut to year T"]
    S3["③ TIER 1 · research agents produce studies<br/>Group A on real data · Group B on real literature"]
    S4{"④ TIER 2 · CIVER/BRIM<br/>plan/PIR · monitor · ECW"}
    S6["⑤ TIER 3 · accumulating corpus<br/>free = everything · constrained = warranted only"]
    S7["⑥ TIER 4 · SR/MA synthesis → guideline<br/>(direction + GRADE-style strength)"]
    S8["⑦ COMPARE natural all-output vs ECW-compliant<br/>+ sealed replay"]
    S1 --> S2 --> S3 --> S4
    S4 -->|warrant issued| S6
    S4 -.->|refused — free keeps it anyway| S6
    S6 --> S7
    S7 -->|feeds back as prior · advance era| S2
    S7 --> S8
```

We replay a **real historical reversal** to validate the instrument: hormone
therapy (HRT) for chronic-disease prevention — recommended *for* prevention
before 2002, reversed by the WHI trials, USPSTF grade D *against* ever since
(the figure at the top). A faithful ecology should re-live that flip from each
era's literature; the gate should hold the guideline on course while an ungated
arm drifts.

## The simulated researchers (Tier 1)

Agents do research the way people do — and **fail the way fallible researchers
do.** That failure, not an injected fake, is the contamination.

```mermaid
flowchart TD
    Q["a claim to investigate, at era T"]
    Q --> GA["GROUP A · empirical"]
    Q --> GB["GROUP B · evidence synthesis"]
    GA --> GA1["design analysis → load a real NHANES slice<br/>→ run statistics in a sandbox → interpret"]
    GB --> GB1["search PubMed ≤ T → appraise the real papers"]
    GA1 --> OK["✅ GROUNDED study<br/>resolvable provenance, scope-bounded"]
    GB1 --> OK
    GA1 --> BAD["⚠️ FAILURE — can't ground it / over-reaches scope<br/>→ UNGROUNDED study (emergent, not injected)"]
    GB1 --> BAD
    OK --> G([to the CIVER gate])
    BAD --> G
```

The failure *rate* is anchored to a measured quantity (how often LLMs produce
structurally-valid but unfaithful claims), never a tuned dial.

## The synthesis model (Tier 4 · SR/MA)

Synthesis agents read **only** the accumulated corpus — never re-querying the
world — exactly as a real guideline panel works from the published record.

```mermaid
flowchart LR
    DB["Tier-3 corpus<br/>(this branch only)"] --> AP["appraise each study<br/>quality · sample size · risk of bias"]
    AP --> PL["pool the effect<br/>weighted · heterogeneity"]
    PL --> CE["GRADE-style certainty"]
    CE --> OUT["guideline claim<br/>direction + strength level"]
```

## How drift emerges — and how the gate answers it

```mermaid
flowchart TD
    subgraph FREE["🔴 FREE arm — no gate"]
      direction TB
      F1["ungrounded studies accumulate in the corpus"] --> F2["the pooled estimate shifts"] --> F3["the guideline drifts off the real trajectory"]
    end
    subgraph CON["🟢 CONSTRAINED arm — CIVER"]
      direction TB
      C1["ungrounded work is refused at the gate"] --> C2["the inherited corpus stays grounded"] --> C3["the guideline holds course"]
    end
```

The gate judges only provenance, scope, and chain integrity — never a
"this one is fake" label — so it catches fabrication and over-reach but not an
honest analysis that is simply wrong (that drifts both arms equally and cancels).
The headline is the leakage- and competence-cancelled **gap** between the arms,
with confidence intervals on both axes, benchmarked against the real USPSTF
trajectory and checked against volume-matched and random-gate controls. MedEvo's
claim is **auditable corruption-resistance** — never "AI that writes more correct
guidelines."

## Status

The engine is built end-to-end, 90 tests green, and now has two scored runs
on its multi-directional CVD instrument and a 30-claim multi-domain battery.

### Run 1 — official baseline, 2026-05-22 (Sonnet 4.6, 4 CVD claims)

`services/worker/data/artifacts/shadow-20260522T072303Z/` · 500 s wall · 163 LLM
calls (144 cache hits / 19 fresh / 19 writes). Drift signal solid: mean distance
to truth **0.281**; per-claim drift highest on alcohol cardioprotection (0.625,
the documented Mendelian-randomization reversal) and the obesity paradox (0.375,
honest analysis of observational evidence that itself encodes the methodological
flaw). Study-level CIVER discriminates on 4 / 4 metrics, FPR = 0, FNR = 0.57.
Guideline-level CIVER delta = 0 (tie) — the strong baseline left no junk for the
gate to remove, so the premise of Paper 3 was recorded as untestable in this regime.

### Run 2 — 30-claim multi-domain battery, 2026-05-22 (MIMO-v2.5-pro)

Full write-up: [`docs/runs/RUN_2_30CLAIM_MIMO.md`](docs/runs/RUN_2_30CLAIM_MIMO.md).

`services/worker/data/artifacts/shadow-20260522T131131Z/` (raw) + `shadow-20260522T174426Z/`
(analyzer re-pass after ground-truth re-keying) · 4 h 33 m wall · 1 032 LLM calls
on Xiaomi MIMO-v2.5-pro (1 T params, OpenAI-compat at
`token-plan-sgp.xiaomimimo.com/v1`). 30 claims across six medical domains with
verified primary-source trajectories and eight era-reversals.

| Endpoint | Run 1 (Sonnet 4.6, 4 claims) | Run 2 (MIMO-v2.5-pro, 30 claims) |
|---|---|---|
| **E1** drift to truth | 0.281 | **0.346** |
| **E2** study-level pass rate | 42 / 48 ≈ 87% | 33 / 180 ≈ 18% |
| **E2** signal (passed vs failed) | 4 / 4 metrics, FPR 0, FNR 0.57 | 4 / 4 metrics, larger margins |
| **E3** all_to_truth | 0.156 (post-fix) | 0.346 |
| **E3** warranted_to_truth | 0.156 (tie) | **0.250** |
| **E3** delta | 0 | **+0.096** |
| **E3** vs volume-matched null (500 iter) | n/a | **CIVER beats** (warranted 0.250 < ci_low 0.263) |

Run 2 produced the first measurable **guideline-level** CIVER effect. The
mechanism is now legible: AI-driven drift propagates from studies to guidelines
*when the baseline model is weak enough* that ungrounded studies dominate the
corpus, and CIVER catches the propagation by filtering at study level before
SR/MA inherits the junk. Run 1's reading was correct for its regime; the full
claim of Paper 3 must be A0-conditional.

This is a research instrument, not clinical decision support.

## Run

```bash
cd services/worker && python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m scripts.evaluate --topic hrt \
  --backend openai-compatible --base-url https://openrouter.ai/api/v1 \
  --model deepseek/deepseek-v4 --api-key-env OPENROUTER_API_KEY \
  --horizons 2000,2010,2020
```

Or spend a Claude subscription as the model (no API key — uses the local `claude` CLI):

```bash
python -m scripts.evaluate --topic hrt --backend claude-cli \
  --model claude-sonnet-4-6 --horizons 2000,2010,2020
```

Cheap lane via Google AI Studio Gemini API (uses `GEMINI_API_KEY`, default model
`gemini-3-flash`, default base URL `https://generativelanguage.googleapis.com/v1beta/openai`):

```bash
python -m scripts.evaluate --topic cvd --backend gemini --max-calls 500
```

Run-ops guardrails before spending model calls:

```bash
cd services/worker
./.venv/bin/python -m scripts.evaluate --topic cvd --backend claude-cli --dry-run
./.venv/bin/python -m scripts.evaluate --topic cvd --backend claude-cli --max-calls 500
```

Live LLM calls are cached by default in `services/worker/data/llm_cache` (gitignored).
Set `MEDEVO_LLM_CACHE_ONLY=1` to replay only cached responses, or
`MEDEVO_LLM_CACHE=0` to force fresh calls.

Static replay site: `npm run build:static --prefix apps/web`.
Tests: `cd services/worker && ./.venv/bin/pytest`.

## Author

**Tuyen Tran, MD** — pediatric surgeon working at the intersection of
evidence-based medicine, AI, and low-resource clinical settings. ORCID
[0009-0003-0535-6225](https://orcid.org/0009-0003-0535-6225).

A research instrument, not clinical decision support. Nothing it outputs should
inform the care of an actual patient.
