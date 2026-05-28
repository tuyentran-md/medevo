# Paper 1 Locked Batch — 2026-05-28

This file locks the first post-substrate batch intended to support MedEvo Paper 1.

## Purpose

Paper 1 is the benchmark/product paper.

Its primary claim is:

> MedEvo can measure pipeline-level epistemic drift from generated study-like outputs to guideline-like recommendations across historically anchored clinical claims.

This batch is not designed to prove CIVER value. Any gated-versus-ungated result is secondary context only.

## Lock

- Code commit: `f55282f`
- Backend: `claude-cli`
- Model: `claude-sonnet-4-6`
- Claim set: `services/worker/data/input_battery_paper1_15claim.txt`
- Ground truth: `services/worker/data/ground_truth/battery_paper1_15claim.json`
- Horizons: `2000,2012,2024`
- Engine seed: default `7` from `scripts.evaluate_shadow`
- Studies per claim per era: `4`
- Max constrained attempts per cell: `8`
- Output-match target ratio: `0.85`

## Why this subset

The 15 claims were selected from the verified 30-claim battery to preserve heterogeneity while keeping one fresh run tractable:

- stable supports: smoking, statin secondary prevention, surfactant, alteplase, colorectal screening
- reversals: alcohol, HRT, aspirin primary prevention, omega-3 primary prevention, arthroscopy, vertebroplasty, vitamin E
- stable refute / contested: Class I antiarrhythmics, obesity paradox, glucosamine

This gives Paper 1 enough structure to show three things if the run behaves:

1. measurable drift is not claim-uniform
2. drift can take different forms: unsupported output, neutral collapse, historically faithful but later-wrong propagation
3. the ecology is clinically textured rather than mono-directional

## Output contract

The run is acceptable for Paper 1 if it yields:

- `scientific=true`
- full 15-claim x 3-era artifact bundle
- claim-level free-branch drift table
- enough per-claim variation to support the failure taxonomy

The run is still publishable for Paper 1 if gated-versus-ungated is null. That endpoint is not the success criterion for this batch.
