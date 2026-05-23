# Run 2 abstract - 30-claim multi-domain battery

## Background

AI systems are increasingly able to generate biomedical hypotheses and research-like outputs. Clinical medicine, however, acts through accumulated studies, reviews, and guideline-like recommendations. MedEvo tests whether AI-generated research artifacts can propagate into downstream evidence drift, and whether a process-integrity gate can reduce that drift.

## Methods

We ran a 30-claim, six-domain simulated evidence ecology across historical horizons 2000, 2012, and 2024. AI research agents generated study-like outputs. A FREE branch retained all emitted studies, while a CONSTRAINED branch retained only CIVER/BRIM-warranted studies. Both branches were synthesized into guideline-like recommendations and compared against verified historical ground-truth trajectories.

## Results

Run 2 used MIMO-v2.5-pro and generated 1,032 LLM calls over approximately 16M tokens. FREE branch guideline distance to truth was 0.346. CONSTRAINED branch distance was 0.250. The difference, delta = +0.096, beat a 500-iteration volume-matched null.

## Interpretation

This is the first measurable guideline-level CIVER effect in MedEvo. The result suggests that process-integrity gating may reduce downstream guideline-like drift when the baseline model produces enough ungrounded research artifacts for contamination to dominate the evidence corpus.

## Limitations

The signal is preliminary: n=1 model, n=1 seed, thin margin against the null, and high design-abstention rate. MedEvo is a research instrument, not clinical decision support.
