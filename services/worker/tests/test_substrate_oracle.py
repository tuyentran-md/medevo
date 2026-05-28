"""Deterministic substrate oracle for the SRMA/evidence layer.

These tests encode the CORRECT behaviour of the evidence-substrate functions —
sample-size extraction, quality scoring, effect attribution, and the
effect→pooled-direction sign — independent of any LLM. They are the oracle that
the end-to-end pipeline rests on: if these are wrong, no claim result is
interpretable (see _WIKI/medevo.md 2026-05-28).

Fixtures mirror the two real records that drove the v6 claim-2 artifact:
  * REC_GENE (PMID 11177205): a 132-person gene-nutrient/LDL study that merely
    mentions alcohol and CHD in passing — NOT an alcohol→CHD effect study.
  * REC_MR (PMID 39580711): the 28278-person Mendelian-randomization alcohol
    study — the scientifically pivotal evidence, with no clean ratio in its
    abstract.
The v6 bug: REC_GENE scored quality 0.75 and contributed +0.69 SUPPORTS off a
spurious OR=2.0, while REC_MR scored 0.50 and was screened out.
"""

from app.agents import _extract_sample_size, _quality_score
from app.models import PubMedRecord, Study
from app.synthesis import (
    claim_polarity,
    direction_from_pooled_effect,
    numeric_effect_component,
    pooled_effect,
)

REC_GENE = PubMedRecord(
    pmid="11177205",
    title=(
        "Gene-nutrient interactions: dietary behaviour associated with high "
        "coronary heart disease risk affects serum LDL cholesterol in ApoE "
        "epsilon4 carriers"
    ),
    abstract=(
        "ApoE genotype influence on dietary risk factors was investigated in "
        "132 free-living individuals. Carriers showed an odds ratio of 2.0 for "
        "elevated LDL. Moderate alcohol intake was recorded as a covariate."
    ),
    year=2000,
    journal="Test",
)

REC_MR = PubMedRecord(
    pmid="39580711",
    title=(
        "A Mendelian randomization study of alcohol use and cardiometabolic "
        "disease risk in a multi-ancestry cohort"
    ),
    abstract=(
        "Observational studies link moderate alcohol consumption to reduced "
        "risk of coronary heart disease. We followed 28278 participants. "
        "Mendelian randomization suggests these associations are due to "
        "confounding; genetically predicted alcohol use was not protective."
    ),
    year=2024,
    journal="Test",
)


def test_spc04_uses_canonical_mesh_not_agent_variables(monkeypatch) -> None:
    """SpC-04 is the structural-gate surface of the 'medevo-rule-in-the-loop'
    substitution of human attribute confirmation: the EVIDENCE node's outcome is
    pinned from canonical MeSH descriptors PubMed attached to the cited records
    AND matched against the claim outcome via MeSH tree hierarchy (standard SR
    `[MeSH]` explosion: descendant or equal = match; sibling/ancestor = no
    match). The agent's free-form `evidence_attrs.variables` label is therefore
    irrelevant — whatever it writes, SpC-04 reads canonical PubMed MeSH and
    canonical NLM tree numbers.

    Generic across claims (CHD / MI / MetS / cancer subtypes all live in MeSH);
    no per-vocabulary acronym/synonym band-aid needed."""
    import app.mesh as mesh_mod
    from app.ecology import (
        _reachable_lookup_from_catalog,
        _spc04_evidence_measures_claim_outcome,
    )
    from app.models import ClaimGraph, PIRNodeAttributes, PubMedRecord, ResearchPlan

    # Stub MeSH client — hardcoded tree numbers for the descriptors used in
    # this test. No network in unit tests; the live Entrez path is exercised
    # only by integration scripts.
    canonical_trees = {
        "coronary heart disease": ["C14.280.647.250"],
        "coronary disease": ["C14.280.647.250"],
        "metabolic syndrome": ["C18.452.394.968.500.570"],
        "alcohol drinking": ["F01.145.317.269.500", "G07.203.650.353.500"],
        "risk factors": [],
        "cohort studies": [],
    }

    class _StubClient:
        def tree_numbers(self, descriptor: str) -> list[str]:
            return list(canonical_trees.get((descriptor or "").lower().strip(), []))

    monkeypatch.setattr(mesh_mod, "_DEFAULT_CLIENT", _StubClient())

    claim = (
        "Light to moderate alcohol consumption reduces risk of coronary heart "
        "disease by elevating HDL and lowering platelet aggregation."
    )
    cg = ClaimGraph(claim_id="c", claim_text=claim, nodes=[], edges=[])
    chd_record = PubMedRecord(
        pmid="1001", title="A", abstract="A", year=2000,
        mesh_terms=["alcohol drinking", "coronary disease", "risk factors"],
    )
    mets_record = PubMedRecord(
        pmid="2002", title="B", abstract="B", year=2012,
        mesh_terms=["alcohol drinking", "metabolic syndrome", "cohort studies"],
    )
    catalog = [chd_record, mets_record]
    lookup = _reachable_lookup_from_catalog(catalog)

    def _plan(pmids, ev_label):
        # Agent's evidence variable label is DELIBERATELY misleading to prove
        # SpC-04 ignores it and reads canonical MeSH from the cited record.
        return ResearchPlan(
            plan_id="p", claim_id="c", year=2000, question="q", method="m",
            committed_pmids=pmids,
            evidence_attrs=PIRNodeAttributes(variables=[ev_label]),
        )

    # On-endpoint (record indexed at 'coronary disease' = exact tree match).
    ok, _ = _spc04_evidence_measures_claim_outcome(_plan(["1001"], "junk"), cg, lookup)
    assert ok
    # Off-endpoint (MetS indexed in a different MeSH subtree), even if the agent
    # labels its variable "coronary-heart-disease" -> BLOCK on canonical MeSH.
    blocked, reason = _spc04_evidence_measures_claim_outcome(
        _plan(["2002"], "coronary-heart-disease"), cg, lookup
    )
    assert not blocked
    assert "metabolic" in reason.lower()
    # No MeSH attached (recent unindexed article) -> vacuous (permissive),
    # defers to the LLM relevance screen.
    bare = PubMedRecord(pmid="3003", title="X", abstract="X", year=2000)
    ok_vacuous, _ = _spc04_evidence_measures_claim_outcome(
        _plan(["3003"], "anything"),
        cg,
        _reachable_lookup_from_catalog([bare]),
    )
    assert ok_vacuous


def test_mesh_hierarchy_match_rules() -> None:
    """Standard SR explosion: equal/descendant matches, sibling/ancestor doesn't."""
    from app.mesh import mesh_hierarchy_match

    chd = ["C14.280.647.250"]
    # exact descendant (Coronary Artery Disease is C14.280.647.250.250)
    assert mesh_hierarchy_match(chd, {"C14.280.647.250.250"})
    # exact match
    assert mesh_hierarchy_match(chd, {"C14.280.647.250"})
    # sibling (Myocardial Infarction C14.280.647.500) -> NO
    assert not mesh_hierarchy_match(chd, {"C14.280.647.500"})
    # ancestor (Cardiovascular Diseases C14) -> NO (broader, not explosion target)
    assert not mesh_hierarchy_match(chd, {"C14"})
    # unrelated (Metabolic Syndrome C18.452.*) -> NO
    assert not mesh_hierarchy_match(chd, {"C18.452.394.968.500.570"})


def test_claim_polarity_ignores_mechanism_clauses() -> None:
    """Polarity must anchor to the risk/incidence noun, not mechanism verbs.
    The v7 rerun exposed this: 'reduces risk of CHD by ELEVATING HDL and
    LOWERING platelet aggregation' returned 0 (elevating/lowering confused it),
    so the sign-coherence gate silently disabled itself and a harmful OR pooled
    as SUPPORTS again."""
    alcohol = (
        "Light to moderate alcohol consumption reduces risk of coronary heart "
        "disease by elevating high-density lipoprotein cholesterol and lowering "
        "platelet aggregation."
    )
    assert claim_polarity(alcohol) == -1
    assert claim_polarity("Smoking increases the risk of coronary heart disease.") == 1
    assert claim_polarity("Statins lower cardiovascular mortality.") == -1
    # HRT: 'all-cause mortality' must NOT trigger the increase verb 'cause'
    # (v11 polarity exposed: bare 'cause' was matching, making the parser flag
    # the claim ambiguous; verb-form-only `caus(es|ed|ing)` fixed it).
    hrt = (
        "Menopausal hormone therapy with combined estrogen and progestin "
        "reduces risk of coronary heart disease and all-cause mortality."
    )
    assert claim_polarity(hrt) == -1
    # Multi-word window: statin claim has 5 words between 'reduces' and
    # 'mortality' — the (0,6) forward window must reach it.
    assert claim_polarity(
        "Statin therapy reduces major adverse cardiovascular events and all-cause mortality."
    ) == -1
    # No risk framing -> unknown -> fall back to the model label (safe).
    assert claim_polarity("Bronchodilators should not be continued routinely.") == 0


def test_sample_size_extracted_for_both() -> None:
    assert _extract_sample_size(REC_GENE) == 132
    assert _extract_sample_size(REC_MR) == 28278


def test_quality_increases_with_sample_size() -> None:
    """A 28k-person study must not score BELOW a 132-person one. v6 had the
    inversion (0.50 < 0.75) because _quality_score ignored n and rewarded a
    spurious effect-regex hit."""
    q_gene = _quality_score(record=REC_GENE, numeric=True)
    q_mr = _quality_score(record=REC_MR, numeric=False)
    assert q_mr > q_gene, f"MR(n=28278)={q_mr} should exceed gene(n=132)={q_gene}"


def _study(direction, effect, *, n, q=0.6) -> Study:
    return Study(
        id=f"s-{direction}-{effect}",
        claim_id="claim-1",
        year=2024,
        direction=direction,
        effect_point=effect,
        n=n,
        quality=q,
        provenance="GROUNDED",
        numeric=effect is not None,
        rationale="",
        pmids=["12345"],
    )


def test_harmful_ratio_labeled_supports_does_not_pool_as_support() -> None:
    """A study whose extracted ratio is HARMFUL (OR/HR > 1, i.e. raises risk)
    must not contribute a SUPPORTS-direction magnitude to a 'reduces risk'
    claim just because the model labelled it SUPPORTS. v6 let OR=2.0 (harmful)
    count as +0.69 toward SUPPORTS. Coherent behaviour: the effect's own
    direction governs the sign (or the incoherent study is neutralised), so it
    must NOT pool to a confident SUPPORTS."""
    pooled = pooled_effect([_study("SUPPORTS", 2.0, n=132)], claim_polarity=-1)
    assert direction_from_pooled_effect(pooled) != "SUPPORTS", (
        f"harmful OR=2.0 pooled to {pooled} -> "
        f"{direction_from_pooled_effect(pooled)} (should not be SUPPORTS)"
    )


def test_protective_ratio_supports_reduces_risk_claim() -> None:
    """A protective ratio (HR=0.6) correctly labelled SUPPORTS for a reduces-
    risk claim must pool positive (SUPPORTS). This is the coherent case and
    must keep working after the sign fix."""
    pooled = pooled_effect([_study("SUPPORTS", 0.6, n=5000)], claim_polarity=-1)
    assert direction_from_pooled_effect(pooled) == "SUPPORTS"


def test_quality_floor_is_deterministic_not_llm() -> None:
    """The quality floor is the harness's call, not the LLM screen's. Even if the
    LLM screen 'includes' a below-floor study, synthesis must drop it (v6 had the
    LLM include/exclude the same study inconsistently across arms)."""
    from app.synthesis import (
        RobAssessment,
        ScreeningDecision,
        SrReview,
        synthesize_guideline_claim,
    )

    good = _study("REFUTES", None, n=28278, q=0.62)  # above floor, relevant
    junk = Study(
        id="junk",
        claim_id="claim-1",
        year=2024,
        direction="SUPPORTS",
        effect_point=None,
        n=None,
        quality=0.10,  # below 0.30 floor
        provenance="UNGROUNDED",
        numeric=False,
        rationale="",
        pmids=[],
    )
    llm_screen = SrReview(
        screening=[
            ScreeningDecision(study_id="junk", included=True, reason="llm kept it"),
            ScreeningDecision(study_id=good.id, included=True, reason="relevant"),
        ],
        rob=RobAssessment(),
        included_ids=["junk", good.id],
        n_included=2,
        n_excluded=0,
    )
    out = synthesize_guideline_claim(
        claim_id="claim-1",
        year=2024,
        studies=[good, junk],
        screening=llm_screen,
        claim_text="Alcohol reduces risk of coronary heart disease.",
    )
    assert out.n_included == 1  # junk dropped by deterministic floor
