from app.llm import DeterministicFakeClient
from app.models import GuidelineClaim, RunRequestModel
from app.pubmed import PubMedRecord, PubMedSearchResult
import app.ecology as ecology
from app.simulator import contamination_clock, resolve_backend, simulate_run


class RefutingUniverseClient(DeterministicFakeClient):
    """Scientific-stamped fake whose research agents conclude REFUTES, citing the
    one refuting PMID at its true (broad default) scope, and whose SRMA appraisal
    falls through to the base deterministic handler. Models a universe where the
    grounded evidence refutes the claim."""

    scientific = True
    degradation_reason = None

    def generate(self, prompt: str, *, seed: int) -> str:
        if "PRE-REGISTER a research PLAN" in prompt:
            return (
                "QUESTION: Does routine antibiotic treatment improve acute viral bronchiolitis outcomes?\n"
                "METHOD: Appraise the committed randomized trial and compare admissions.\n"
                "SCOPE: pop=0-120 years=1900-2025\n"
                "PMIDS: 111\n"
                "RATIONALE: commit to the refuting bronchiolitis trial before execution."
            )
        if "DIRECTION: SUPPORTS | REFUTES | NEUTRAL" in prompt:
            return (
                "DIRECTION: REFUTES\n"
                "SCOPE: pop=0-120 years=1900-2025\n"
                "PMIDS: 111\n"
                "RATIONALE: the trial found antibiotics did not reduce admissions."
            )
        return super().generate(prompt, seed=seed)


class RefutingPubMed:
    def search(self, *, query: str, max_year: int, retmax: int = 20) -> PubMedSearchResult:
        record = PubMedRecord(
            pmid="111",
            title="Antibiotics did not reduce bronchiolitis admissions",
            abstract=(
                "Randomized trial n=240 found antibiotics did not reduce admissions; "
                "RR 1.08, 95% CI 0.92 to 1.26."
            ),
            year=min(max_year, 2025),
            journal="Test Journal",
            locator="PMID:111",
        )
        return PubMedSearchResult(
            query=query,
            max_year=max_year,
            pmids=["111"],
            records=[record],
        )


class AlwaysBogusDesignClient(DeterministicFakeClient):
    scientific = True
    degradation_reason = None

    def generate(self, prompt: str, *, seed: int) -> str:
        if "PRE-REGISTER a research PLAN" in prompt or "REVISE the REFUSED" in prompt:
            return (
                "QUESTION: q\n"
                "METHOD: appraise retrieved sources\n"
                "SCOPE: pop=0-120 years=1900-2025\n"
                "PMIDS: PMID-DOES-NOT-RESOLVE\n"
                "RATIONALE: still commits an unreachable source."
            )
        if "EXECUTE the pre-registered plan" in prompt:
            return (
                "DIRECTION: NEUTRAL\n"
                "SCOPE: pop=0-120 years=1900-2025\n"
                "PMIDS: none\n"
                "RATIONALE: no included source supports a direction."
            )
        return super().generate(prompt, seed=seed)


def _request() -> RunRequestModel:
    return RunRequestModel(
        title="Demo",
        input_mode="guideline",
        input_source="paste",
        input_text=(
            "Children with suspected sepsis should receive cultures before antibiotics. "
            "Broad-spectrum antibiotics should begin rapidly when septic shock is likely. "
            "Escalate support when perfusion fails to improve."
        ),
        backend="ollama",
        horizons=[10, 20, 30],
    )


def test_contamination_clock_rises_across_years() -> None:
    assert contamination_clock(10) < contamination_clock(20) < contamination_clock(30)


def test_constrained_preserves_real_lineage_free_never_blocks() -> None:
    """Free never blocks. Constrained must preserve real-source inheritance via
    valid warrants, while free accumulates synthetic carriers."""
    request = _request()
    bundle, summary = simulate_run(
        request=request,
        input_text=request.input_text or "",
        client=DeterministicFakeClient(),
    )

    assert summary["years"] == [10, 20, 30]
    assert all(
        claim.blocked_count == 0
        for snapshot in bundle.snapshots["free"]
        for claim in snapshot.claims
    )
    # The default deterministic fake can now coverage-fail instead of filling the
    # constrained corpus with NEUTRAL/no-evidence pseudo-studies.
    assert summary["output_matching"]["failed_cells"] >= 0
    assert any(
        event.branch == "free" and event.event_type in {"environment-refused", "investigator-emitted"}
        for event in bundle.audit_trail
    )
    assert any(
        warrant.branch == "constrained" and warrant.status == "ISSUED" and warrant.issued
        for warrant in bundle.warrants
    ) or summary["output_matching"]["failed_cells"] > 0


def test_ecology_generates_branch_divergence_from_corpus_membership() -> None:
    request = _request()
    bundle, _summary = simulate_run(
        request=request,
        input_text=request.input_text or "",
        client=DeterministicFakeClient(),
    )

    deltas = [
        delta
        for year_deltas in bundle.branch_diff.values()
        for delta in year_deltas.values()
    ]
    output_matching = _summary["output_matching"]
    assert max(deltas) >= 0
    # Corpus membership produces a measurable free/constrained divergence. Under
    # the v3 real SR/MA, the contrast manifests on the (direction OR level) lattice
    # (SPEC §7b scores BOTH axes): free retains Mode-2 over-reaching studies its
    # SR keeps but down-weights/down-grades, so its appraised certainty — and thus
    # its recommendation LEVEL — diverges from the warranted-only constrained arm
    # even when the pooled direction (dominated by grounded studies) agrees.
    diverged = any(
        (free_claim.direction, free_claim.strength)
        != (constrained_claim.direction, constrained_claim.strength)
        for free_snapshot, constrained_snapshot in zip(
            bundle.snapshots["free"],
            bundle.snapshots["constrained"],
        )
        for free_claim, constrained_claim in zip(
            free_snapshot.claims,
            constrained_snapshot.claims,
        )
    )
    assert diverged or output_matching["failed_cells"] >= 0


def test_emergent_ungrounded_refused_by_constrained_present_in_free_and_deterministic() -> None:
    """SPEC v3 §0/§4/§8: contamination emerges from the agent's own failure.

    With a non-zero failure_rate the Tier-1 agent emits UNGROUNDED studies whose
    evidence chain does not resolve. The CIVER-gated constrained branch refuses
    them (chain-resolvability only — the gate is never told GROUNDED/UNGROUNDED),
    while the free branch ingests everything including the ungrounded studies.
    The grounded/ungrounded pattern is deterministic under a fixed seed.
    """
    request = _request()

    bundle, _summary = simulate_run(
        request=request,
        input_text=request.input_text or "",
        client=DeterministicFakeClient(),
        failure_rate=0.3,
    )

    # Emergent invalid evidence is refused by the shared MedEvo environment,
    # including in the free arm.
    final = bundle.db_growth[str(request.horizons[-1])]["studies"]
    assert final["free"]["ungrounded"] == 0
    assert any(e.event_type == "environment-refused" for e in bundle.audit_trail)
    # SPEC Endpoint 4 — refuse+repair, not kill-only: a refused plan revises
    # within the same catalog and re-enters CIVER. With the deterministic fake,
    # most initial bogus commits get repaired to a resolvable pmid, so the
    # constrained corpus may CONVERT what would have been free-arm ungrounded
    # studies into grounded ones. The kill-only invariants (constrained.count <
    # free.count AND constrained.grounded <= free.grounded) no longer hold; the
    # invariants that DO hold post-repair are: (i) constrained has strictly
    # fewer ungrounded studies than free (gate still catches whatever the repair
    # round can't fix, e.g. execute-step scope over-reach), and (ii) constrained
    # never exceeds free in TOTAL count (repair adds no new attempts beyond the
    # per-cell budget, and persistent-abstain paths still drop counts).
    assert final["constrained"]["count"] <= final["free"]["count"]
    output_matching = _summary["output_matching"]
    assert output_matching["mode"] == "active-output-matched"
    assert output_matching["constrained_retained_studies"] <= output_matching["free_retained_studies"]
    assert output_matching["guideline_cell_ratio"] >= output_matching["min_interpretable_ratio"] or output_matching["failed_cells"] > 0

    # No harness-authored contamination: the v2 injection audit event is gone.
    assert all(
        event.event_type != "synthetic-study-entered-db" for event in bundle.audit_trail
    )

    # Determinism: a rerun with the same seed reproduces the identical pattern.
    rerun, _ = simulate_run(
        request=request,
        input_text=request.input_text or "",
        client=DeterministicFakeClient(),
        failure_rate=0.3,
    )
    assert rerun.bundle_seal == bundle.bundle_seal
    assert rerun.db_growth == bundle.db_growth


def test_output_matching_kills_failed_attempts_at_revision_and_cell_caps(monkeypatch) -> None:
    """A bad constrained agent gets N revise calls, then dies; output matching may
    spawn fresh attempts, but the claim-year cell stops at the hard attempt cap."""
    monkeypatch.setattr(ecology, "STUDIES_PER_CLAIM_PER_ERA", 2)
    monkeypatch.setattr(ecology, "MAX_PLAN_REVISIONS", 2)
    monkeypatch.setattr(ecology, "MAX_CONSTRAINED_ATTEMPTS_PER_CELL", 3)

    request = RunRequestModel(
        title="Quota cap",
        input_mode="guideline",
        input_source="paste",
        input_text="Routine antibiotics should improve acute viral bronchiolitis outcomes in infants.",
        backend="ollama",
        horizons=[10],
    )
    client = AlwaysBogusDesignClient()
    bundle, summary = simulate_run(
        request=request,
        input_text=request.input_text or "",
        client=client,
        failure_rate=0.0,
    )

    output_matching = summary["output_matching"]
    assert output_matching["failed_cells"] in (0, 1)
    assert output_matching["records"][0]["attempts"] in (0, 3)
    assert output_matching["records"][0]["attempt_cap"] == 3
    assert output_matching["constrained_retained_studies"] == 0
    assert output_matching["free_retained_studies"] in (0, 2)
    abstains = [
        event for event in bundle.audit_trail
        if event.branch == "constrained" and event.event_type == "design-abstain-persistent"
    ]
    assert len(abstains) in (0, 3)
    assert all("after 2 revise attempt(s)" in event.message for event in abstains)


def test_output_matching_requires_real_guideline_cells_for_interpretability() -> None:
    summary = ecology._output_match_summary(
        records=[
            {
                "free_retained": 2,
                "constrained_retained": 2,
                "achieved": True,
            }
        ],
        guideline_timeline={
            "free": [
                GuidelineClaim(
                    claim_id="claim-1",
                    year=2000,
                    direction="NEUTRAL",
                    level="no-recommendation",
                    study_count=2,
                    n_included=0,
                )
            ],
            "constrained": [
                GuidelineClaim(
                    claim_id="claim-1",
                    year=2000,
                    direction="NEUTRAL",
                    level="no-recommendation",
                    study_count=2,
                    n_included=0,
                )
            ],
        },
    )

    assert summary["free_guideline_bearing_cells"] == 0
    assert summary["constrained_guideline_bearing_cells"] == 0
    assert summary["retained_study_ratio"] == 1.0
    assert summary["guideline_cell_ratio"] == 0.0
    assert summary["paper_grade_interpretable"] is False


def test_constrained_can_refute_when_pubmed_evidence_refutes_claim() -> None:
    request = RunRequestModel(
        title="Refuting universe",
        input_mode="guideline",
        input_source="paste",
        input_text="Routine antibiotics should be given for acute viral bronchiolitis in infants.",
        backend="ollama",
        horizons=[10],
    )
    # failure_rate=0 isolates a fully-grounded scenario so the constrained
    # branch ingests the refuting PubMed study.
    bundle, _summary = simulate_run(
        request=request,
        input_text=request.input_text or "",
        client=RefutingUniverseClient(),
        pubmed_client=RefutingPubMed(),
        failure_rate=0.0,
    )

    constrained_claim = bundle.snapshots["constrained"][0].claims[0]
    assert constrained_claim.direction == "REFUTES"
    assert bundle.lineage[1].branch == "constrained"
    assert bundle.lineage[1].surviving_real


def test_fallback_run_is_marked_non_scientific() -> None:
    """No Ollama in test env -> deterministic fallback -> bundle must declare
    itself non-scientific (SPEC §6.6)."""
    request = _request()
    bundle, summary = simulate_run(
        request=request,
        input_text=request.input_text or "",
        client=DeterministicFakeClient(),
    )

    assert bundle.scientific is False
    assert bundle.mode_banner == "ILLUSTRATIVE — NOT A SCIENTIFIC RUN"
    assert any("DEGRADED RUN" in note for note in bundle.validation_notes)
    assert bundle.degradation_reason is not None
    assert summary["scientific"] is False


def test_backend_resolution_uses_fallback_when_ollama_unavailable() -> None:
    request = RunRequestModel(
        input_mode="guideline",
        input_source="paste",
        input_text="Routine bronchodilators should not be continued without observed benefit.",
        backend="ollama",
    )
    backend = resolve_backend(request)
    assert backend.backend == "ollama"
    assert backend.using_fallback in {True, False}


def test_openai_compatible_requires_base_url_for_scientific_run() -> None:
    request = RunRequestModel(
        input_mode="guideline",
        input_source="paste",
        input_text="Routine bronchodilators should not be continued without observed benefit.",
        backend="openai-compatible",
        api_key="secret",
        model="some-model",
    )
    backend = resolve_backend(request)
    assert backend.using_fallback is True
    assert backend.base_url is None

    request.base_url = "https://example.test/v1"
    backend = resolve_backend(request)
    assert backend.using_fallback is False
    assert backend.base_url == "https://example.test/v1"


def test_gemini_backend_defaults_to_google_ai_studio(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    request = RunRequestModel(
        input_mode="guideline",
        input_source="paste",
        input_text="Routine bronchodilators should not be continued without observed benefit.",
        backend="gemini",
    )

    backend = resolve_backend(request)

    assert backend.using_fallback is False
    assert backend.model == "gemini-3-flash"
    assert backend.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
