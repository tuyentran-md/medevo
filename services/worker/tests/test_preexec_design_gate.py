"""Tests for the v4 per-arm asymmetric Tier-1 + multi-step SR/MA (this session).

Covers:
- CONSTRAINED arm runs a separable DESIGN call -> pre-execution CIVER gate
  (Article I, prove integrity BEFORE the process runs) -> EXECUTE call. A refused
  design never executes and no study enters the constrained corpus.
- FREE/natural arm also records design + execute so shadow CIVER/BRIM can audit
  the research process; only the constrained arm enforces the gate.
- The SR/MA is a real multi-step process: SCREEN / RISK-OF-BIAS / SYNTHESIZE are
  each their own LLM call.
- Article II execution-deviation (citing outside the registered plan) is caught
  as a WARN audit event.
- The pre-execution gate is BLIND to the harness ground-truth provenance label.
"""

from __future__ import annotations

import inspect

from app.agents import ResearchAgent, SrmaAgent, parse_research_plan
from app.ecology import (
    ClaimSeed,
    admit_research_plan,
    _reachable_lookup_from_catalog,
)
from app.llm import ModelDescriptor
from app.models import EvidenceScope, PubMedRecord, Study
from app.pubmed import PubMedSearchResult
from app.simulator import build_claim_graph


# --------------------------------------------------------------------------- #
# A scripted LLM that ROUTES by prompt sentinel, so a single instance can drive
# the constrained arm's design + execute (which the old single-response scripted
# client could not — both calls would return the same string).
# --------------------------------------------------------------------------- #
class RoutingLLM:
    scientific = True
    degradation_reason = None

    def __init__(self, *, design: str, execute: str) -> None:
        self._design = design
        self._execute = execute
        self.labels: list[str] = []
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, seed: int) -> str:
        self.prompts.append(prompt)
        if "PRE-REGISTER a research PLAN" in prompt:
            return self._design
        if "EXECUTE the pre-registered plan" in prompt:
            return self._execute
        return "DIRECTION: NEUTRAL\nSCOPE: pop=0-120 years=1900-2025\nPMIDS: none\nRATIONALE: x."

    def describe(self) -> ModelDescriptor:
        return ModelDescriptor(name="routing", digest="test")


class ScopedPubMed:
    SCOPE = EvidenceScope(population_low=40, population_high=60, year_start=2015, year_end=2018)

    def search(self, *, query: str, max_year: int, retmax: int = 20) -> PubMedSearchResult:
        record = PubMedRecord(
            pmid="111",
            title="Antibiotics did not reduce bronchiolitis admissions",
            abstract="Randomized trial n=240 found antibiotics did not reduce admissions; RR 1.08, 95% CI 0.92 to 1.26.",
            year=min(max_year, 2018),
            journal="Test Journal",
            locator="PMID:111",
            scope=self.SCOPE.model_copy(deep=True),
        )
        return PubMedSearchResult(query=query, max_year=max_year, pmids=["111"], records=[record])


CLAIM = ClaimSeed("claim-1", "Some clinical claim about a treatment.", "moderate")


def _agent(llm: RoutingLLM) -> ResearchAgent:
    return ResearchAgent(pubmed=ScopedPubMed(), llm=llm, retmax=5)


# --- 1. design-refused blocks execution -------------------------------------


def test_refused_design_never_executes() -> None:
    # The plan commits a PMID NOT in the retrieved catalog -> pre-execution gate
    # refuses -> the agent must not execute. We verify the gate verdict + that no
    # EXECUTE call would be reached.
    llm = RoutingLLM(
        design=(
            "QUESTION: q\nMETHOD: appraise sources\n"
            "SCOPE: pop=40-60 years=2015-2018\nPMIDS: 99999\nRATIONALE: bogus commit."
        ),
        execute="DIRECTION: SUPPORTS\nSCOPE: pop=40-60 years=2015-2018\nPMIDS: 111\nRATIONALE: x.",
    )
    agent = _agent(llm)
    plan, catalog = agent.run_design(
        claim_id="claim-1", claim_text=CLAIM.text, simulated_year=2020
    )
    admitted, reasons = admit_research_plan(
        plan=plan,
        claim_graph=build_claim_graph(CLAIM),
        reachable_lookup=_reachable_lookup_from_catalog(catalog),
    )
    assert admitted is False
    assert any("do not resolve" in r for r in reasons)
    # Only the DESIGN call happened; no EXECUTE call was made.
    assert len([p for p in llm.prompts if "EXECUTE the pre-registered plan" in p]) == 0


def test_incoherent_design_is_refused() -> None:
    # No METHOD line -> parse_ok False -> incoherent design refused pre-execution.
    plan = parse_research_plan(
        "QUESTION: q\nPMIDS: 111\nSCOPE: pop=40-60 years=2015-2018\nRATIONALE: r.",
        plan_id="p", claim_id="claim-1", year=2020, claim_text=CLAIM.text,
    )
    assert plan.parse_ok is False
    admitted, reasons = admit_research_plan(
        plan=plan, claim_graph=build_claim_graph(CLAIM), reachable_lookup={"111": object()}  # type: ignore[dict-item]
    )
    assert admitted is False
    assert any("incoherent" in r for r in reasons)


def test_admitted_design_executes_and_grounds() -> None:
    llm = RoutingLLM(
        design=(
            "QUESTION: q\nMETHOD: appraise sources\n"
            "SCOPE: pop=40-60 years=2015-2018\nPMIDS: 111\nRATIONALE: commit 111."
        ),
        execute="DIRECTION: REFUTES\nSCOPE: pop=40-60 years=2015-2018\nPMIDS: 111\nRATIONALE: no benefit.",
    )
    agent = _agent(llm)
    plan, catalog = agent.run_design(
        claim_id="claim-1", claim_text=CLAIM.text, simulated_year=2020
    )
    admitted, _ = admit_research_plan(
        plan=plan,
        claim_graph=build_claim_graph(CLAIM),
        reachable_lookup=_reachable_lookup_from_catalog(catalog),
    )
    assert admitted is True
    study = agent.run_execute(plan=plan, catalog=catalog, claim_text=CLAIM.text)
    assert study.provenance == "GROUNDED"
    assert study.direction == "REFUTES"
    assert study.plan_id == plan.plan_id


def test_overreaching_design_scope_is_refused_before_execution() -> None:
    # The plan commits a real source but over-widens the population/timeframe it
    # claims the source can support. This must fail before EXECUTE, not only after
    # a study output exists.
    llm = RoutingLLM(
        design=(
            "QUESTION: q\nMETHOD: appraise sources\n"
            "SCOPE: pop=0-100 years=1990-2025\nPMIDS: 111\nRATIONALE: over-wide commit."
        ),
        execute="DIRECTION: REFUTES\nSCOPE: pop=40-60 years=2015-2018\nPMIDS: 111\nRATIONALE: x.",
    )
    agent = _agent(llm)
    plan, catalog = agent.run_design(
        claim_id="claim-1", claim_text=CLAIM.text, simulated_year=2020
    )
    admitted, reasons = admit_research_plan(
        plan=plan,
        claim_graph=build_claim_graph(CLAIM),
        reachable_lookup=_reachable_lookup_from_catalog(catalog),
    )
    assert admitted is False
    assert any("scope clause" in r for r in reasons)
    assert len([p for p in llm.prompts if "EXECUTE the pre-registered plan" in p]) == 0


# --- 2. free records process; constrained enforces process ------------------


def test_free_records_plan_execute_but_does_not_enforce_gate() -> None:
    free_llm = RoutingLLM(
        design=(
            "QUESTION: q\nMETHOD: appraise\n"
            "SCOPE: pop=40-60 years=2015-2018\nPMIDS: 111\nRATIONALE: r."
        ),
        execute="DIRECTION: REFUTES\nSCOPE: pop=40-60 years=2015-2018\nPMIDS: 111\nRATIONALE: r.",
    )
    free_agent = _agent(free_llm)
    plan, catalog = free_agent.run_design(
        claim_id="claim-1", claim_text=CLAIM.text, simulated_year=2020
    )
    study = free_agent.run_execute(plan=plan, catalog=catalog, claim_text=CLAIM.text)
    assert study.research_plan is not None
    assert len([p for p in free_llm.prompts if "PRE-REGISTER a research PLAN" in p]) == 1
    assert len([p for p in free_llm.prompts if "EXECUTE the pre-registered plan" in p]) == 1

    con_llm = RoutingLLM(
        design=(
            "QUESTION: q\nMETHOD: appraise\n"
            "SCOPE: pop=40-60 years=2015-2018\nPMIDS: 111\nRATIONALE: r."
        ),
        execute="DIRECTION: REFUTES\nSCOPE: pop=40-60 years=2015-2018\nPMIDS: 111\nRATIONALE: r.",
    )
    con_agent = _agent(con_llm)
    plan, catalog = con_agent.run_design(
        claim_id="claim-1", claim_text=CLAIM.text, simulated_year=2020
    )
    con_agent.run_execute(plan=plan, catalog=catalog, claim_text=CLAIM.text)
    assert len([p for p in con_llm.prompts if "PRE-REGISTER a research PLAN" in p]) == 1
    assert len([p for p in con_llm.prompts if "EXECUTE the pre-registered plan" in p]) == 1


# --- 3. SR/MA makes per-step SCREEN / ROB / SYNTHESIZE LLM calls ------------


class _StaticStudyReader:
    def __init__(self, studies: list[Study]) -> None:
        self._studies = studies

    def list_studies(self, *, run_id, branch, claim_id, up_to_year):  # noqa: ANN001
        return list(self._studies)


def test_srma_makes_three_per_step_calls() -> None:
    labels: list[str] = []

    def invoke(label: str, prompt: str, seed: int) -> str:
        labels.append(label)
        if "SCREEN each study for inclusion" in prompt:
            ids = '"study_id": "s-0"'  # minimal; parser falls back if needed
            return (
                '{"screening": [{"study_id": "s-0", "include": true, "reason": "ok"}, '
                '{"study_id": "s-1", "include": true, "reason": "ok"}, '
                '{"study_id": "s-2", "include": true, "reason": "ok"}]}'
            )
        if "grade the RISK OF BIAS" in prompt or "SYNTHESIZE the appraised body" in prompt:
            return '{"study_appraisals": [], "certainty_adjustment": 0.0, "summary": "x"}'
        return "{}"

    studies = [
        Study(
            id=f"s-{i}", claim_id="claim-1", year=2020, direction="SUPPORTS",
            effect_point=0.7, effect_ci=(0.6, 0.82), n=400, quality=0.9,
            provenance="GROUNDED", pmids=[f"s-{i}"], numeric=True, rationale="r",
            claimed_scope=EvidenceScope(population_low=40, population_high=60, year_start=2015, year_end=2018),
            source_scope=EvidenceScope(population_low=40, population_high=60, year_start=2015, year_end=2018),
            output_hash=f"h-{i}",
        )
        for i in range(3)
    ]
    agent = SrmaAgent(study_reader=_StaticStudyReader(studies), invoke_model=invoke)
    agent.run(run_id="r", branch="constrained", claim_id="claim-1", claim_text=CLAIM.text, year=2020)

    assert any(lbl.startswith("srma-screen/") for lbl in labels)
    assert any(lbl.startswith("srma-rob/") for lbl in labels)
    assert any(lbl.startswith("srma-synth/") for lbl in labels)
    assert len(labels) == 3


# --- 4. Article II execution deviation (cite outside committed plan) --------


def test_execution_outside_plan_is_a_deviation() -> None:
    # Catalog has two records; the plan commits ONLY 111, but execution cites 222
    # too -> Article II deviation (the harness flags WARN; see ecology loop).
    class TwoRecordPubMed:
        def search(self, *, query, max_year, retmax=20):  # noqa: ANN001
            recs = [
                PubMedRecord(pmid="111", title="A", abstract="n=240 RR 1.08, 95% CI 0.92 to 1.26.", year=2016,
                             scope=EvidenceScope(population_low=40, population_high=60, year_start=2015, year_end=2018)),
                PubMedRecord(pmid="222", title="B", abstract="n=300 RR 0.80, 95% CI 0.70 to 0.91.", year=2017,
                             scope=EvidenceScope(population_low=40, population_high=60, year_start=2015, year_end=2018)),
            ]
            return PubMedSearchResult(query=query, max_year=max_year, pmids=["111", "222"], records=recs)

    llm = RoutingLLM(
        design=(
            "QUESTION: q\nMETHOD: appraise\n"
            "SCOPE: pop=40-60 years=2015-2018\nPMIDS: 111\nRATIONALE: commit only 111."
        ),
        execute="DIRECTION: REFUTES\nSCOPE: pop=40-60 years=2015-2018\nPMIDS: 111, 222\nRATIONALE: drifted.",
    )
    agent = ResearchAgent(pubmed=TwoRecordPubMed(), llm=llm, retmax=5)
    plan, catalog = agent.run_design(claim_id="claim-1", claim_text=CLAIM.text, simulated_year=2020)
    assert plan.committed_pmids == ["111"]
    study = agent.run_execute(plan=plan, catalog=catalog, claim_text=CLAIM.text)
    out_of_plan = [p for p in study.pmids if p not in set(plan.committed_pmids)]
    assert out_of_plan == ["222"]  # the ecology loop turns this into a WARN event


# --- 5. pre-execution gate blindness (signature defense) --------------------


def test_admit_research_plan_is_blind_to_provenance_label() -> None:
    sig = inspect.signature(admit_research_plan)
    assert "provenance" not in sig.parameters
    assert "failure_mode" not in sig.parameters
    assert "study" not in sig.parameters


# --- 6. end-to-end: the new gate/monitor events appear and chain verifies ---


def test_design_and_deviation_events_present_end_to_end() -> None:
    """A full run emits the pre-execution design events on the constrained arm and
    keeps the hash chain verifiable. design-refused / execution-deviated only ever
    reference the constrained branch (free runs no gate / no plan)."""
    from app.llm import DeterministicFakeClient
    from app.models import RunRequestModel
    from app.simulator import simulate_run
    from app.ecology import verify_audit_chain

    request = RunRequestModel(
        title="e2e",
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
    bundle, _summary = simulate_run(
        request=request,
        input_text=request.input_text or "",
        client=DeterministicFakeClient(),
        failure_rate=0.45,
    )
    event_types = {e.event_type for e in bundle.audit_trail}
    # After the SPEC Endpoint 4 repair loop landed, the final constrained-arm
    # design outcomes are `design-repaired` (repair succeeded) or
    # `design-abstain-persistent` (all MAX_PLAN_REVISIONS exhausted). The legacy
    # `design-refused` event type is no longer emitted as a final outcome.
    assert "design-repaired" in event_types
    assert "execution-deviated" in event_types
    assert verify_audit_chain(bundle.audit_trail)
    # design / execution phase events are recorded for both branches; only
    # constrained enters the repair loop / abstains.
    assert {e.branch for e in bundle.audit_trail if e.phase == "design"} == {
        "free",
        "constrained",
    }
    # A `design-repaired` event records a constrained-arm success of refuse+repair.
    repaired = [e for e in bundle.audit_trail if e.event_type == "design-repaired"]
    assert repaired and all(e.severity == "info" and e.branch == "constrained" for e in repaired)
    # A persistent abstain (if any) must be a constrained-arm block-severity event.
    abstain = [e for e in bundle.audit_trail if e.event_type == "design-abstain-persistent"]
    assert all(e.severity == "block" and e.branch == "constrained" for e in abstain)


# --- 7. SPEC Endpoint 4: refuse + repair loop -------------------------------


def test_repair_loop_admits_revised_plan_after_initial_refusal() -> None:
    """Constrained-arm repair loop (SPEC Endpoint 4): initial design commits a
    fabricated PMID → CIVER refuses → agent revises to commit a resolvable PMID
    → CIVER admits. Verifies the loop runs, that the revised plan executes, and
    that no kill-only abstain happens when repair is possible."""
    from app.agents import _attempt_seed  # noqa: F401  (sanity-check import works)

    class RepairRoutingLLM:
        scientific = True
        degradation_reason = None

        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, prompt: str, *, seed: int) -> str:
            self.prompts.append(prompt)
            if "PRE-REGISTER a research PLAN" in prompt:
                # Bogus initial commit -> refused by pre-execution gate.
                return (
                    "QUESTION: q\nMETHOD: appraise sources\n"
                    "SCOPE: pop=40-60 years=2015-2018\n"
                    "PMIDS: PMID-FAKE-1\nRATIONALE: bogus."
                )
            if "REVISE the REFUSED research PLAN" in prompt:
                # Repair: commit the real source at source scope -> admitted.
                return (
                    "QUESTION: q\nMETHOD: appraise the committed source\n"
                    "SCOPE: pop=40-60 years=2015-2018\n"
                    "PMIDS: 111\nRATIONALE: revised to commit resolvable 111."
                )
            return "DIRECTION: REFUTES\nSCOPE: pop=40-60 years=2015-2018\nPMIDS: 111\nRATIONALE: x."

        def describe(self) -> ModelDescriptor:
            return ModelDescriptor(name="repair-routing", digest="test")

    llm = RepairRoutingLLM()
    agent = ResearchAgent(pubmed=ScopedPubMed(), llm=llm, retmax=5)
    plan, catalog = agent.run_design(
        claim_id="claim-1", claim_text=CLAIM.text, simulated_year=2020
    )
    reachable = _reachable_lookup_from_catalog(catalog)
    admitted, reasons = admit_research_plan(
        plan=plan, claim_graph=build_claim_graph(CLAIM), reachable_lookup=reachable
    )
    assert admitted is False
    revised = agent.run_revise(
        prior_plan=plan,
        refusal_reasons=reasons,
        catalog=catalog,
        claim_text=CLAIM.text,
        revision=1,
    )
    admitted2, _ = admit_research_plan(
        plan=revised, claim_graph=build_claim_graph(CLAIM), reachable_lookup=reachable
    )
    assert admitted2 is True
    assert revised.plan_id.endswith("-rev1")
    assert revised.committed_pmids == ["111"]
    assert len([p for p in llm.prompts if "REVISE the REFUSED research PLAN" in p]) == 1


def test_patent_ic01_blocks_when_analysis_lacks_evidence_edge() -> None:
    """Patent IC-01 BLOCK: ANALYSIS node present but missing ANALYZES edge to
    EVIDENCE node. Graph has every required node type so Tier-1 chain rule
    passes, but the analysis link is incomplete — IC-01 must catch it."""
    from app.models import ClaimEdge, ClaimGraph as ClaimGraphModel, ClaimNode

    broken_graph = ClaimGraphModel(
        claim_id="claim-1",
        claim_text=CLAIM.text,
        nodes=[
            ClaimNode(id="q", label="q", node_type="QUESTION", timestamp=1),
            ClaimNode(id="m", label="m", node_type="METHOD", timestamp=2),
            ClaimNode(id="e", label="e", node_type="EVIDENCE", timestamp=3),
            ClaimNode(id="a", label="a", node_type="ANALYSIS", timestamp=4),
            ClaimNode(id="c", label="c", node_type="CLAIM", timestamp=5),
        ],
        edges=[
            ClaimEdge(source="q", target="m", edge_type="ADDRESSES"),
            ClaimEdge(source="m", target="e", edge_type="PRODUCES"),
            # MISSING: any ANALYZES edge between e and a
            ClaimEdge(source="a", target="c", edge_type="SUPPORTS"),
        ],
    )
    plan = parse_research_plan(
        "QUESTION: q\nMETHOD: appraise sources\n"
        "SCOPE: pop=40-60 years=2015-2018\nPMIDS: 111\nRATIONALE: ok.",
        plan_id="p", claim_id="claim-1", year=2020, claim_text=CLAIM.text,
    )
    catalog_lookup = {"111": object()}  # type: ignore[dict-item]
    result = admit_research_plan(
        plan=plan, claim_graph=broken_graph, reachable_lookup=catalog_lookup,
    )
    admitted, reasons = result
    assert admitted is False
    assert any("Patent IC-01" in r for r in reasons)
    assert any("IC-01" in b for b in result.blocks)


def test_patent_gc02_blocks_when_no_question_to_claim_path() -> None:
    """Patent GC-02 BLOCK: graph lists every required node type but the edges
    do not form any directed path from QUESTION to CLAIM."""
    from app.models import ClaimEdge, ClaimGraph as ClaimGraphModel, ClaimNode

    disconnected = ClaimGraphModel(
        claim_id="claim-1",
        claim_text=CLAIM.text,
        nodes=[
            ClaimNode(id="q", label="q", node_type="QUESTION", timestamp=1),
            ClaimNode(id="m", label="m", node_type="METHOD", timestamp=2),
            ClaimNode(id="e", label="e", node_type="EVIDENCE", timestamp=3),
            ClaimNode(id="a", label="a", node_type="ANALYSIS", timestamp=4),
            ClaimNode(id="c", label="c", node_type="CLAIM", timestamp=5),
        ],
        edges=[
            # Edges exist but form two disconnected sub-graphs: {q,m} and {e,a,c}
            ClaimEdge(source="q", target="m", edge_type="ADDRESSES"),
            ClaimEdge(source="e", target="a", edge_type="ANALYZES"),
            ClaimEdge(source="a", target="c", edge_type="SUPPORTS"),
        ],
    )
    plan = parse_research_plan(
        "QUESTION: q\nMETHOD: appraise sources\n"
        "SCOPE: pop=40-60 years=2015-2018\nPMIDS: 111\nRATIONALE: ok.",
        plan_id="p", claim_id="claim-1", year=2020, claim_text=CLAIM.text,
    )
    result = admit_research_plan(
        plan=plan, claim_graph=disconnected, reachable_lookup={"111": object()},  # type: ignore[dict-item]
    )
    assert result.admitted is False
    assert any("GC-02" in b for b in result.blocks)


def test_patent_ic03_warns_when_multiple_claims_share_analysis_parent() -> None:
    """Patent IC-03 WARN: two CLAIM nodes both linked from the same ANALYSIS
    via SUPPORTS edges — structural prerequisite of multi-claim scope conflict.
    Does NOT block on its own; contributes to GC-01 WARN accumulation."""
    from app.models import ClaimEdge, ClaimGraph as ClaimGraphModel, ClaimNode

    shared_analysis = ClaimGraphModel(
        claim_id="claim-1",
        claim_text=CLAIM.text,
        nodes=[
            ClaimNode(id="q", label="q", node_type="QUESTION", timestamp=1),
            ClaimNode(id="m", label="m", node_type="METHOD", timestamp=2),
            ClaimNode(id="e", label="e", node_type="EVIDENCE", timestamp=3),
            ClaimNode(id="a", label="a", node_type="ANALYSIS", timestamp=4),
            ClaimNode(id="c1", label="c1", node_type="CLAIM", timestamp=5),
            ClaimNode(id="c2", label="c2", node_type="CLAIM", timestamp=5),
        ],
        edges=[
            ClaimEdge(source="q", target="m", edge_type="ADDRESSES"),
            ClaimEdge(source="m", target="e", edge_type="PRODUCES"),
            ClaimEdge(source="e", target="a", edge_type="ANALYZES"),
            ClaimEdge(source="a", target="c1", edge_type="SUPPORTS"),
            ClaimEdge(source="a", target="c2", edge_type="SUPPORTS"),
        ],
    )
    plan = parse_research_plan(
        "QUESTION: q\nMETHOD: appraise sources\n"
        "SCOPE: pop=40-60 years=2015-2018\nPMIDS: 111\nRATIONALE: ok.",
        plan_id="p", claim_id="claim-1", year=2020, claim_text=CLAIM.text,
    )
    # Use real CorpusItem so the SpC-01 scope-vs-source check passes (IC-03 is
    # the only intended violation in this fixture).
    from app.ecology import CorpusItem as CI
    real_lookup = {
        "111": CI(
            item_id="111", kind="real", text="x", rationale="x",
            direction="NEUTRAL", cited_ids=["111"], resolved_real_ids=["111"],
            resolved_locators=["PMID:111"],
            scope=EvidenceScope(
                population_low=40, population_high=60,
                year_start=2015, year_end=2018,
            ),
        )
    }
    result = admit_research_plan(
        plan=plan, claim_graph=shared_analysis, reachable_lookup=real_lookup,
    )
    # Still admitted (WARN does not block alone), but warn list carries IC-03.
    assert result.admitted is True
    assert any("share an ANALYSIS parent" in w for w in result.warns)
    assert any("IC-03" in r for r in result.reasons)


def test_patent_spc02_warns_on_small_n_generalizing_claim_and_gc01_escalates() -> None:
    """Patent SpC-02 WARN fires when claim generalizes (wide scope) but study
    sample size is below threshold. Patent GC-01 escalates to BLOCK when total
    WARN count >= WARN_ACCUMULATION_BLOCK (default 5)."""
    from app.models import EvidenceScope as ScopeModel, ResearchPlan as Plan
    from app.process_gate import (
        WARN_ACCUMULATION_BLOCK,
        ProcessViolation,
        assess_research_process,
        execution_deviations,
        process_integrity_score,
    )

    # Build a generalizing claim (wide population + timeframe) with small n.
    wide_scope = ScopeModel(
        population_low=0, population_high=80, year_start=1990, year_end=2025
    )
    narrow_source = ScopeModel(
        population_low=0, population_high=80, year_start=1990, year_end=2025
    )
    plan = Plan(
        plan_id="p", claim_id="claim-1", year=2020,
        question="q", method="appraise",
        committed_pmids=["111"], claimed_scope=wide_scope,
    )
    study = Study(
        id="s", claim_id="claim-1", year=2020, direction="SUPPORTS",
        quality=0.9, provenance="GROUNDED", pmids=["111"], catalog_pmids=["111"],
        numeric=True, n=10, rationale="r",
        claimed_scope=wide_scope, source_scope=narrow_source,
        plan_id="p",
        research_plan=plan,
    )

    # SpC-02 fires on this single study.
    violations = execution_deviations(plan=plan, study=study)
    assert any(v.code == "SpC-02" and v.severity == "warn" for v in violations)
    assert all(v.severity != "block" for v in violations)

    # Single WARN: score drops 0.1; still above threshold 0.60.
    score_one_warn = process_integrity_score(civer_passed=True, violations=violations)
    assert score_one_warn == 0.9

    # GC-01: WARN_ACCUMULATION_BLOCK WARNs → score 0.0 (escalation).
    many_warns = [
        ProcessViolation(code="SpC-02", severity="warn", message=f"warn {i}")
        for i in range(WARN_ACCUMULATION_BLOCK)
    ]
    score_gc01 = process_integrity_score(civer_passed=True, violations=many_warns)
    assert score_gc01 == 0.0


def test_repair_loop_persistent_abstain_after_max_revisions() -> None:
    """If revise also emits a bogus commit, the constrained outcome is a
    persistent abstain (no study) after MAX_PLAN_REVISIONS attempts. Verifies
    the audit trail records `design-abstain-persistent` with block severity and
    that no execution happened."""
    from app.llm import DeterministicFakeClient
    from app.models import RunRequestModel
    from app.simulator import simulate_run

    class AlwaysBogusLLM:
        scientific = True
        degradation_reason = None

        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, prompt: str, *, seed: int) -> str:
            self.prompts.append(prompt)
            if "PRE-REGISTER a research PLAN" in prompt or "REVISE the REFUSED" in prompt:
                return (
                    "QUESTION: q\nMETHOD: appraise sources\n"
                    "SCOPE: pop=40-60 years=2015-2018\n"
                    "PMIDS: PMID-FAKE-9\nRATIONALE: still bogus."
                )
            return "DIRECTION: NEUTRAL\nSCOPE: pop=40-60 years=2015-2018\nPMIDS: none\nRATIONALE: x."

        def describe(self) -> ModelDescriptor:
            return ModelDescriptor(name="always-bogus", digest="test")

    llm = AlwaysBogusLLM()
    agent = ResearchAgent(pubmed=ScopedPubMed(), llm=llm, retmax=5)
    plan, catalog = agent.run_design(
        claim_id="claim-1", claim_text=CLAIM.text, simulated_year=2020
    )
    reachable = _reachable_lookup_from_catalog(catalog)
    admitted, reasons = admit_research_plan(
        plan=plan, claim_graph=build_claim_graph(CLAIM), reachable_lookup=reachable
    )
    assert admitted is False
    # Simulate 2 revise rounds; both still bogus -> persistent abstain.
    for rev in (1, 2):
        plan = agent.run_revise(
            prior_plan=plan,
            refusal_reasons=reasons,
            catalog=catalog,
            claim_text=CLAIM.text,
            revision=rev,
        )
        admitted, reasons = admit_research_plan(
            plan=plan, claim_graph=build_claim_graph(CLAIM), reachable_lookup=reachable
        )
        assert admitted is False
    # Two revise prompts issued; never executed.
    revise_calls = [p for p in llm.prompts if "REVISE the REFUSED" in p]
    execute_calls = [p for p in llm.prompts if "EXECUTE the pre-registered plan" in p]
    assert len(revise_calls) == 2
    assert len(execute_calls) == 0
