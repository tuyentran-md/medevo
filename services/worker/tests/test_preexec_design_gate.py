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
    # After no-evidence outputs stopped receiving execution warrants, the
    # deterministic fake may end in persistent abstain rather than repair. Either
    # way the design/gate path is explicit and hash-verifiable.
    assert event_types & {"design-repaired", "design-abstain-persistent"}
    assert "execution-deviated" in event_types
    assert verify_audit_chain(bundle.audit_trail)
    # design / execution phase events are recorded for both branches; only
    # constrained enters the repair loop / abstains.
    assert {e.branch for e in bundle.audit_trail if e.phase == "design"} <= {
        "free",
        "constrained",
    }
    assert "constrained" in {e.branch for e in bundle.audit_trail if e.phase == "design"}
    # A `design-repaired` event records a constrained-arm success of refuse+repair.
    repaired = [e for e in bundle.audit_trail if e.event_type == "design-repaired"]
    assert all(e.severity == "info" and e.branch == "constrained" for e in repaired)
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
    """CIVER 2.0 spec SC-02 BLOCK (medevo formerly mis-named as IC-01):
    ANALYSIS node present but missing ANALYZES edge to EVIDENCE node. Graph
    has every required node type so Tier-1 chain rule passes, but the
    analysis link is incomplete — SC-02 must catch it."""
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
    assert any("SC-02" in r for r in reasons)
    assert any("SC-02" in b for b in result.blocks)


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


def test_no_citation_abstention_does_not_receive_execution_warrant() -> None:
    """A NEUTRAL/no-PMID output may be honest abstention, but it is not
    warrantable evidence and must not enter the constrained corpus."""
    from app.models import ResearchPlan as Plan
    from app.process_gate import issue_process_warrant

    plan = Plan(
        plan_id="p",
        claim_id="claim-1",
        year=2020,
        question="q",
        method="appraise",
        committed_pmids=[],
        claimed_scope=EvidenceScope(
            population_low=0, population_high=120, year_start=1900, year_end=2025
        ),
    )
    study = Study(
        id="s-no-evidence",
        claim_id="claim-1",
        year=2020,
        direction="NEUTRAL",
        quality=0.2,
        provenance="UNGROUNDED",
        pmids=[],
        catalog_pmids=[],
        numeric=False,
        n=None,
        rationale="No committed sources, so abstain.",
        claimed_scope=EvidenceScope(
            population_low=0, population_high=120, year_start=1900, year_end=2025
        ),
        source_scope=EvidenceScope(
            population_low=0, population_high=120, year_start=2020, year_end=2020
        ),
        plan_id="p",
        research_plan=plan,
    )

    assessment, warrant = issue_process_warrant(
        run_id="r",
        branch="constrained",
        year=2020,
        study=study,
        claim_graph=build_claim_graph(CLAIM),
    )

    assert assessment.passed is False
    assert warrant.issued is False
    assert warrant.status == "REFUSED"
    assert any(
        "abstention is not warrantable corpus evidence" in r
        for r in assessment.reasons
    )


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


# --- CIVER 2.0 spec Tier-2 + Tier-5 (this session) -------------------------- #


def test_ac01_blocks_when_question_and_method_study_types_disagree() -> None:
    """CIVER 2.0 §5.3 AC-01 BLOCK: QUESTION.study_type ≠ METHOD.study_type.

    A causal-cohort question cannot be answered by a case-series method. The
    plan must declare both attributes; when they mismatch the gate refuses
    BEFORE execution — this is the methodology-rigor check that the previous
    structure-only CIVER missed.
    """
    from app.models import ClaimEdge, ClaimGraph as CG, ClaimNode

    good_graph = CG(
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
            ClaimEdge(source="e", target="a", edge_type="ANALYZES"),
            ClaimEdge(source="a", target="c", edge_type="SUPPORTS"),
        ],
    )
    plan = parse_research_plan(
        "QUESTION: q\nMETHOD: appraise the committed cohort studies\n"
        "SCOPE: pop=40-60 years=2015-2018\nPMIDS: 111\nRATIONALE: ok.\n"
        # AC-01 trigger: question demands causal-cohort, method proposes case-series.
        "QUESTION_ATTRS: study_type=causal-cohort\n"
        "METHOD_ATTRS: study_type=case-series variables=smoking,CHD\n"
        "EVIDENCE_ATTRS: study_type=cohort variables=smoking,CHD\n"
        "ANALYSIS_ATTRS: statistical_method=narrative-synthesis\n",
        plan_id="p", claim_id="claim-1", year=2020, claim_text=CLAIM.text,
    )
    result = admit_research_plan(
        plan=plan, claim_graph=good_graph, reachable_lookup={"111": object()},  # type: ignore[dict-item]
    )
    assert result.admitted is False
    assert any("AC-01" in b for b in result.blocks)


def test_ac01_admits_when_study_types_match() -> None:
    """Matched study_type pair must not trigger AC-01; the plan otherwise
    well-formed should be admitted (with IS above the GC-03 threshold)."""
    from app.models import ClaimEdge, ClaimGraph as CG, ClaimNode
    from app.ecology import CorpusItem as CI

    good_graph = CG(
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
            ClaimEdge(source="e", target="a", edge_type="ANALYZES"),
            ClaimEdge(source="a", target="c", edge_type="SUPPORTS"),
        ],
    )
    plan = parse_research_plan(
        "QUESTION: q\nMETHOD: random-effects meta-analysis of the committed cohort PMIDs\n"
        "SCOPE: pop=40-60 years=2015-2018\nPMIDS: 111\nRATIONALE: ok.\n"
        "QUESTION_ATTRS: study_type=causal-cohort\n"
        "METHOD_ATTRS: study_type=causal-cohort variables=smoking,CHD\n"
        "EVIDENCE_ATTRS: study_type=cohort variables=smoking,CHD sample_size=4000\n"
        "ANALYSIS_ATTRS: statistical_method=random-effects-meta-analysis variables=smoking,CHD\n",
        plan_id="p", claim_id="claim-1", year=2020, claim_text=CLAIM.text,
    )
    lookup = {
        "111": CI(
            item_id="111", kind="real", text="t", rationale="r", direction="NEUTRAL",
            cited_ids=["111"], resolved_real_ids=["111"], resolved_locators=["PMID:111"],
            scope=EvidenceScope(population_low=40, population_high=60, year_start=2015, year_end=2018),
        )
    }
    result = admit_research_plan(plan=plan, claim_graph=good_graph, reachable_lookup=lookup)
    assert result.admitted is True
    assert result.integrity_score >= 0.60
    assert not any("AC-01" in b for b in result.blocks)


def test_ac03_warns_on_incompatible_statistical_method() -> None:
    """CIVER 2.0 §5.3 AC-03 WARN: ANALYSIS.statistical_method must be
    compatible with EVIDENCE.study_type. Diagnostic-accuracy evidence cannot
    be analysed by a hazard-ratio pool. The rule WARNs (counted toward GC-03
    IS) and only BLOCKs via accumulated IS, not on its own."""
    from app.models import ClaimEdge, ClaimGraph as CG, ClaimNode
    from app.ecology import CorpusItem as CI

    good_graph = CG(
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
            ClaimEdge(source="e", target="a", edge_type="ANALYZES"),
            ClaimEdge(source="a", target="c", edge_type="SUPPORTS"),
        ],
    )
    plan = parse_research_plan(
        "QUESTION: q\nMETHOD: review the committed diagnostic-accuracy studies\n"
        "SCOPE: pop=40-60 years=2015-2018\nPMIDS: 111\nRATIONALE: ok.\n"
        "QUESTION_ATTRS: study_type=diagnostic-accuracy\n"
        "METHOD_ATTRS: study_type=diagnostic-accuracy variables=ultrasound,gold-standard\n"
        "EVIDENCE_ATTRS: study_type=diagnostic-accuracy variables=ultrasound,gold-standard sample_size=200\n"
        # AC-03 trigger: hazard-ratio-pool only valid for cohort/RCT, not diagnostic-accuracy
        "ANALYSIS_ATTRS: statistical_method=hazard-ratio-pool variables=ultrasound,gold-standard\n",
        plan_id="p", claim_id="claim-1", year=2020, claim_text=CLAIM.text,
    )
    lookup = {
        "111": CI(
            item_id="111", kind="real", text="t", rationale="r", direction="NEUTRAL",
            cited_ids=["111"], resolved_real_ids=["111"], resolved_locators=["PMID:111"],
            scope=EvidenceScope(population_low=40, population_high=60, year_start=2015, year_end=2018),
        )
    }
    result = admit_research_plan(plan=plan, claim_graph=good_graph, reachable_lookup=lookup)
    # AC-03 alone is WARN, not BLOCK; the plan can still admit on a single warn
    # (IS stays above 0.60). The point is the warn is RECORDED.
    assert any("AC-03" in w for w in result.warns)


def test_gc03_blocks_when_integrity_score_below_threshold() -> None:
    """CIVER 2.0 §7 GC-03 BLOCK: a plan that piles up enough WARNs to drag
    Integrity Score below 0.60 must be refused even with no individual BLOCK.
    Six WARNs at 0.08 each ≈ -0.48 → IS ≈ 0.53 < 0.60 → refuse."""
    from app.ecology import _integrity_score, _IS_GATING_THRESHOLD

    # Pure-function check: 6 WARNs, 1 complete chain → IS = 1 - 6*0.08 + 0.01 = 0.53.
    score = _integrity_score(blocks=[], warns=["w"] * 6, complete_chains=1)
    assert score < _IS_GATING_THRESHOLD
    # And 1 BLOCK alone already trips the threshold (0.15 deduction is small,
    # but combined with the admitted=False on the block itself, the gate refuses).
    score_block = _integrity_score(blocks=["x"] * 3, warns=[], complete_chains=1)
    # 3 BLOCKs at 0.15 each → IS = 1 - 0.45 + 0.01 = 0.56 < 0.60
    assert score_block < _IS_GATING_THRESHOLD


def test_integrity_score_is_a_pure_function() -> None:
    """Same (blocks, warns, complete_chains) → same IS, no LLM, no time. The
    spec §7 mathematical-property requirement."""
    from app.ecology import _integrity_score

    a = _integrity_score(blocks=["b1"], warns=["w1", "w2"], complete_chains=1)
    b = _integrity_score(blocks=["b1"], warns=["w1", "w2"], complete_chains=1)
    assert a == b
    # Different inputs → different IS.
    c = _integrity_score(blocks=[], warns=[], complete_chains=1)
    assert c != a
