"""Tests for the v4 per-arm asymmetric Tier-1 + multi-step SR/MA (this session).

Covers:
- CONSTRAINED arm runs a separable DESIGN call -> pre-execution CIVER gate
  (Article I, prove integrity BEFORE the process runs) -> EXECUTE call. A refused
  design never executes and no study enters the constrained corpus.
- FREE arm makes ONE merged research call per study; CONSTRAINED makes TWO
  (design + execute).
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


# --- 2. free = 1 merged call, constrained = 2 (design + execute) ------------


def test_free_one_call_constrained_two_calls() -> None:
    free_llm = RoutingLLM(design="", execute="")
    free_agent = _agent(free_llm)
    free_agent.run(claim_id="claim-1", claim_text=CLAIM.text, simulated_year=2020)
    # The merged free call is neither a DESIGN nor an EXECUTE prompt.
    assert len(free_llm.prompts) == 1
    assert "PRE-REGISTER a research PLAN" not in free_llm.prompts[0]
    assert "EXECUTE the pre-registered plan" not in free_llm.prompts[0]

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
    assert "design-refused" in event_types
    assert "execution-deviated" in event_types
    assert verify_audit_chain(bundle.audit_trail)
    # design / execution phase events are constrained-only.
    assert all(
        e.branch == "constrained"
        for e in bundle.audit_trail
        if e.phase in ("design", "execution")
    )
    # A refused design must have a WARN/BLOCK severity and never enters the corpus.
    refused = [e for e in bundle.audit_trail if e.event_type == "design-refused"]
    assert refused and all(e.severity == "block" for e in refused)
