from datetime import datetime
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.agents.critic import run_critic_agent
from app.agents.planning import run_planning_agent
from app.agents.reporter import run_reporting_agent
from app.agents.solver import SolverSynthesis, run_solver_agent
from app.agents.understanding import run_understanding_agent
from app.agents.verifier import run_verification_agent
from app.core.logging import get_logger
from app.core.tracing import trace_span
from app.graph.state import VeriSTEMState
from app.schemas.models import (
    CorrectionRequest,
    ExecutionPlan,
    HumanReviewDecision,
    ProblemDomain,
    ProblemSpecification,
    ToolExecutionRecord,
    VerificationReport,
    VerificationStatus,
    WorkflowStatus,
)
from app.services.gemini_client import LLMGateway
from app.tools.registry import SafeToolRegistry

logger = get_logger(__name__)


def build_veristem_graph(
    *,
    llm: LLMGateway,
    registry: SafeToolRegistry,
    checkpointer: BaseCheckpointSaver[Any],
    on_transition: Any | None = None,
) -> Any:
    """Construct and compile the VeriSTEM LangGraph StateGraph with conditional routing and interrupts."""
    tool_cache: dict[str, ToolExecutionRecord] = {}

    async def _notify(stage: str, state: VeriSTEMState, artifact: Any = None) -> None:
        if on_transition is not None:
            await on_transition(stage, state, artifact)

    async def understand_node(state: VeriSTEMState) -> VeriSTEMState:
        with trace_span("agent.understand", {"problem_id": state["problem_id"]}):
            domain_hint = ProblemDomain(state["domain_hint"]) if state.get("domain_hint") else None
            spec, agent_res = await run_understanding_agent(
                llm=llm,
                question=state["question"],
                domain_hint=domain_hint,
                clarification_response=state.get("clarification_response"),
            )
            timeline = [*state.get("agent_timeline", []), agent_res.model_dump(mode="json")]
            next_status = (
                WorkflowStatus.NEEDS_CLARIFICATION.value
                if spec.needs_clarification
                else WorkflowStatus.RUNNING.value
            )
            update: VeriSTEMState = {
                "current_stage": "UNDERSTANDING",
                "workflow_status": next_status,
                "specification": spec.model_dump(mode="json"),
                "agent_timeline": timeline,
            }
            await _notify("UNDERSTANDING", {**state, **update}, agent_res)
            return update

    async def clarify_node(state: VeriSTEMState) -> VeriSTEMState:
        spec = ProblemSpecification.model_validate(state["specification"])
        await _notify("NEEDS_CLARIFICATION", state, None)
        resume_payload = interrupt(
            {
                "type": "clarification_required",
                "problem_id": state["problem_id"],
                "question": spec.clarification_question or "Additional parameters are required.",
                "missing_information": spec.missing_information,
            }
        )
        clarification_text = (
            resume_payload.get("clarification_response")
            if isinstance(resume_payload, dict)
            else str(resume_payload)
        )
        return {
            "clarification_response": clarification_text,
            "current_stage": "CLARIFICATION_RECEIVED",
            "workflow_status": WorkflowStatus.RUNNING.value,
        }

    async def plan_node(state: VeriSTEMState) -> VeriSTEMState:
        with trace_span("agent.plan", {"problem_id": state["problem_id"]}):
            spec = ProblemSpecification.model_validate(state["specification"])
            plan, agent_res = await run_planning_agent(llm=llm, spec=spec, registry=registry)
            timeline = [*state.get("agent_timeline", []), agent_res.model_dump(mode="json")]
            update: VeriSTEMState = {
                "current_stage": "PLANNING",
                "workflow_status": WorkflowStatus.RUNNING.value,
                "plan": plan.model_dump(mode="json"),
                "agent_timeline": timeline,
            }
            await _notify("PLANNING", {**state, **update}, (plan, agent_res))
            return update

    async def solve_node(state: VeriSTEMState) -> VeriSTEMState:
        with trace_span("agent.solve", {"problem_id": state["problem_id"]}):
            spec = ProblemSpecification.model_validate(state["specification"])
            plan = ExecutionPlan.model_validate(state["plan"])
            correction = (
                CorrectionRequest.model_validate(state["latest_correction"])
                if state.get("latest_correction")
                else None
            )
            synthesis, agent_res = await run_solver_agent(
                llm=llm,
                spec=spec,
                plan=plan,
                registry=registry,
                correction=correction,
                tool_cache=tool_cache,
            )
            timeline = [*state.get("agent_timeline", []), agent_res.model_dump(mode="json")]
            tools = [*state.get("tool_records", []), *[r.model_dump(mode="json") for r in agent_res.tool_records]]
            update: VeriSTEMState = {
                "current_stage": "SOLVING",
                "solver_synthesis": synthesis.model_dump(mode="json"),
                "agent_timeline": timeline,
                "tool_records": tools,
            }
            await _notify("SOLVING", {**state, **update}, agent_res)
            return update

    async def verify_node(state: VeriSTEMState) -> VeriSTEMState:
        with trace_span("agent.verify", {"problem_id": state["problem_id"]}):
            spec = ProblemSpecification.model_validate(state["specification"])
            synthesis = SolverSynthesis.model_validate(state["solver_synthesis"])
            solver_tools = [ToolExecutionRecord.model_validate(t) for t in state.get("tool_records", [])]
            report, agent_res = await run_verification_agent(
                llm=llm,
                spec=spec,
                synthesis=synthesis,
                solver_tools=solver_tools,
                registry=registry,
                tool_cache=tool_cache,
            )
            timeline = [*state.get("agent_timeline", []), agent_res.model_dump(mode="json")]
            tools = [*state.get("tool_records", []), *[r.model_dump(mode="json") for r in agent_res.tool_records]]
            update: VeriSTEMState = {
                "current_stage": "VERIFICATION",
                "verification_report": report.model_dump(mode="json"),
                "agent_timeline": timeline,
                "tool_records": tools,
            }
            await _notify("VERIFICATION", {**state, **update}, (report, agent_res))
            return update

    async def critic_node(state: VeriSTEMState) -> VeriSTEMState:
        with trace_span("agent.critic", {"problem_id": state["problem_id"]}):
            next_attempt = int(state.get("correction_attempts", 0)) + 1
            spec = ProblemSpecification.model_validate(state["specification"])
            synthesis = SolverSynthesis.model_validate(state["solver_synthesis"])
            verification = VerificationReport.model_validate(state["verification_report"])
            correction, agent_res = await run_critic_agent(
                llm=llm,
                attempt_number=next_attempt,
                spec=spec,
                synthesis=synthesis,
                verification=verification,
            )
            history = [*state.get("correction_history", []), correction.model_dump(mode="json")]
            timeline = [*state.get("agent_timeline", []), agent_res.model_dump(mode="json")]
            update: VeriSTEMState = {
                "current_stage": "CRITIC",
                "correction_attempts": next_attempt,
                "latest_correction": correction.model_dump(mode="json"),
                "correction_history": history,
                "agent_timeline": timeline,
            }
            await _notify("CRITIC", {**state, **update}, (correction, agent_res))
            return update

    async def human_review_node(state: VeriSTEMState) -> VeriSTEMState:
        await _notify("HUMAN_REVIEW", {**state, "workflow_status": WorkflowStatus.HUMAN_REVIEW.value}, None)
        decision_raw = interrupt(
            {
                "type": "human_review_required",
                "problem_id": state["problem_id"],
                "correction_attempts": state.get("correction_attempts", 0),
                "verification_report": state.get("verification_report"),
            }
        )
        decision = HumanReviewDecision.model_validate(decision_raw)
        synthesis_dict = dict(state.get("solver_synthesis") or {})
        if decision.reviewer_action == "override" and decision.override_answer:
            synthesis_dict["final_answer"] = decision.override_answer
        next_status = (
            WorkflowStatus.UNRESOLVED.value
            if decision.reviewer_action == "reject"
            else WorkflowStatus.RUNNING.value
        )
        update: VeriSTEMState = {
            "current_stage": "HUMAN_REVIEW_DECIDED",
            "workflow_status": next_status,
            "human_review_decision": decision.model_dump(mode="json"),
            "solver_synthesis": synthesis_dict,
        }
        await _notify("HUMAN_REVIEW_DECIDED", {**state, **update}, decision)
        return update

    async def report_node(state: VeriSTEMState) -> VeriSTEMState:
        with trace_span("agent.report", {"problem_id": state["problem_id"]}):
            spec = ProblemSpecification.model_validate(state["specification"])
            synthesis = SolverSynthesis.model_validate(state["solver_synthesis"])
            verification = VerificationReport.model_validate(state["verification_report"])
            corrections = [CorrectionRequest.model_validate(c) for c in state.get("correction_history", [])]
            tools = [ToolExecutionRecord.model_validate(t) for t in state.get("tool_records", [])]
            started_at = datetime.fromisoformat(state["started_at"])

            final_report, agent_res = await run_reporting_agent(
                llm=llm,
                problem_id=state["problem_id"],
                execution_id=state["execution_id"],
                question=state["question"],
                spec=spec,
                synthesis=synthesis,
                verification=verification,
                corrections=corrections,
                tool_history=tools,
                started_at=started_at,
            )
            timeline = [*state.get("agent_timeline", []), agent_res.model_dump(mode="json")]
            final_status = (
                WorkflowStatus.UNRESOLVED.value
                if state.get("workflow_status") == WorkflowStatus.UNRESOLVED.value
                or verification.status != VerificationStatus.PASS
                and not state.get("human_review_decision")
                else WorkflowStatus.COMPLETED.value
            )
            if state.get("human_review_decision"):
                dec = HumanReviewDecision.model_validate(state["human_review_decision"])
                final_status = (
                    WorkflowStatus.UNRESOLVED.value
                    if dec.reviewer_action == "reject"
                    else WorkflowStatus.COMPLETED.value
                )

            update: VeriSTEMState = {
                "current_stage": "COMPLETED" if final_status == WorkflowStatus.COMPLETED.value else "UNRESOLVED",
                "workflow_status": final_status,
                "final_report": final_report.model_dump(mode="json"),
                "agent_timeline": timeline,
            }
            await _notify("REPORTING", {**state, **update}, (final_report, agent_res))
            return update

    def route_after_understanding(state: VeriSTEMState) -> str:
        spec = ProblemSpecification.model_validate(state["specification"])
        return "clarify" if spec.needs_clarification else "plan"

    def route_after_verification(state: VeriSTEMState) -> str:
        verif = VerificationReport.model_validate(state["verification_report"])
        if verif.status == VerificationStatus.PASS:
            return "report"
        attempts = int(state.get("correction_attempts", 0))
        max_attempts = int(state.get("max_correction_attempts", 3))
        if attempts < max_attempts:
            return "critic"
        return "human_review"

    def route_after_human_review(state: VeriSTEMState) -> str:
        dec = HumanReviewDecision.model_validate(state["human_review_decision"])
        if dec.reviewer_action == "clarify":
            return "understand"
        return "report"

    graph = StateGraph(VeriSTEMState)
    graph.add_node("understand", understand_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("plan", plan_node)
    graph.add_node("solve", solve_node)
    graph.add_node("verify", verify_node)
    graph.add_node("critic", critic_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "understand")
    graph.add_conditional_edges(
        "understand",
        route_after_understanding,
        {"clarify": "clarify", "plan": "plan"},
    )
    graph.add_edge("clarify", "understand")
    graph.add_edge("plan", "solve")
    graph.add_edge("solve", "verify")
    graph.add_conditional_edges(
        "verify",
        route_after_verification,
        {"report": "report", "critic": "critic", "human_review": "human_review"},
    )
    graph.add_edge("critic", "solve")
    graph.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {"understand": "understand", "report": "report"},
    )
    graph.add_edge("report", END)

    return graph.compile(checkpointer=checkpointer)
