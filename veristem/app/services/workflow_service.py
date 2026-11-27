import asyncio
import sqlite3
import uuid
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.core.config import get_settings
from app.core.logging import get_logger
from app.database.repository import ProblemRepository
from app.database.session import get_session_factory
from app.graph.state import VeriSTEMState
from app.graph.workflow import build_veristem_graph
from app.schemas.models import (
    AgentExecutionResult,
    CorrectionRequest,
    ExecutionPlan,
    FinalSolutionReport,
    HumanReviewDecision,
    ProblemSpecification,
    ResumeProblemRequest,
    VerificationReport,
    WorkflowStatus,
    utc_now,
)
from app.services.gemini_client import GeminiGateway, LLMGateway
from app.tools.registry import get_tool_registry

logger = get_logger(__name__)


class WorkflowService:
    """Coordinates LangGraph execution, checkpoint recovery, and SQLAlchemy persistence."""

    def __init__(self, llm: LLMGateway | None = None) -> None:
        self.llm = llm or GeminiGateway()
        self.registry = get_tool_registry()
        self._active_tasks: dict[str, asyncio.Task[None]] = {}

    async def _persist_transition(self, stage: str, state: VeriSTEMState, artifact: Any) -> None:
        problem_id = state["problem_id"]
        execution_id = state["execution_id"]
        factory = get_session_factory()
        async with factory() as session:
            repo = ProblemRepository(session)
            if stage == "UNDERSTANDING" and artifact is not None:
                spec = ProblemSpecification.model_validate(state["specification"])
                await repo.update_workflow_state(
                    problem_id,
                    status=WorkflowStatus(state["workflow_status"]),
                    current_stage="UNDERSTANDING",
                    specification=spec,
                )
                await repo.add_agent_result(problem_id, execution_id, artifact)
            elif stage == "NEEDS_CLARIFICATION":
                await repo.update_workflow_state(
                    problem_id,
                    status=WorkflowStatus.NEEDS_CLARIFICATION,
                    current_stage="NEEDS_CLARIFICATION",
                )
            elif stage == "PLANNING" and artifact is not None:
                plan, agent_res = artifact
                assert isinstance(plan, ExecutionPlan)
                await repo.update_workflow_state(problem_id, current_stage="PLANNING")
                await repo.add_execution_plan(problem_id, execution_id, plan)
                await repo.add_agent_result(problem_id, execution_id, agent_res)
            elif stage == "SOLVING" and artifact is not None:
                assert isinstance(artifact, AgentExecutionResult)
                await repo.update_workflow_state(problem_id, current_stage="SOLVING")
                await repo.add_agent_result(problem_id, execution_id, artifact)
            elif stage == "VERIFICATION" and artifact is not None:
                report, agent_res = artifact
                assert isinstance(report, VerificationReport)
                await repo.update_workflow_state(problem_id, current_stage="VERIFICATION")
                await repo.add_verification_result(
                    problem_id,
                    execution_id,
                    int(state.get("correction_attempts", 0)),
                    report,
                )
                await repo.add_agent_result(problem_id, execution_id, agent_res)
            elif stage == "CRITIC" and artifact is not None:
                correction, agent_res = artifact
                assert isinstance(correction, CorrectionRequest)
                await repo.update_workflow_state(
                    problem_id,
                    current_stage="CRITIC",
                    correction_attempts=correction.attempt_number,
                )
                await repo.add_retry_record(problem_id, execution_id, correction)
                await repo.add_agent_result(problem_id, execution_id, agent_res)
            elif stage == "HUMAN_REVIEW":
                await repo.update_workflow_state(
                    problem_id,
                    status=WorkflowStatus.HUMAN_REVIEW,
                    current_stage="HUMAN_REVIEW",
                )
            elif stage == "HUMAN_REVIEW_DECIDED" and artifact is not None:
                assert isinstance(artifact, HumanReviewDecision)
                await repo.add_human_review_decision(problem_id, execution_id, artifact)
                await repo.update_workflow_state(
                    problem_id,
                    status=WorkflowStatus(state["workflow_status"]),
                    current_stage="HUMAN_REVIEW_DECIDED",
                )
            elif stage == "REPORTING" and artifact is not None:
                final_report, agent_res = artifact
                assert isinstance(final_report, FinalSolutionReport)
                await repo.add_agent_result(problem_id, execution_id, agent_res)
                await repo.save_final_report(problem_id, execution_id, final_report)
                await repo.update_workflow_state(
                    problem_id,
                    status=WorkflowStatus(state["workflow_status"]),
                    current_stage=state["current_stage"],
                )

    async def execute_problem(
        self,
        *,
        problem_id: str,
        execution_id: str | None = None,
        max_correction_attempts: int = 3,
    ) -> None:
        settings = get_settings()
        exec_id = execution_id or f"exec_{uuid.uuid4().hex[:12]}"
        factory = get_session_factory()

        async with factory() as session:
            repo = ProblemRepository(session)
            problem = await repo.get_problem(problem_id)
            await repo.update_workflow_state(
                problem_id,
                execution_id=exec_id,
                status=WorkflowStatus.RUNNING,
                current_stage="STARTING",
                max_correction_attempts=max_correction_attempts,
            )
            initial_state: VeriSTEMState = {
                "problem_id": problem.id,
                "execution_id": exec_id,
                "question": problem.question,
                "domain_hint": problem.domain_hint,
                "started_at": utc_now().isoformat(),
                "current_stage": "STARTING",
                "workflow_status": WorkflowStatus.RUNNING.value,
                "max_correction_attempts": max_correction_attempts,
                "correction_attempts": 0,
                "correction_history": [],
                "tool_records": [],
                "agent_timeline": [],
            }

        try:
            async with AsyncSqliteSaver.from_conn_string(settings.langgraph_checkpoint_db) as checkpointer:
                graph = build_veristem_graph(
                    llm=self.llm,
                    registry=self.registry,
                    checkpointer=checkpointer,
                    on_transition=self._persist_transition,
                )
                config = {"configurable": {"thread_id": problem_id}}
                await asyncio.wait_for(
                    graph.ainvoke(initial_state, config=config),
                    timeout=settings.workflow_timeout_seconds,
                )
        except sqlite3.Error as exc:
            logger.exception("checkpoint_error", problem_id=problem_id, error=str(exc))
            await self._mark_failed(problem_id, f"Checkpoint error: {exc}")
        except Exception as exc:
            logger.exception("workflow_execution_failed", problem_id=problem_id, error=str(exc))
            await self._mark_failed(problem_id, str(exc))

    async def resume_problem(self, *, problem_id: str, request: ResumeProblemRequest) -> None:
        settings = get_settings()
        factory = get_session_factory()
        async with factory() as session:
            repo = ProblemRepository(session)
            problem = await repo.get_problem(problem_id)
            await repo.update_workflow_state(
                problem_id,
                status=WorkflowStatus.RUNNING,
                current_stage="RESUMING",
            )
            status = WorkflowStatus(problem.status)

        resume_payload: dict[str, Any]
        if status == WorkflowStatus.NEEDS_CLARIFICATION:
            resume_payload = {"clarification_response": request.clarification_response or request.notes}
        else:
            decision = HumanReviewDecision(
                reviewer_action=request.action,
                notes=request.notes,
                clarification_response=request.clarification_response,
                override_answer=request.override_answer,
            )
            resume_payload = decision.model_dump(mode="json")

        try:
            async with AsyncSqliteSaver.from_conn_string(settings.langgraph_checkpoint_db) as checkpointer:
                graph = build_veristem_graph(
                    llm=self.llm,
                    registry=self.registry,
                    checkpointer=checkpointer,
                    on_transition=self._persist_transition,
                )
                config = {"configurable": {"thread_id": problem_id}}
                await asyncio.wait_for(
                    graph.ainvoke(Command(resume=resume_payload), config=config),
                    timeout=settings.workflow_timeout_seconds,
                )
        except Exception as exc:
            logger.exception("workflow_resume_failed", problem_id=problem_id, error=str(exc))
            await self._mark_failed(problem_id, str(exc))

    async def _mark_failed(self, problem_id: str, reason: str) -> None:
        factory = get_session_factory()
        async with factory() as session:
            repo = ProblemRepository(session)
            await repo.update_workflow_state(
                problem_id,
                status=WorkflowStatus.FAILED,
                current_stage=f"FAILED: {reason[:120]}",
            )

    def launch_background_solve(
        self, *, problem_id: str, execution_id: str, max_correction_attempts: int
    ) -> None:
        task = asyncio.create_task(
            self.execute_problem(
                problem_id=problem_id,
                execution_id=execution_id,
                max_correction_attempts=max_correction_attempts,
            )
        )
        self._active_tasks[problem_id] = task
        task.add_done_callback(lambda _: self._active_tasks.pop(problem_id, None))

    async def shutdown(self) -> None:
        for task in list(self._active_tasks.values()):
            task.cancel()
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks.values(), return_exceptions=True)
