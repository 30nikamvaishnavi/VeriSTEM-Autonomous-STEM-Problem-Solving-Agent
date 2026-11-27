from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.exceptions import ProblemNotFoundError
from app.database.models import (
    AgentStateORM,
    ExecutionPlanORM,
    FinalReportORM,
    HumanReviewDecisionORM,
    IdempotencyRecordORM,
    ProblemORM,
    RetryHistoryORM,
    ToolRecordORM,
    VerificationResultORM,
)
from app.schemas.models import (
    AgentExecutionResult,
    CorrectionRequest,
    ExecutionPlan,
    FinalSolutionReport,
    HumanReviewDecision,
    MetricsResponse,
    ProblemDetailResponse,
    ProblemDomain,
    ProblemSpecification,
    ProblemStatusResponse,
    ToolExecutionRecord,
    VerificationReport,
    VerificationStatus,
    WorkflowStatus,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ProblemRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_problem(
        self,
        problem_id: str,
        question: str,
        domain_hint: ProblemDomain | None,
        context: dict[str, Any],
    ) -> ProblemORM:
        record = ProblemORM(
            id=problem_id,
            question=question,
            domain_hint=domain_hint.value if domain_hint else None,
            context_json=context,
            status=WorkflowStatus.PENDING.value,
            current_stage="CREATED",
        )
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    async def get_idempotent_execution(self, idempotency_key: str) -> IdempotencyRecordORM | None:
        return await self.session.get(IdempotencyRecordORM, idempotency_key)

    async def save_idempotent_execution(
        self, idempotency_key: str, problem_id: str, execution_id: str
    ) -> None:
        self.session.add(
            IdempotencyRecordORM(
                idempotency_key=idempotency_key,
                problem_id=problem_id,
                execution_id=execution_id,
            )
        )
        await self.session.commit()

    async def get_problem(self, problem_id: str) -> ProblemORM:
        stmt = (
            select(ProblemORM)
            .where(ProblemORM.id == problem_id)
            .options(
                selectinload(ProblemORM.plans),
                selectinload(ProblemORM.agent_states),
                selectinload(ProblemORM.tool_records),
                selectinload(ProblemORM.verifications),
                selectinload(ProblemORM.retries),
                selectinload(ProblemORM.human_reviews),
                selectinload(ProblemORM.final_reports),
            )
        )
        result = await self.session.execute(stmt)
        problem = result.scalar_one_or_none()
        if problem is None:
            raise ProblemNotFoundError(problem_id)
        return problem

    async def update_workflow_state(
        self,
        problem_id: str,
        *,
        execution_id: str | None = None,
        status: WorkflowStatus | None = None,
        current_stage: str | None = None,
        specification: ProblemSpecification | None = None,
        correction_attempts: int | None = None,
        max_correction_attempts: int | None = None,
    ) -> None:
        problem = await self.session.get(ProblemORM, problem_id)
        if problem is None:
            raise ProblemNotFoundError(problem_id)
        if execution_id is not None:
            problem.execution_id = execution_id
        if status is not None:
            problem.status = status.value
        if current_stage is not None:
            problem.current_stage = current_stage
        if specification is not None:
            problem.specification_json = specification.model_dump(mode="json")
        if correction_attempts is not None:
            problem.correction_attempts = correction_attempts
        if max_correction_attempts is not None:
            problem.max_correction_attempts = max_correction_attempts
        problem.updated_at = _utcnow()
        await self.session.commit()

    async def add_execution_plan(
        self, problem_id: str, execution_id: str, plan: ExecutionPlan
    ) -> None:
        self.session.add(
            ExecutionPlanORM(
                problem_id=problem_id,
                execution_id=execution_id,
                plan_json=plan.model_dump(mode="json"),
            )
        )
        await self.session.commit()

    async def add_agent_result(
        self, problem_id: str, execution_id: str, result: AgentExecutionResult
    ) -> None:
        self.session.add(
            AgentStateORM(
                problem_id=problem_id,
                execution_id=execution_id,
                agent_name=result.agent_name,
                stage=result.stage,
                status=result.status,
                result_json=result.model_dump(mode="json"),
                duration_ms=result.duration_ms,
            )
        )
        for tool_rec in result.tool_records:
            self.session.add(
                ToolRecordORM(
                    problem_id=problem_id,
                    execution_id=execution_id,
                    tool_call_id=tool_rec.tool_call_id,
                    tool_name=tool_rec.tool_name,
                    record_json=tool_rec.model_dump(mode="json"),
                    success=1 if tool_rec.success else 0,
                    duration_ms=tool_rec.duration_ms,
                )
            )
        await self.session.commit()

    async def add_verification_result(
        self,
        problem_id: str,
        execution_id: str,
        attempt_number: int,
        report: VerificationReport,
    ) -> None:
        self.session.add(
            VerificationResultORM(
                problem_id=problem_id,
                execution_id=execution_id,
                attempt_number=attempt_number,
                status=report.status.value,
                report_json=report.model_dump(mode="json"),
            )
        )
        await self.session.commit()

    async def add_retry_record(
        self, problem_id: str, execution_id: str, correction: CorrectionRequest
    ) -> None:
        self.session.add(
            RetryHistoryORM(
                problem_id=problem_id,
                execution_id=execution_id,
                attempt_number=correction.attempt_number,
                correction_json=correction.model_dump(mode="json"),
            )
        )
        await self.session.commit()

    async def add_human_review_decision(
        self, problem_id: str, execution_id: str, decision: HumanReviewDecision
    ) -> None:
        self.session.add(
            HumanReviewDecisionORM(
                problem_id=problem_id,
                execution_id=execution_id,
                decision_json=decision.model_dump(mode="json"),
            )
        )
        await self.session.commit()

    async def save_final_report(
        self, problem_id: str, execution_id: str, report: FinalSolutionReport
    ) -> None:
        stmt = select(FinalReportORM).where(FinalReportORM.problem_id == problem_id)
        existing = (await self.session.execute(stmt)).scalar_one_or_none()
        if existing is not None:
            existing.execution_id = execution_id
            existing.verification_status = report.verification_status.value
            existing.report_json = report.model_dump(mode="json")
        else:
            self.session.add(
                FinalReportORM(
                    problem_id=problem_id,
                    execution_id=execution_id,
                    verification_status=report.verification_status.value,
                    report_json=report.model_dump(mode="json"),
                )
            )
        await self.session.commit()

    async def to_status_response(self, problem_id: str) -> ProblemStatusResponse:
        problem = await self.get_problem(problem_id)
        spec = (
            ProblemSpecification.model_validate(problem.specification_json)
            if problem.specification_json
            else None
        )
        latest_verif = (
            VerificationReport.model_validate(problem.verifications[-1].report_json)
            if problem.verifications
            else None
        )
        return ProblemStatusResponse(
            problem_id=problem.id,
            execution_id=problem.execution_id,
            status=WorkflowStatus(problem.status),
            current_stage=problem.current_stage,
            correction_attempts=problem.correction_attempts,
            verification_status=latest_verif.status if latest_verif else None,
            needs_clarification=spec.needs_clarification if spec else False,
            clarification_question=spec.clarification_question if spec else None,
            updated_at=problem.updated_at,
        )

    async def to_detail_response(self, problem_id: str) -> ProblemDetailResponse:
        problem = await self.get_problem(problem_id)
        return ProblemDetailResponse(
            problem_id=problem.id,
            execution_id=problem.execution_id,
            question=problem.question,
            domain_hint=ProblemDomain(problem.domain_hint) if problem.domain_hint else None,
            status=WorkflowStatus(problem.status),
            current_stage=problem.current_stage,
            specification=(
                ProblemSpecification.model_validate(problem.specification_json)
                if problem.specification_json
                else None
            ),
            plan=(
                ExecutionPlan.model_validate(problem.plans[-1].plan_json)
                if problem.plans
                else None
            ),
            agent_timeline=[
                AgentExecutionResult.model_validate(item.result_json)
                for item in sorted(problem.agent_states, key=lambda x: x.id)
            ],
            tool_records=[
                ToolExecutionRecord.model_validate(item.record_json)
                for item in sorted(problem.tool_records, key=lambda x: x.id)
            ],
            verification_report=(
                VerificationReport.model_validate(problem.verifications[-1].report_json)
                if problem.verifications
                else None
            ),
            correction_history=[
                CorrectionRequest.model_validate(item.correction_json)
                for item in sorted(problem.retries, key=lambda x: x.id)
            ],
            human_review=(
                HumanReviewDecision.model_validate(problem.human_reviews[-1].decision_json)
                if problem.human_reviews
                else None
            ),
            final_report=(
                FinalSolutionReport.model_validate(problem.final_reports[-1].report_json)
                if problem.final_reports
                else None
            ),
            created_at=problem.created_at,
            updated_at=problem.updated_at,
        )

    async def get_metrics(self) -> MetricsResponse:
        total = (await self.session.execute(select(func.count(ProblemORM.id)))).scalar() or 0
        completed = (
            await self.session.execute(
                select(func.count(ProblemORM.id)).where(ProblemORM.status == WorkflowStatus.COMPLETED.value)
            )
        ).scalar() or 0
        unresolved = (
            await self.session.execute(
                select(func.count(ProblemORM.id)).where(
                    ProblemORM.status.in_([WorkflowStatus.UNRESOLVED.value, WorkflowStatus.HUMAN_REVIEW.value])
                )
            )
        ).scalar() or 0
        passed = (
            await self.session.execute(
                select(func.count(FinalReportORM.id)).where(
                    FinalReportORM.verification_status == VerificationStatus.PASS.value
                )
            )
        ).scalar() or 0
        failed = (
            await self.session.execute(
                select(func.count(FinalReportORM.id)).where(
                    FinalReportORM.verification_status == VerificationStatus.FAIL.value
                )
            )
        ).scalar() or 0
        avg_retries = (
            await self.session.execute(select(func.avg(ProblemORM.correction_attempts)))
        ).scalar() or 0.0
        avg_dur = (await self.session.execute(select(func.avg(AgentStateORM.duration_ms)))).scalar() or 0.0
        total_tools = (await self.session.execute(select(func.count(ToolRecordORM.id)))).scalar() or 0
        return MetricsResponse(
            total_problems=int(total),
            completed_problems=int(completed),
            passed_verifications=int(passed),
            failed_verifications=int(failed),
            unresolved_problems=int(unresolved),
            average_correction_attempts=round(float(avg_retries), 3),
            average_duration_ms=round(float(avg_dur), 2),
            total_tool_executions=int(total_tools),
        )
