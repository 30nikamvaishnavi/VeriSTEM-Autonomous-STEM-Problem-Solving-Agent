import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.exceptions import InvalidStateTransitionError, ProblemNotFoundError
from app.core.rate_limiter import SlidingWindowRateLimiter
from app.core.security import sanitize_problem_text, verify_api_key
from app.database.repository import ProblemRepository
from app.database.session import check_db_health, get_db_session
from app.schemas.models import (
    CreateProblemRequest,
    CreateProblemResponse,
    FinalSolutionReport,
    HealthResponse,
    MetricsResponse,
    ProblemDetailResponse,
    ProblemStatusResponse,
    ResumeProblemRequest,
    SolveProblemRequest,
    SolveProblemResponse,
    WorkflowStatus,
    utc_now,
)
from app.services.workflow_service import WorkflowService

router = APIRouter(prefix="/api/v1", tags=["veristem-v1"])

_settings = get_settings()
_rate_limiter = SlidingWindowRateLimiter(max_requests=_settings.rate_limit_requests_per_minute)
_workflow_service = WorkflowService()


def get_workflow_service() -> WorkflowService:
    return _workflow_service


async def enforce_rate_limit(request: Request) -> None:
    client_host = request.client.host if request.client else "unknown"
    _rate_limiter.check(client_host)


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    settings = get_settings()
    db_ok = await check_db_health()
    gemini_ok = bool(settings.gemini_api_key and settings.gemini_api_key != "MY_GEMINI_API_KEY")
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        version=settings.app_version,
        database_connected=db_ok,
        gemini_configured=gemini_ok,
        timestamp=utc_now(),
    )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)],
)
async def get_metrics(session: AsyncSession = Depends(get_db_session)) -> MetricsResponse:
    repo = ProblemRepository(session)
    return await repo.get_metrics()


@router.post(
    "/problems",
    response_model=CreateProblemResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)],
)
async def create_problem(
    body: CreateProblemRequest,
    session: AsyncSession = Depends(get_db_session),
) -> CreateProblemResponse:
    settings = get_settings()
    cleaned_q = sanitize_problem_text(body.question, max_chars=settings.max_problem_chars)
    problem_id = f"prob_{uuid.uuid4().hex[:12]}"
    repo = ProblemRepository(session)
    record = await repo.create_problem(
        problem_id=problem_id,
        question=cleaned_q,
        domain_hint=body.domain_hint,
        context=body.context,
    )
    return CreateProblemResponse(
        problem_id=record.id,
        status=WorkflowStatus(record.status),
        created_at=record.created_at,
    )


@router.post(
    "/problems/{problem_id}/solve",
    response_model=SolveProblemResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)],
)
async def solve_problem(
    problem_id: str,
    body: SolveProblemRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db_session),
    workflow: WorkflowService = Depends(get_workflow_service),
) -> SolveProblemResponse:
    repo = ProblemRepository(session)
    problem = await repo.get_problem(problem_id)

    if body.idempotency_key:
        existing = await repo.get_idempotent_execution(body.idempotency_key)
        if existing is not None:
            return SolveProblemResponse(
                problem_id=existing.problem_id,
                execution_id=existing.execution_id,
                status=WorkflowStatus(problem.status),
                message="Idempotent solve request already registered.",
            )

    if problem.status == WorkflowStatus.RUNNING.value:
        raise InvalidStateTransitionError("Problem workflow is already running")

    execution_id = f"exec_{uuid.uuid4().hex[:12]}"
    if body.idempotency_key:
        await repo.save_idempotent_execution(body.idempotency_key, problem_id, execution_id)

    if body.run_async:
        background_tasks.add_task(
            workflow.execute_problem,
            problem_id=problem_id,
            execution_id=execution_id,
            max_correction_attempts=body.max_correction_attempts,
        )
        return SolveProblemResponse(
            problem_id=problem_id,
            execution_id=execution_id,
            status=WorkflowStatus.RUNNING,
            message="Multi-agent verification workflow started in background.",
        )

    await workflow.execute_problem(
        problem_id=problem_id,
        execution_id=execution_id,
        max_correction_attempts=body.max_correction_attempts,
    )
    refreshed = await repo.get_problem(problem_id)
    return SolveProblemResponse(
        problem_id=problem_id,
        execution_id=execution_id,
        status=WorkflowStatus(refreshed.status),
        message="Multi-agent verification workflow completed synchronously.",
    )


@router.get(
    "/problems/{problem_id}",
    response_model=ProblemDetailResponse,
    dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)],
)
async def get_problem_detail(
    problem_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> ProblemDetailResponse:
    repo = ProblemRepository(session)
    return await repo.to_detail_response(problem_id)


@router.get(
    "/problems/{problem_id}/status",
    response_model=ProblemStatusResponse,
    dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)],
)
async def get_problem_status(
    problem_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> ProblemStatusResponse:
    repo = ProblemRepository(session)
    return await repo.to_status_response(problem_id)


@router.get(
    "/problems/{problem_id}/report",
    response_model=FinalSolutionReport,
    dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)],
)
async def get_problem_report(
    problem_id: str,
    format: str = Query(default="json", pattern="^(json|markdown)$"),
    session: AsyncSession = Depends(get_db_session),
) -> FinalSolutionReport | Response:
    repo = ProblemRepository(session)
    detail = await repo.to_detail_response(problem_id)
    if detail.final_report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Final report for '{problem_id}' is not ready yet (current status: {detail.status.value})",
        )
    if format == "markdown":
        return Response(content=detail.final_report.explanation_markdown, media_type="text/markdown")
    return detail.final_report


@router.post(
    "/problems/{problem_id}/resume",
    response_model=SolveProblemResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)],
)
async def resume_problem(
    problem_id: str,
    body: ResumeProblemRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db_session),
    workflow: WorkflowService = Depends(get_workflow_service),
) -> SolveProblemResponse:
    repo = ProblemRepository(session)
    problem = await repo.get_problem(problem_id)
    if problem.status not in (WorkflowStatus.NEEDS_CLARIFICATION.value, WorkflowStatus.HUMAN_REVIEW.value):
        raise InvalidStateTransitionError(
            f"Problem '{problem_id}' cannot be resumed from status '{problem.status}'"
        )
    background_tasks.add_task(workflow.resume_problem, problem_id=problem_id, request=body)
    return SolveProblemResponse(
        problem_id=problem_id,
        execution_id=problem.execution_id or f"exec_{uuid.uuid4().hex[:12]}",
        status=WorkflowStatus.RUNNING,
        message="Resumed workflow from persisted LangGraph checkpoint.",
    )
