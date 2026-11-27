from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ProblemDomain(str, Enum):
    MATHEMATICS = "mathematics"
    PHYSICS = "physics"
    QUANTITATIVE_FINANCE = "quantitative_finance"
    GENERAL_STEM = "general_stem"


class VerificationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


class WorkflowStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    COMPLETED = "COMPLETED"
    UNRESOLVED = "UNRESOLVED"
    FAILED = "FAILED"


class MathematicalVariable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=128)
    value: float | str | None = None
    unit: str = Field(default="dimensionless", max_length=64)
    constraints: list[str] = Field(default_factory=list)
    description: str = Field(default="", max_length=512)


class ProblemSpecification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: ProblemDomain
    problem_type: str = Field(..., min_length=1, max_length=128)
    objective: str = Field(..., min_length=1, max_length=1024)
    variables: list[MathematicalVariable] = Field(default_factory=list)
    equations: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str | None = None


class ExecutionTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(..., min_length=1, max_length=64)
    title: str = Field(..., min_length=1, max_length=256)
    description: str = Field(..., min_length=1, max_length=1024)
    method: str = Field(..., min_length=1, max_length=256)
    tool_name: str = Field(..., min_length=1, max_length=64)
    depends_on: list[str] = Field(default_factory=list)
    expected_output: str = Field(default="", max_length=512)


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    methodology: str = Field(..., min_length=1, max_length=512)
    selected_tools: list[str] = Field(default_factory=list)
    tasks: list[ExecutionTask] = Field(..., min_length=1)
    verification_strategy: str = Field(..., min_length=1, max_length=1024)


class ToolExecutionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    tool_name: str
    input_payload: dict[str, Any]
    output_payload: dict[str, Any] = Field(default_factory=dict)
    success: bool
    error_message: str | None = None
    duration_ms: float = Field(..., ge=0.0)
    executed_at: datetime = Field(default_factory=utc_now)


class AgentExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_name: str
    stage: str
    status: str
    summary: str
    structured_output: dict[str, Any] = Field(default_factory=dict)
    tool_records: list[ToolExecutionRecord] = Field(default_factory=list)
    tokens_input: int = Field(default=0, ge=0)
    tokens_output: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0, ge=0.0)
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime = Field(default_factory=utc_now)


class VerificationCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_name: str
    method: str
    passed: bool
    expected_value: str | None = None
    observed_value: str | None = None
    relative_error: float | None = Field(default=None, ge=0.0)
    tolerance: float | None = Field(default=None, ge=0.0)
    details: str = ""


class VerificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: VerificationStatus
    checks: list[VerificationCheck] = Field(default_factory=list)
    independent_recalculation: str | None = None
    dimensional_consistency_passed: bool | None = None
    boundary_conditions_passed: bool | None = None
    numerical_stability_passed: bool | None = None
    failure_reasons: list[str] = Field(default_factory=list)
    evidence_summary: str
    verified_at: datetime = Field(default_factory=utc_now)


class CorrectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt_number: int = Field(..., ge=1, le=5)
    root_cause: str
    error_category: str
    corrective_instructions: str
    tools_to_reexecute: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class HumanReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer_action: str = Field(..., pattern="^(approve|clarify|override|reject)$")
    notes: str = Field(default="", max_length=2000)
    clarification_response: str | None = None
    override_answer: str | None = None
    decided_at: datetime = Field(default_factory=utc_now)


class FinalSolutionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem_id: str
    execution_id: str
    original_question: str
    detected_domain: ProblemDomain
    extracted_variables: list[MathematicalVariable] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    solution_methodology: str
    equations: list[str] = Field(default_factory=list)
    intermediate_results: list[dict[str, Any]] = Field(default_factory=list)
    final_answer: str
    numeric_value: float | None = None
    units: str = "dimensionless"
    verification_status: VerificationStatus
    verification_evidence: list[VerificationCheck] = Field(default_factory=list)
    correction_attempts: int = Field(default=0, ge=0)
    correction_history: list[CorrectionRequest] = Field(default_factory=list)
    tool_execution_history: list[ToolExecutionRecord] = Field(default_factory=list)
    confidence_limitations: list[str] = Field(default_factory=list)
    explanation_markdown: str
    started_at: datetime
    completed_at: datetime = Field(default_factory=utc_now)


# --- API Request / Response Contracts ---


class CreateProblemRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=5, max_length=8000)
    domain_hint: ProblemDomain | None = None
    context: dict[str, Any] = Field(default_factory=dict)

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        cleaned = v.strip()
        if len(cleaned) < 5:
            raise ValueError("Question must contain at least 5 non-whitespace characters")
        return cleaned


class CreateProblemResponse(BaseModel):
    problem_id: str
    status: WorkflowStatus
    created_at: datetime


class SolveProblemRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idempotency_key: str | None = Field(default=None, max_length=128)
    max_correction_attempts: int = Field(default=3, ge=1, le=5)
    run_async: bool = True


class SolveProblemResponse(BaseModel):
    problem_id: str
    execution_id: str
    status: WorkflowStatus
    message: str


class ResumeProblemRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str = Field(..., pattern="^(approve|clarify|override|reject)$")
    clarification_response: str | None = Field(default=None, max_length=4000)
    notes: str = Field(default="", max_length=2000)
    override_answer: str | None = Field(default=None, max_length=1000)


class ProblemStatusResponse(BaseModel):
    problem_id: str
    execution_id: str | None
    status: WorkflowStatus
    current_stage: str
    correction_attempts: int
    verification_status: VerificationStatus | None = None
    needs_clarification: bool = False
    clarification_question: str | None = None
    updated_at: datetime


class ProblemDetailResponse(BaseModel):
    problem_id: str
    execution_id: str | None
    question: str
    domain_hint: ProblemDomain | None
    status: WorkflowStatus
    current_stage: str
    specification: ProblemSpecification | None = None
    plan: ExecutionPlan | None = None
    agent_timeline: list[AgentExecutionResult] = Field(default_factory=list)
    tool_records: list[ToolExecutionRecord] = Field(default_factory=list)
    verification_report: VerificationReport | None = None
    correction_history: list[CorrectionRequest] = Field(default_factory=list)
    human_review: HumanReviewDecision | None = None
    final_report: FinalSolutionReport | None = None
    created_at: datetime
    updated_at: datetime


class HealthResponse(BaseModel):
    status: str
    version: str
    database_connected: bool
    gemini_configured: bool
    timestamp: datetime


class MetricsResponse(BaseModel):
    total_problems: int
    completed_problems: int
    passed_verifications: int
    failed_verifications: int
    unresolved_problems: int
    average_correction_attempts: float
    average_duration_ms: float
    total_tool_executions: int


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    correlation_id: str
    details: dict[str, Any] | None = None
