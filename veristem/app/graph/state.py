from typing import Any, TypedDict


class VeriSTEMState(TypedDict, total=False):
    problem_id: str
    execution_id: str
    question: str
    domain_hint: str | None
    clarification_response: str | None
    started_at: str
    current_stage: str
    workflow_status: str
    max_correction_attempts: int
    correction_attempts: int

    # Structured stage artifacts (JSON-serializable dicts validated by Pydantic at boundaries)
    specification: dict[str, Any] | None
    plan: dict[str, Any] | None
    solver_synthesis: dict[str, Any] | None
    verification_report: dict[str, Any] | None
    latest_correction: dict[str, Any] | None
    correction_history: list[dict[str, Any]]
    tool_records: list[dict[str, Any]]
    agent_timeline: list[dict[str, Any]]
    human_review_decision: dict[str, Any] | None
    final_report: dict[str, Any] | None
    error_message: str | None
