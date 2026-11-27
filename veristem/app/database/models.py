from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ProblemORM(Base):
    __tablename__ = "problems"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    execution_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    domain_hint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    context_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="PENDING", index=True)
    current_stage: Mapped[str] = mapped_column(String(64), default="CREATED")
    specification_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    correction_attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_correction_attempts: Mapped[int] = mapped_column(Integer, default=3)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    plans: Mapped[list["ExecutionPlanORM"]] = relationship(back_populates="problem", cascade="all, delete-orphan")
    agent_states: Mapped[list["AgentStateORM"]] = relationship(back_populates="problem", cascade="all, delete-orphan")
    tool_records: Mapped[list["ToolRecordORM"]] = relationship(back_populates="problem", cascade="all, delete-orphan")
    verifications: Mapped[list["VerificationResultORM"]] = relationship(back_populates="problem", cascade="all, delete-orphan")
    retries: Mapped[list["RetryHistoryORM"]] = relationship(back_populates="problem", cascade="all, delete-orphan")
    human_reviews: Mapped[list["HumanReviewDecisionORM"]] = relationship(back_populates="problem", cascade="all, delete-orphan")
    final_reports: Mapped[list["FinalReportORM"]] = relationship(back_populates="problem", cascade="all, delete-orphan")


class ExecutionPlanORM(Base):
    __tablename__ = "execution_plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    problem_id: Mapped[str] = mapped_column(ForeignKey("problems.id", ondelete="CASCADE"), index=True)
    execution_id: Mapped[str] = mapped_column(String(64), index=True)
    plan_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    problem: Mapped[ProblemORM] = relationship(back_populates="plans")


class AgentStateORM(Base):
    __tablename__ = "agent_states"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    problem_id: Mapped[str] = mapped_column(ForeignKey("problems.id", ondelete="CASCADE"), index=True)
    execution_id: Mapped[str] = mapped_column(String(64), index=True)
    agent_name: Mapped[str] = mapped_column(String(64), nullable=False)
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    problem: Mapped[ProblemORM] = relationship(back_populates="agent_states")


class ToolRecordORM(Base):
    __tablename__ = "tool_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    problem_id: Mapped[str] = mapped_column(ForeignKey("problems.id", ondelete="CASCADE"), index=True)
    execution_id: Mapped[str] = mapped_column(String(64), index=True)
    tool_call_id: Mapped[str] = mapped_column(String(64), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(64), nullable=False)
    record_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    success: Mapped[int] = mapped_column(Integer, default=1)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    problem: Mapped[ProblemORM] = relationship(back_populates="tool_records")


class VerificationResultORM(Base):
    __tablename__ = "verification_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    problem_id: Mapped[str] = mapped_column(ForeignKey("problems.id", ondelete="CASCADE"), index=True)
    execution_id: Mapped[str] = mapped_column(String(64), index=True)
    attempt_number: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    report_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    problem: Mapped[ProblemORM] = relationship(back_populates="verifications")


class RetryHistoryORM(Base):
    __tablename__ = "retry_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    problem_id: Mapped[str] = mapped_column(ForeignKey("problems.id", ondelete="CASCADE"), index=True)
    execution_id: Mapped[str] = mapped_column(String(64), index=True)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    correction_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    problem: Mapped[ProblemORM] = relationship(back_populates="retries")


class HumanReviewDecisionORM(Base):
    __tablename__ = "human_review_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    problem_id: Mapped[str] = mapped_column(ForeignKey("problems.id", ondelete="CASCADE"), index=True)
    execution_id: Mapped[str] = mapped_column(String(64), index=True)
    decision_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    problem: Mapped[ProblemORM] = relationship(back_populates="human_reviews")


class FinalReportORM(Base):
    __tablename__ = "final_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    problem_id: Mapped[str] = mapped_column(ForeignKey("problems.id", ondelete="CASCADE"), unique=True, index=True)
    execution_id: Mapped[str] = mapped_column(String(64), index=True)
    verification_status: Mapped[str] = mapped_column(String(32), nullable=False)
    report_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    problem: Mapped[ProblemORM] = relationship(back_populates="final_reports")


class IdempotencyRecordORM(Base):
    __tablename__ = "idempotency_keys"

    idempotency_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    problem_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    execution_id: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
