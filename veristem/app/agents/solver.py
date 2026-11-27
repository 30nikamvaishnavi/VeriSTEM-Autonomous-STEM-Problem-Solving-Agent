import json
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.models import (
    AgentExecutionResult,
    CorrectionRequest,
    ExecutionPlan,
    ProblemSpecification,
    ToolExecutionRecord,
    utc_now,
)
from app.services.gemini_client import LLMGateway
from app.tools.registry import SafeToolRegistry


class PlannedToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    tool_name: str
    arguments_json: str = Field(..., description="JSON-encoded arguments object matching the tool schema")


class SolverToolCallPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning_strategy: str
    equations_used: list[str] = Field(default_factory=list)
    tool_calls: list[PlannedToolCall] = Field(..., min_length=1)


class SolverSynthesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    solution_methodology: str
    equations: list[str] = Field(default_factory=list)
    intermediate_results: list[dict[str, Any]] = Field(default_factory=list)
    final_answer: str
    numeric_value: float | None = None
    units: str = "dimensionless"
    assumptions_applied: list[str] = Field(default_factory=list)


_TOOL_PLAN_SYSTEM = """You are Agent 3 (STEM Solver Agent) in VeriSTEM.
Generate exact computational tool calls to solve the problem rigorously.
CRITICAL RULE: Never invent or guess computational outputs. Every calculation must be executed through the registered tools.
Available tools and their JSON schemas are provided. For each PlannedToolCall, set arguments_json to a valid JSON object string matching that tool's schema."""

_SYNTHESIS_SYSTEM = """You are Agent 3 (STEM Solver Agent) synthesizing verified computational tool outputs.
Use ONLY the actual tool execution outputs provided below. Do NOT fabricate or alter numeric values from tool results.
Extract the exact final answer, numeric_value (if applicable), units, equations, and intermediate results."""


async def run_solver_agent(
    *,
    llm: LLMGateway,
    spec: ProblemSpecification,
    plan: ExecutionPlan,
    registry: SafeToolRegistry,
    correction: CorrectionRequest | None = None,
    tool_cache: dict[str, ToolExecutionRecord] | None = None,
) -> tuple[SolverSynthesis, AgentExecutionResult]:
    started_at = utc_now()
    t0 = time.perf_counter()

    tool_schemas = json.dumps(registry.list_tools(), indent=2)
    correction_block = (
        f"\n\nPREVIOUS VERIFICATION FAILED — APPLY CORRECTION:\n{correction.model_dump_json(indent=2)}"
        if correction
        else ""
    )

    plan_prompt = (
        f"ProblemSpecification:\n{spec.model_dump_json(indent=2)}\n\n"
        f"ExecutionPlan:\n{plan.model_dump_json(indent=2)}\n\n"
        f"Tool Schemas:\n{tool_schemas}{correction_block}"
    )

    plan_resp = await llm.generate_structured(
        system_instruction=_TOOL_PLAN_SYSTEM,
        prompt=plan_prompt,
        response_model=SolverToolCallPlan,
    )
    tool_plan = SolverToolCallPlan.model_validate(plan_resp.parsed)

    cache = tool_cache if tool_cache is not None else {}
    records = await registry.execute_parallel_with_cache(
        [(c.task_id, c.tool_name, c.arguments_json) for c in tool_plan.tool_calls],
        cache,
    )

    synth_prompt = (
        f"ProblemSpecification:\n{spec.model_dump_json(indent=2)}\n\n"
        f"Solver Strategy:\n{tool_plan.reasoning_strategy}\n\n"
        f"Actual Executed Tool Records:\n"
        f"{json.dumps([r.model_dump(mode='json') for r in records], indent=2)}"
    )
    synth_resp = await llm.generate_structured(
        system_instruction=_SYNTHESIS_SYSTEM,
        prompt=synth_prompt,
        response_model=SolverSynthesis,
    )
    synthesis = SolverSynthesis.model_validate(synth_resp.parsed)

    duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    agent_result = AgentExecutionResult(
        agent_name="STEMSolverAgent",
        stage="SOLVING",
        status="COMPLETED",
        summary=f"Executed {len(records)} tool call(s); candidate answer: {synthesis.final_answer}",
        structured_output=synthesis.model_dump(mode="json"),
        tool_records=records,
        tokens_input=plan_resp.tokens_input + synth_resp.tokens_input,
        tokens_output=plan_resp.tokens_output + synth_resp.tokens_output,
        duration_ms=duration_ms,
        started_at=started_at,
        completed_at=utc_now(),
    )
    return synthesis, agent_result
