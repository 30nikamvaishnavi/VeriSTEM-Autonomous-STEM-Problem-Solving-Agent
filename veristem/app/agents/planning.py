import json
import time

from app.schemas.models import (
    AgentExecutionResult,
    ExecutionPlan,
    ProblemSpecification,
    utc_now,
)
from app.services.gemini_client import LLMGateway
from app.tools.registry import SafeToolRegistry

_SYSTEM_PROMPT = """You are Agent 2 (Planning Agent) in VeriSTEM.
Given a typed ProblemSpecification and the available computational tools:
1. Decompose the problem into dependency-aware ExecutionTasks.
2. Select exact tool names from the available registry: symbolic_math, numerical_compute, scientific_opt, statistical_analysis, dimensional_analysis, equation_validator, independent_verifier, python_sandbox.
3. Specify the mathematical/scientific method and expected output for each step.
4. Define a concrete independent verification_strategy using a separate computational method when practical.
Return strictly valid JSON matching the ExecutionPlan schema."""


async def run_planning_agent(
    *,
    llm: LLMGateway,
    spec: ProblemSpecification,
    registry: SafeToolRegistry,
) -> tuple[ExecutionPlan, AgentExecutionResult]:
    started_at = utc_now()
    t0 = time.perf_counter()
    tools_desc = json.dumps(
        [{"name": t["name"], "description": t["description"]} for t in registry.list_tools()],
        indent=2,
    )
    prompt = (
        f"ProblemSpecification:\n{spec.model_dump_json(indent=2)}\n\n"
        f"Available Tools:\n{tools_desc}"
    )
    resp = await llm.generate_structured(
        system_instruction=_SYSTEM_PROMPT,
        prompt=prompt,
        response_model=ExecutionPlan,
    )
    plan = ExecutionPlan.model_validate(resp.parsed)
    duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    agent_result = AgentExecutionResult(
        agent_name="PlanningAgent",
        stage="PLANNING",
        status="COMPLETED",
        summary=f"Created {len(plan.tasks)}-step execution plan using tools: {', '.join(plan.selected_tools)}.",
        structured_output=plan.model_dump(mode="json"),
        tokens_input=resp.tokens_input,
        tokens_output=resp.tokens_output,
        duration_ms=duration_ms,
        started_at=started_at,
        completed_at=utc_now(),
    )
    return plan, agent_result
