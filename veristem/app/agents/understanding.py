import time

from app.schemas.models import (
    AgentExecutionResult,
    ProblemDomain,
    ProblemSpecification,
    utc_now,
)
from app.services.gemini_client import LLMGateway

_SYSTEM_PROMPT = """You are Agent 1 (Problem Understanding Agent) in VeriSTEM.
Analyze the user's natural-language STEM problem carefully.
1. Identify the exact domain (mathematics, physics, quantitative_finance, or general_stem) and problem_type.
2. Extract every mathematical/physical/financial variable with symbol, name, numeric or symbolic value, SI/domain unit, constraints, and description.
3. Extract explicit equations, constraints, and stated or standard modeling assumptions.
4. Determine whether essential information is missing such that the problem cannot be solved uniquely without clarification.
5. Only set needs_clarification=true if critical numerical parameters or boundary conditions are genuinely absent.
Return strictly valid JSON matching the schema."""


async def run_understanding_agent(
    *,
    llm: LLMGateway,
    question: str,
    domain_hint: ProblemDomain | None = None,
    clarification_response: str | None = None,
) -> tuple[ProblemSpecification, AgentExecutionResult]:
    started_at = utc_now()
    t0 = time.perf_counter()

    prompt_parts = [f"Problem:\n{question}"]
    if domain_hint is not None:
        prompt_parts.append(f"User domain hint: {domain_hint.value}")
    if clarification_response:
        prompt_parts.append(f"Additional user clarification:\n{clarification_response}")

    resp = await llm.generate_structured(
        system_instruction=_SYSTEM_PROMPT,
        prompt="\n\n".join(prompt_parts),
        response_model=ProblemSpecification,
    )
    spec = ProblemSpecification.model_validate(resp.parsed)
    duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    agent_result = AgentExecutionResult(
        agent_name="ProblemUnderstandingAgent",
        stage="UNDERSTANDING",
        status="NEEDS_CLARIFICATION" if spec.needs_clarification else "COMPLETED",
        summary=f"Classified as {spec.domain.value} ({spec.problem_type}) with {len(spec.variables)} variables.",
        structured_output=spec.model_dump(mode="json"),
        tokens_input=resp.tokens_input,
        tokens_output=resp.tokens_output,
        duration_ms=duration_ms,
        started_at=started_at,
        completed_at=utc_now(),
    )
    return spec, agent_result
