import time

from app.agents.solver import SolverSynthesis
from app.schemas.models import (
    AgentExecutionResult,
    CorrectionRequest,
    ProblemSpecification,
    VerificationReport,
    utc_now,
)
from app.services.gemini_client import LLMGateway

_SYSTEM_PROMPT = """You are Agent 5 (Critic and Self-Correction Agent) in VeriSTEM.
Analyze the failed or inconclusive VerificationReport and identify the exact mathematical, unit, boundary, or tool-parameter error.
Provide precise corrective_instructions and list the tools_to_reexecute so the STEM Solver Agent can fix the defect."""


async def run_critic_agent(
    *,
    llm: LLMGateway,
    attempt_number: int,
    spec: ProblemSpecification,
    synthesis: SolverSynthesis,
    verification: VerificationReport,
) -> tuple[CorrectionRequest, AgentExecutionResult]:
    started_at = utc_now()
    t0 = time.perf_counter()

    prompt = (
        f"Attempt Number: {attempt_number}\n\n"
        f"ProblemSpecification:\n{spec.model_dump_json(indent=2)}\n\n"
        f"Failed Candidate Solution:\n{synthesis.model_dump_json(indent=2)}\n\n"
        f"VerificationReport:\n{verification.model_dump_json(indent=2)}"
    )
    resp = await llm.generate_structured(
        system_instruction=_SYSTEM_PROMPT,
        prompt=prompt,
        response_model=CorrectionRequest,
    )
    correction = CorrectionRequest.model_validate(resp.parsed)
    correction.attempt_number = attempt_number

    duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    agent_result = AgentExecutionResult(
        agent_name="CriticSelfCorrectionAgent",
        stage="CRITIC",
        status="CORRECTION_REQUESTED",
        summary=f"Attempt {attempt_number}: {correction.error_category} — {correction.root_cause}",
        structured_output=correction.model_dump(mode="json"),
        tokens_input=resp.tokens_input,
        tokens_output=resp.tokens_output,
        duration_ms=duration_ms,
        started_at=started_at,
        completed_at=utc_now(),
    )
    return correction, agent_result
