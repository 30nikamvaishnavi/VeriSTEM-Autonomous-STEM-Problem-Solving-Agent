import json
import time
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.agents.solver import SolverSynthesis
from app.schemas.models import (
    AgentExecutionResult,
    CorrectionRequest,
    FinalSolutionReport,
    ProblemSpecification,
    ToolExecutionRecord,
    VerificationReport,
    utc_now,
)
from app.services.gemini_client import LLMGateway


class NarrativeExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    explanation_markdown: str
    confidence_limitations: list[str] = Field(default_factory=list)


_SYSTEM_PROMPT = """You are Agent 6 (Explanation and Reporting Agent) in VeriSTEM.
Generate a clear, rigorous scientific explanation in Markdown using LaTeX equations ($...$ and $$...$$).
Strictly separate:
1. Modeling Assumptions (unstated or stated conditions)
2. Verified Computational Evidence (only from actual tool outputs and verification checks)
3. Limitations and Edge Cases
Never present LLM subjective confidence as mathematical proof."""


async def run_reporting_agent(
    *,
    llm: LLMGateway,
    problem_id: str,
    execution_id: str,
    question: str,
    spec: ProblemSpecification,
    synthesis: SolverSynthesis,
    verification: VerificationReport,
    corrections: list[CorrectionRequest],
    tool_history: list[ToolExecutionRecord],
    started_at: datetime,
) -> tuple[FinalSolutionReport, AgentExecutionResult]:
    agent_started = utc_now()
    t0 = time.perf_counter()

    prompt = (
        f"Question:\n{question}\n\n"
        f"ProblemSpecification:\n{spec.model_dump_json(indent=2)}\n\n"
        f"Solution Synthesis:\n{synthesis.model_dump_json(indent=2)}\n\n"
        f"VerificationReport:\n{verification.model_dump_json(indent=2)}\n\n"
        f"Correction History:\n{json.dumps([c.model_dump(mode='json') for c in corrections], indent=2)}"
    )
    resp = await llm.generate_structured(
        system_instruction=_SYSTEM_PROMPT,
        prompt=prompt,
        response_model=NarrativeExplanation,
    )
    narrative = NarrativeExplanation.model_validate(resp.parsed)

    final_report = FinalSolutionReport(
        problem_id=problem_id,
        execution_id=execution_id,
        original_question=question,
        detected_domain=spec.domain,
        extracted_variables=spec.variables,
        assumptions=list(dict.fromkeys([*spec.assumptions, *synthesis.assumptions_applied])),
        solution_methodology=synthesis.solution_methodology,
        equations=synthesis.equations,
        intermediate_results=synthesis.intermediate_results,
        final_answer=synthesis.final_answer,
        numeric_value=synthesis.numeric_value,
        units=synthesis.units,
        verification_status=verification.status,
        verification_evidence=verification.checks,
        correction_attempts=len(corrections),
        correction_history=corrections,
        tool_execution_history=tool_history,
        confidence_limitations=narrative.confidence_limitations,
        explanation_markdown=narrative.explanation_markdown,
        started_at=started_at,
        completed_at=utc_now(),
    )

    duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    agent_result = AgentExecutionResult(
        agent_name="ExplanationReportingAgent",
        stage="REPORTING",
        status="COMPLETED",
        summary=f"Generated verified report ({verification.status.value}) with {len(tool_history)} tool records.",
        structured_output={"final_answer": final_report.final_answer, "verification_status": verification.status.value},
        tokens_input=resp.tokens_input,
        tokens_output=resp.tokens_output,
        duration_ms=duration_ms,
        started_at=agent_started,
        completed_at=utc_now(),
    )
    return final_report, agent_result
