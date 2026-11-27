import json
import time

from pydantic import BaseModel, ConfigDict, Field

from app.agents.solver import PlannedToolCall, SolverSynthesis
from app.schemas.models import (
    AgentExecutionResult,
    ProblemSpecification,
    ToolExecutionRecord,
    VerificationCheck,
    VerificationReport,
    VerificationStatus,
    utc_now,
)
from app.services.gemini_client import LLMGateway
from app.tools.registry import SafeToolRegistry


class VerificationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verification_calls: list[PlannedToolCall] = Field(..., min_length=1)
    dimensional_check_notes: str = ""
    boundary_check_notes: str = ""


_VERIFY_PLAN_SYSTEM = """You are Agent 4 (Independent Verification Agent) in VeriSTEM.
Design independent computational checks to verify the candidate solution WITHOUT reusing the exact same calculation path when practical.
Use tools from: independent_verifier, equation_validator, dimensional_analysis, symbolic_math, numerical_compute, python_sandbox.
Every arguments_json must be a valid JSON object string matching the chosen tool schema."""

_VERIFY_JUDGE_SYSTEM = """You are Agent 4 (Independent Verification Agent) issuing a formal VerificationReport.
Base your judgment STRICTLY on the actual verification tool outputs below:
- If no verification tools succeeded, you MUST return status="INCONCLUSIVE" or "FAIL" (never PASS).
- If any independent check failed or tolerance/dimensions/constraints failed, return status="FAIL".
- Only return status="PASS" when independent computational checks succeeded and confirm the candidate answer."""


async def run_verification_agent(
    *,
    llm: LLMGateway,
    spec: ProblemSpecification,
    synthesis: SolverSynthesis,
    solver_tools: list[ToolExecutionRecord],
    registry: SafeToolRegistry,
) -> tuple[VerificationReport, AgentExecutionResult]:
    started_at = utc_now()
    t0 = time.perf_counter()

    tool_schemas = json.dumps(registry.list_tools(), indent=2)
    plan_prompt = (
        f"ProblemSpecification:\n{spec.model_dump_json(indent=2)}\n\n"
        f"Candidate Solution:\n{synthesis.model_dump_json(indent=2)}\n\n"
        f"Solver Tool History:\n{json.dumps([r.model_dump(mode='json') for r in solver_tools], indent=2)}\n\n"
        f"Tool Schemas:\n{tool_schemas}"
    )

    vplan_resp = await llm.generate_structured(
        system_instruction=_VERIFY_PLAN_SYSTEM,
        prompt=plan_prompt,
        response_model=VerificationPlan,
    )
    vplan = VerificationPlan.model_validate(vplan_resp.parsed)

    records: list[ToolExecutionRecord] = []
    for call in vplan.verification_calls:
        try:
            args = json.loads(call.arguments_json)
            if not isinstance(args, dict):
                raise ValueError("arguments_json must decode to a JSON object")
            rec = registry.execute(call.tool_name, args)
        except Exception as exc:
            rec = ToolExecutionRecord(
                tool_call_id=f"tc_v_err_{call.task_id}",
                tool_name=call.tool_name,
                input_payload={"raw_arguments_json": call.arguments_json},
                output_payload={},
                success=False,
                error_message=str(exc),
                duration_ms=0.0,
            )
        records.append(rec)

    judge_prompt = (
        f"ProblemSpecification:\n{spec.model_dump_json(indent=2)}\n\n"
        f"Candidate Solution:\n{synthesis.model_dump_json(indent=2)}\n\n"
        f"Actual Independent Verification Tool Outputs:\n"
        f"{json.dumps([r.model_dump(mode='json') for r in records], indent=2)}"
    )
    judge_resp = await llm.generate_structured(
        system_instruction=_VERIFY_JUDGE_SYSTEM,
        prompt=judge_prompt,
        response_model=VerificationReport,
    )
    report = VerificationReport.model_validate(judge_resp.parsed)

    # Hard algorithmic safeguard: never accept PASS if zero tool records succeeded
    successful_records = [r for r in records if r.success]
    if report.status == VerificationStatus.PASS and not successful_records:
        report = VerificationReport(
            status=VerificationStatus.INCONCLUSIVE,
            checks=[
                VerificationCheck(
                    check_name="tool_execution_presence",
                    method="deterministic_guard",
                    passed=False,
                    details="No independent verification tool executed successfully.",
                )
            ],
            failure_reasons=["Independent verification tools failed to execute."],
            evidence_summary="Rejected PASS because zero verification tools succeeded.",
        )

    # Enforce explicit failure if any tool returned passed=False or valid=False or consistent=False
    for r in successful_records:
        for flag_key in ("passed", "valid", "consistent"):
            if flag_key in r.output_payload and r.output_payload[flag_key] is False:
                report.status = VerificationStatus.FAIL
                if f"Tool {r.tool_name} reported {flag_key}=False" not in report.failure_reasons:
                    report.failure_reasons.append(f"Tool {r.tool_name} reported {flag_key}=False")

    duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    agent_result = AgentExecutionResult(
        agent_name="IndependentVerificationAgent",
        stage="VERIFICATION",
        status=report.status.value,
        summary=report.evidence_summary,
        structured_output=report.model_dump(mode="json"),
        tool_records=records,
        tokens_input=vplan_resp.tokens_input + judge_resp.tokens_input,
        tokens_output=vplan_resp.tokens_output + judge_resp.tokens_output,
        duration_ms=duration_ms,
        started_at=started_at,
        completed_at=utc_now(),
    )
    return report, agent_result
