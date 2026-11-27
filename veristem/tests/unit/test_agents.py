import json

import pytest

from app.agents.solver import PlannedToolCall, SolverSynthesis, SolverToolCallPlan
from app.agents.verifier import VerificationPlan, run_verification_agent
from app.schemas.models import (
    ProblemDomain,
    ProblemSpecification,
    VerificationCheck,
    VerificationReport,
    VerificationStatus,
)
from app.tools.registry import SafeToolRegistry
from tests.conftest import MockLLMGateway


@pytest.mark.asyncio
async def test_verifier_rejects_pass_when_all_tools_fail() -> None:
    spec = ProblemSpecification(
        domain=ProblemDomain.MATHEMATICS,
        problem_type="calculus",
        objective="Integrate x^2 on [0, 3]",
    )
    synthesis = SolverSynthesis(
        solution_methodology="Symbolic integration",
        equations=["int_0^3 x^2 dx = 9"],
        final_answer="9",
        numeric_value=9.0,
    )
    mock_llm = MockLLMGateway(
        responses={
            VerificationPlan: [
                VerificationPlan(
                    verification_calls=[
                        PlannedToolCall(
                            task_id="v1",
                            tool_name="nonexistent_tool",
                            arguments_json=json.dumps({"x": 1}),
                        )
                    ]
                )
            ],
            VerificationReport: [
                VerificationReport(
                    status=VerificationStatus.PASS,
                    checks=[VerificationCheck(check_name="dummy", method="llm", passed=True, details="ok")],
                    evidence_summary="Claimed PASS without working tool",
                )
            ],
        }
    )

    report, _ = await run_verification_agent(
        llm=mock_llm,
        spec=spec,
        synthesis=synthesis,
        solver_tools=[],
        registry=SafeToolRegistry(),
    )
    assert report.status == VerificationStatus.INCONCLUSIVE
