import json
from pathlib import Path

import pytest

from app.agents.reporter import NarrativeExplanation
from app.agents.solver import PlannedToolCall, SolverSynthesis, SolverToolCallPlan
from app.agents.verifier import VerificationPlan
from app.evaluation.runner import BenchmarkItem, evaluate_configuration, write_evaluation_reports
from app.schemas.models import (
    ExecutionPlan,
    ExecutionTask,
    ProblemDomain,
    ProblemSpecification,
    VerificationReport,
    VerificationStatus,
)
from tests.conftest import MockLLMGateway


@pytest.mark.asyncio
async def test_e2e_evaluation_pipeline_generates_reports(tmp_path: Path) -> None:
    item = BenchmarkItem(
        id="e2e_01",
        domain="mathematics",
        category="calculus",
        difficulty="standard",
        question="Evaluate the definite integral of x^2 * exp(-x) from x = 0 to x = 2.",
        expected_answer="2 - 10*exp(-2)",
        expected_numeric=0.6466471676338731,
        expected_unit="dimensionless",
        tolerance=1e-5,
        verification_method="scipy_quad_adaptive_quadrature",
    )

    mock_llm = MockLLMGateway(
        responses={
            ProblemSpecification: [
                ProblemSpecification(
                    domain=ProblemDomain.MATHEMATICS,
                    problem_type="calculus",
                    objective="Evaluate definite integral",
                )
            ],
            ExecutionPlan: [
                ExecutionPlan(
                    methodology="SymPy integration + SciPy quadrature",
                    selected_tools=["symbolic_math", "independent_verifier"],
                    tasks=[ExecutionTask(task_id="t1", title="Integrate", description="Int", method="sympy", tool_name="symbolic_math")],
                    verification_strategy="Adaptive quadrature",
                )
            ],
            SolverToolCallPlan: [
                SolverToolCallPlan(
                    reasoning_strategy="SymPy definite integral",
                    tool_calls=[
                        PlannedToolCall(
                            task_id="t1",
                            tool_name="symbolic_math",
                            arguments_json=json.dumps(
                                {
                                    "operation": "integrate",
                                    "expression": "x^2 * exp(-x)",
                                    "variable": "x",
                                    "lower_limit": "0",
                                    "upper_limit": "2",
                                }
                            ),
                        )
                    ],
                )
            ],
            SolverSynthesis: [
                SolverSynthesis(
                    solution_methodology="SymPy integration",
                    final_answer="2 - 10*exp(-2)",
                    numeric_value=0.6466471676338731,
                )
            ],
            VerificationPlan: [
                VerificationPlan(
                    verification_calls=[
                        PlannedToolCall(
                            task_id="v1",
                            tool_name="independent_verifier",
                            arguments_json=json.dumps(
                                {
                                    "verification_type": "integral_quadrature",
                                    "expression": "x^2 * exp(-x)",
                                    "variable": "x",
                                    "lower_limit": 0.0,
                                    "upper_limit": 2.0,
                                    "candidate_value": 0.6466471676338731,
                                    "tolerance": 1e-6,
                                }
                            ),
                        )
                    ]
                )
            ],
            VerificationReport: [
                VerificationReport(status=VerificationStatus.PASS, evidence_summary="Adaptive quadrature matches")
            ],
            NarrativeExplanation: [
                NarrativeExplanation(explanation_markdown="Exact value $2 - 10e^{-2} \\approx 0.646647$.")
            ],
        }
    )

    res = await evaluate_configuration(items=[item], mode="full_multi_agent", llm=mock_llm)
    assert res["summary"]["accuracy"] == 1.0
    assert res["summary"]["false_acceptance_rate"] == 0.0

    jpath, cpath = write_evaluation_reports([res], tmp_path)
    assert jpath.exists()
    assert cpath.exists()
