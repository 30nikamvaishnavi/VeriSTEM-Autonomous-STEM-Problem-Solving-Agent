import json
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.agents.reporter import NarrativeExplanation
from app.agents.solver import PlannedToolCall, SolverSynthesis, SolverToolCallPlan
from app.agents.verifier import VerificationPlan
from app.graph.state import VeriSTEMState
from app.graph.workflow import build_veristem_graph
from app.schemas.models import (
    CorrectionRequest,
    ExecutionPlan,
    ExecutionTask,
    ProblemDomain,
    ProblemSpecification,
    VerificationCheck,
    VerificationReport,
    VerificationStatus,
    WorkflowStatus,
    utc_now,
)
from app.tools.registry import SafeToolRegistry
from tests.conftest import MockLLMGateway


@pytest.mark.asyncio
async def test_langgraph_self_correction_and_pass(tmp_path: Path) -> None:
    ckpt_db = tmp_path / "ckpt_retry.db"
    spec = ProblemSpecification(
        domain=ProblemDomain.MATHEMATICS,
        problem_type="calculus",
        objective="Evaluate integral of x^2 from 0 to 3",
    )
    plan = ExecutionPlan(
        methodology="SymPy symbolic integration verified by adaptive quadrature",
        selected_tools=["symbolic_math", "independent_verifier"],
        tasks=[
            ExecutionTask(
                task_id="t1",
                title="Integrate",
                description="Compute definite integral",
                method="symbolic_math",
                tool_name="symbolic_math",
            )
        ],
        verification_strategy="Compare symbolic result against SciPy quad",
    )

    mock_llm = MockLLMGateway(
        responses={
            ProblemSpecification: [spec],
            ExecutionPlan: [plan],
            SolverToolCallPlan: [
                SolverToolCallPlan(
                    reasoning_strategy="Attempt 1",
                    tool_calls=[
                        PlannedToolCall(
                            task_id="t1",
                            tool_name="symbolic_math",
                            arguments_json=json.dumps(
                                {
                                    "operation": "integrate",
                                    "expression": "x^2",
                                    "variable": "x",
                                    "lower_limit": "0",
                                    "upper_limit": "3",
                                }
                            ),
                        )
                    ],
                ),
                SolverToolCallPlan(
                    reasoning_strategy="Corrected attempt 2",
                    tool_calls=[
                        PlannedToolCall(
                            task_id="t1",
                            tool_name="symbolic_math",
                            arguments_json=json.dumps(
                                {
                                    "operation": "integrate",
                                    "expression": "x^2",
                                    "variable": "x",
                                    "lower_limit": "0",
                                    "upper_limit": "3",
                                }
                            ),
                        )
                    ],
                ),
            ],
            SolverSynthesis: [
                SolverSynthesis(solution_methodology="SymPy", final_answer="8", numeric_value=8.0),
                SolverSynthesis(solution_methodology="SymPy", final_answer="9", numeric_value=9.0),
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
                                    "expression": "x^2",
                                    "variable": "x",
                                    "lower_limit": 0.0,
                                    "upper_limit": 3.0,
                                    "candidate_value": 8.0,
                                    "tolerance": 1e-6,
                                }
                            ),
                        )
                    ]
                ),
                VerificationPlan(
                    verification_calls=[
                        PlannedToolCall(
                            task_id="v2",
                            tool_name="independent_verifier",
                            arguments_json=json.dumps(
                                {
                                    "verification_type": "integral_quadrature",
                                    "expression": "x^2",
                                    "variable": "x",
                                    "lower_limit": 0.0,
                                    "upper_limit": 3.0,
                                    "candidate_value": 9.0,
                                    "tolerance": 1e-6,
                                }
                            ),
                        )
                    ]
                ),
            ],
            VerificationReport: [
                VerificationReport(
                    status=VerificationStatus.FAIL,
                    checks=[VerificationCheck(check_name="quad", method="scipy.quad", passed=False, details="mismatch")],
                    failure_reasons=["8.0 != 9.0"],
                    evidence_summary="Quadrature produced 9.0",
                ),
                VerificationReport(
                    status=VerificationStatus.PASS,
                    checks=[VerificationCheck(check_name="quad", method="scipy.quad", passed=True, details="exact match")],
                    evidence_summary="Confirmed 9.0",
                ),
            ],
            CorrectionRequest: [
                CorrectionRequest(
                    attempt_number=1,
                    root_cause="Arithmetic synthesis returned 8 instead of 9",
                    error_category="synthesis_mismatch",
                    corrective_instructions="Use exact symbolic_math numeric_approx=9.0",
                    tools_to_reexecute=["symbolic_math"],
                )
            ],
            NarrativeExplanation: [
                NarrativeExplanation(
                    explanation_markdown="$$\\int_0^3 x^2\\,dx = 9$$",
                    confidence_limitations=["Exact polynomial integration on finite domain."],
                )
            ],
        }
    )

    async with AsyncSqliteSaver.from_conn_string(str(ckpt_db)) as checkpointer:
        graph = build_veristem_graph(llm=mock_llm, registry=SafeToolRegistry(), checkpointer=checkpointer)
        initial: VeriSTEMState = {
            "problem_id": "prob_retry",
            "execution_id": "exec_retry",
            "question": "Integrate x^2 from 0 to 3.",
            "started_at": utc_now().isoformat(),
            "max_correction_attempts": 3,
            "correction_attempts": 0,
            "correction_history": [],
            "tool_records": [],
            "agent_timeline": [],
        }
        result = await graph.ainvoke(initial, config={"configurable": {"thread_id": "prob_retry"}})

    assert result["workflow_status"] == WorkflowStatus.COMPLETED.value
    assert result["correction_attempts"] == 1
    assert result["final_report"]["numeric_value"] == pytest.approx(9.0)


@pytest.mark.asyncio
async def test_langgraph_clarification_interrupt_and_resume_across_restart(tmp_path: Path) -> None:
    ckpt_db = tmp_path / "ckpt_resume.db"
    spec_missing = ProblemSpecification(
        domain=ProblemDomain.PHYSICS,
        problem_type="kinematics",
        objective="Find projectile range",
        needs_clarification=True,
        missing_information=["launch angle"],
        clarification_question="What is the launch angle in degrees?",
    )
    spec_ready = ProblemSpecification(
        domain=ProblemDomain.PHYSICS,
        problem_type="kinematics",
        objective="Find projectile range at 45 degrees",
        needs_clarification=False,
    )
    plan = ExecutionPlan(
        methodology="Kinematics formula verified numerically",
        selected_tools=["python_sandbox", "independent_verifier"],
        tasks=[ExecutionTask(task_id="t1", title="Compute", description="Range", method="sandbox", tool_name="python_sandbox")],
        verification_strategy="Direct tolerance check",
    )

    mock_llm = MockLLMGateway(
        responses={
            ProblemSpecification: [spec_missing, spec_ready],
            ExecutionPlan: [plan],
            SolverToolCallPlan: [
                SolverToolCallPlan(
                    reasoning_strategy="Kinematics",
                    tool_calls=[
                        PlannedToolCall(
                            task_id="t1",
                            tool_name="python_sandbox",
                            arguments_json=json.dumps({"code": "RESULT = round(10.0**2 / 9.81, 4)"}),
                        )
                    ],
                )
            ],
            SolverSynthesis: [
                SolverSynthesis(solution_methodology="Kinematics", final_answer="10.1937", numeric_value=10.1937, units="m")
            ],
            VerificationPlan: [
                VerificationPlan(
                    verification_calls=[
                        PlannedToolCall(
                            task_id="v1",
                            tool_name="independent_verifier",
                            arguments_json=json.dumps(
                                {
                                    "verification_type": "numeric_tolerance",
                                    "candidate_value": 10.1937,
                                    "expected_reference": 10.1936799,
                                    "tolerance": 1e-3,
                                }
                            ),
                        )
                    ]
                )
            ],
            VerificationReport: [
                VerificationReport(status=VerificationStatus.PASS, evidence_summary="Verified within tolerance")
            ],
            NarrativeExplanation: [
                NarrativeExplanation(explanation_markdown="Range is $10.1937\\,\\text{m}$.")
            ],
        }
    )

    # First run hits interrupt
    async with AsyncSqliteSaver.from_conn_string(str(ckpt_db)) as cp1:
        g1 = build_veristem_graph(llm=mock_llm, registry=SafeToolRegistry(), checkpointer=cp1)
        paused = await g1.ainvoke(
            {
                "problem_id": "prob_pause",
                "execution_id": "exec_pause",
                "question": "A ball is launched at 10 m/s. Find range.",
                "started_at": utc_now().isoformat(),
                "max_correction_attempts": 3,
                "correction_attempts": 0,
                "correction_history": [],
                "tool_records": [],
                "agent_timeline": [],
            },
            config={"configurable": {"thread_id": "prob_pause"}},
        )
        assert "__interrupt__" in paused

    # Simulate fresh process / graph reconstruction resuming from SQLite checkpoint
    async with AsyncSqliteSaver.from_conn_string(str(ckpt_db)) as cp2:
        g2 = build_veristem_graph(llm=mock_llm, registry=SafeToolRegistry(), checkpointer=cp2)
        resumed = await g2.ainvoke(
            Command(resume={"clarification_response": "Launch angle is 45 degrees."}),
            config={"configurable": {"thread_id": "prob_pause"}},
        )
        assert resumed["workflow_status"] == WorkflowStatus.COMPLETED.value
        assert resumed["final_report"]["numeric_value"] == pytest.approx(10.1937)
