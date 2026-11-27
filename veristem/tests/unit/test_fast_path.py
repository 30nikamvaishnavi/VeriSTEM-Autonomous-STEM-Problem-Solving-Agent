import math

from app.router.complexity import classify_problem_complexity
from app.schemas.models import VerificationStatus
from app.services.fast_path import try_fast_path_solve
from app.tools.registry import get_tool_registry


def test_fast_path_sample_integral_exact_and_verified() -> None:
    q = "Evaluate the definite integral of x^2 * exp(-x) from x = 0 to x = 2."
    decision = classify_problem_complexity(q)
    registry = get_tool_registry()
    outcome = try_fast_path_solve(
        problem_id="prob_test_int",
        execution_id="exec_test_int",
        question=q,
        decision=decision,
        registry=registry,
    )
    assert outcome.handled is True
    assert outcome.verification_report is not None
    assert outcome.verification_report.status == VerificationStatus.PASS
    assert outcome.final_report is not None
    assert outcome.final_report.numeric_value is not None
    assert abs(outcome.final_report.numeric_value - (2.0 - 10.0 * math.exp(-2.0))) < 1e-9
    assert outcome.final_report.elapsed_ms is not None
    assert outcome.final_report.elapsed_ms < 2000.0
