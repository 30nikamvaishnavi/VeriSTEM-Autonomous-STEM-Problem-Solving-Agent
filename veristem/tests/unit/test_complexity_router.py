from app.router.complexity import classify_problem_complexity
from app.schemas.models import ComplexityLevel, ExecutionPath


def test_simple_integral_routes_to_fast_path() -> None:
    decision = classify_problem_complexity(
        "Evaluate the definite integral of x^2 * exp(-x) from x = 0 to x = 2."
    )
    assert decision.complexity_level == ComplexityLevel.LEVEL_1_SIMPLE
    assert decision.execution_path == ExecutionPath.FAST_PATH
    assert decision.problem_category == "calculus_integral"


def test_projectile_kinematics_routes_to_multi_agent() -> None:
    decision = classify_problem_complexity(
        "A projectile is launched from ground level at 45 m/s at 35 degrees above horizontal."
    )
    assert decision.complexity_level in (
        ComplexityLevel.LEVEL_2_INTERMEDIATE,
        ComplexityLevel.LEVEL_3_ADVANCED,
    )
    assert decision.execution_path == ExecutionPath.MULTI_AGENT_WORKFLOW


def test_black_scholes_mc_routes_to_multi_agent() -> None:
    decision = classify_problem_complexity(
        "Compute the Black-Scholes European call price for S=100, K=105, r=0.05, sigma=0.20, T=1.0 year, and verify against Monte Carlo simulation."
    )
    assert decision.complexity_level == ComplexityLevel.LEVEL_3_ADVANCED
    assert decision.execution_path == ExecutionPath.MULTI_AGENT_WORKFLOW
