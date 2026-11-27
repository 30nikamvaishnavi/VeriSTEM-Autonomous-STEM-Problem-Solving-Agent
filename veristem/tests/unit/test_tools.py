import pytest

from app.tools.registry import SafeToolRegistry


@pytest.fixture
def registry() -> SafeToolRegistry:
    return SafeToolRegistry()


def test_symbolic_math_integral(registry: SafeToolRegistry) -> None:
    rec = registry.execute(
        "symbolic_math",
        {
            "operation": "integrate",
            "expression": "x^2",
            "variable": "x",
            "lower_limit": "0",
            "upper_limit": "3",
        },
    )
    assert rec.success is True
    assert rec.output_payload["numeric_approx"] == pytest.approx(9.0, rel=1e-9)


def test_numerical_compute_linear_solve(registry: SafeToolRegistry) -> None:
    rec = registry.execute(
        "numerical_compute",
        {
            "operation": "linear_solve",
            "matrix_a": [[4.0, 1.0], [2.0, 3.0]],
            "vector_b": [9.0, 8.0],
        },
    )
    assert rec.success is True
    assert rec.output_payload["result"] == pytest.approx([1.9, 1.4], rel=1e-9)
    assert rec.output_payload["residual_norm"] < 1e-12


def test_scientific_opt_black_scholes(registry: SafeToolRegistry) -> None:
    rec = registry.execute(
        "scientific_opt",
        {
            "operation": "black_scholes",
            "spot": 100.0,
            "strike": 105.0,
            "rate": 0.05,
            "volatility": 0.20,
            "time_to_maturity": 1.0,
            "option_type": "call",
        },
    )
    assert rec.success is True
    assert rec.output_payload["result"]["price"] == pytest.approx(8.021352, rel=1e-5)


def test_statistical_analysis_portfolio_metrics(registry: SafeToolRegistry) -> None:
    rec = registry.execute(
        "statistical_analysis",
        {
            "operation": "portfolio_metrics",
            "weights": [0.6, 0.4],
            "expected_returns": [0.10, 0.06],
            "covariance_matrix": [[0.04, 0.006], [0.006, 0.0144]],
            "risk_free_rate": 0.02,
        },
    )
    assert rec.success is True
    assert rec.output_payload["metrics"]["expected_return"] == pytest.approx(0.084, rel=1e-9)


def test_dimensional_analysis_conversion_and_mismatch(registry: SafeToolRegistry) -> None:
    ok_rec = registry.execute(
        "dimensional_analysis",
        {"operation": "convert", "value": 72.0, "from_unit": "km/hour", "to_unit": "m/s"},
    )
    assert ok_rec.success is True
    assert ok_rec.output_payload["converted_value"] == pytest.approx(20.0, rel=1e-9)

    bad_rec = registry.execute(
        "dimensional_analysis",
        {"operation": "check_equivalence", "lhs_unit": "meter / second", "rhs_unit": "kilogram"},
    )
    assert bad_rec.success is True
    assert bad_rec.output_payload["consistent"] is False


def test_equation_validator_identity(registry: SafeToolRegistry) -> None:
    rec = registry.execute(
        "equation_validator",
        {
            "operation": "check_identity",
            "lhs_expression": "sin(x)^2 + cos(x)^2",
            "rhs_expression": "1",
            "variables": ["x"],
        },
    )
    assert rec.success is True
    assert rec.output_payload["valid"] is True


def test_independent_verifier_quadrature(registry: SafeToolRegistry) -> None:
    rec = registry.execute(
        "independent_verifier",
        {
            "verification_type": "integral_quadrature",
            "expression": "x^2 * exp(-x)",
            "variable": "x",
            "lower_limit": 0.0,
            "upper_limit": 2.0,
            "candidate_value": 0.6466471676,
            "tolerance": 1e-6,
        },
    )
    assert rec.success is True
    assert rec.output_payload["passed"] is True
