import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.tools.dimensional_analysis import DimensionalAnalysisInput, run_dimensional_analysis
from app.tools.independent_verifier import IndependentVerifierInput, run_independent_verifier
from app.tools.numerical_compute import NumericalComputeInput, run_numerical_compute
from app.tools.scientific_opt import ScientificOptInput, run_scientific_opt
from app.tools.symbolic_math import SymbolicMathInput, run_symbolic_math


@given(
    a=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    b=st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=25)
def test_symbolic_integral_matches_adaptive_quadrature(a: float, b: float) -> None:
    sym_out = run_symbolic_math(
        SymbolicMathInput(
            operation="integrate",
            expression=f"({a})*x^2 + 1",
            variable="x",
            lower_limit="0",
            upper_limit=str(b),
        )
    )
    assert sym_out.numeric_approx is not None
    ver_out = run_independent_verifier(
        IndependentVerifierInput(
            verification_type="integral_quadrature",
            expression=f"({a})*x^2 + 1",
            variable="x",
            lower_limit=0.0,
            upper_limit=b,
            candidate_value=sym_out.numeric_approx,
            tolerance=1e-4,
        )
    )
    assert ver_out.passed is True


@given(
    spot=st.floats(min_value=20.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    strike=st.floats(min_value=20.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    rate=st.floats(min_value=0.0, max_value=0.12, allow_nan=False, allow_infinity=False),
    vol=st.floats(min_value=0.05, max_value=0.60, allow_nan=False, allow_infinity=False),
    t=st.floats(min_value=0.1, max_value=3.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=30)
def test_black_scholes_put_call_parity(spot: float, strike: float, rate: float, vol: float, t: float) -> None:
    call_p = run_scientific_opt(
        ScientificOptInput(
            operation="black_scholes",
            spot=spot,
            strike=strike,
            rate=rate,
            volatility=vol,
            time_to_maturity=t,
            option_type="call",
        )
    ).result["price"]
    put_p = run_scientific_opt(
        ScientificOptInput(
            operation="black_scholes",
            spot=spot,
            strike=strike,
            rate=rate,
            volatility=vol,
            time_to_maturity=t,
            option_type="put",
        )
    ).result["price"]
    lhs = call_p - put_p
    rhs = spot - strike * math.exp(-rate * t)
    assert lhs == pytest.approx(rhs, abs=1e-8)


@given(
    v=st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=20)
def test_unit_conversion_roundtrip(v: float) -> None:
    to_ms = run_dimensional_analysis(
        DimensionalAnalysisInput(operation="convert", value=v, from_unit="km/hour", to_unit="m/s")
    )
    back = run_dimensional_analysis(
        DimensionalAnalysisInput(
            operation="convert",
            value=to_ms.converted_value,
            from_unit="m/s",
            to_unit="km/hour",
        )
    )
    assert back.converted_value == pytest.approx(v, rel=1e-9)


@given(
    d1=st.floats(min_value=1.0, max_value=20.0, allow_nan=False, allow_infinity=False),
    d2=st.floats(min_value=1.0, max_value=20.0, allow_nan=False, allow_infinity=False),
    b1=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    b2=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=20)
def test_linear_solve_residual_near_zero(d1: float, d2: float, b1: float, b2: float) -> None:
    out = run_numerical_compute(
        NumericalComputeInput(
            operation="linear_solve",
            matrix_a=[[d1, 0.1], [0.1, d2]],
            vector_b=[b1, b2],
        )
    )
    assert out.residual_norm is not None
    assert out.residual_norm < 1e-10
