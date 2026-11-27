import math

import numpy as np
import sympy as sp
from pydantic import BaseModel, ConfigDict, Field
from scipy import integrate

from app.tools.symbolic_math import safe_parse_sympy


class IndependentVerifierInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verification_type: str = Field(
        ...,
        pattern="^(integral_quadrature|derivative_finite_diff|root_residual|black_scholes_mc|numeric_tolerance)$",
    )
    expression: str | None = None
    variable: str = "x"
    lower_limit: float | None = None
    upper_limit: float | None = None
    point: float | None = None
    candidate_value: float
    tolerance: float = Field(default=1e-4, gt=0)
    # Black-Scholes Monte Carlo check
    spot: float | None = None
    strike: float | None = None
    rate: float | None = None
    volatility: float | None = None
    time_to_maturity: float | None = None
    option_type: str = "call"
    mc_paths: int = Field(default=100_000, ge=5000, le=500_000)
    expected_reference: float | None = None


class IndependentVerifierOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    method_used: str
    independent_value: float
    candidate_value: float
    absolute_error: float
    relative_error: float
    tolerance: float
    details: str


def run_independent_verifier(params: IndependentVerifierInput) -> IndependentVerifierOutput:
    vtype = params.verification_type
    candidate = params.candidate_value

    if vtype == "numeric_tolerance":
        if params.expected_reference is None:
            raise ValueError("expected_reference is required")
        ref = params.expected_reference
        abs_err = abs(candidate - ref)
        rel_err = abs_err / max(abs(ref), 1e-12)
        passed = abs_err <= params.tolerance or rel_err <= params.tolerance
        return IndependentVerifierOutput(
            passed=passed,
            method_used="direct_tolerance_comparison",
            independent_value=ref,
            candidate_value=candidate,
            absolute_error=abs_err,
            relative_error=rel_err,
            tolerance=params.tolerance,
            details=f"abs_err={abs_err:.3e}, rel_err={rel_err:.3e}",
        )

    if vtype == "black_scholes_mc":
        if None in (params.spot, params.strike, params.rate, params.volatility, params.time_to_maturity):
            raise ValueError("Black-Scholes parameters required")
        rng = np.random.default_rng(42)
        s0 = float(params.spot)
        k = float(params.strike)
        r = float(params.rate)
        sigma = float(params.volatility)
        t = float(params.time_to_maturity)
        half = params.mc_paths // 2
        z = rng.standard_normal(half)
        z_antithetic = np.concatenate([z, -z])
        st = s0 * np.exp((r - 0.5 * sigma * sigma) * t + sigma * math.sqrt(t) * z_antithetic)
        payoffs = np.maximum(st - k, 0.0) if params.option_type == "call" else np.maximum(k - st, 0.0)
        mc_price = float(math.exp(-r * t) * np.mean(payoffs))
        abs_err = abs(candidate - mc_price)
        rel_err = abs_err / max(abs(mc_price), 1e-12)
        mc_tol = max(params.tolerance, 0.02 * max(mc_price, 1.0))
        passed = abs_err <= mc_tol
        return IndependentVerifierOutput(
            passed=passed,
            method_used="antithetic_monte_carlo",
            independent_value=mc_price,
            candidate_value=candidate,
            absolute_error=abs_err,
            relative_error=rel_err,
            tolerance=mc_tol,
            details=f"Monte Carlo ({params.mc_paths} antithetic paths) = {mc_price:.6f}",
        )

    if not params.expression:
        raise ValueError("expression is required")
    sym = sp.Symbol(params.variable, real=True)
    expr = safe_parse_sympy(params.expression, [params.variable])
    fn = sp.lambdify(sym, expr, modules=["numpy"])

    if vtype == "integral_quadrature":
        if params.lower_limit is None or params.upper_limit is None:
            raise ValueError("lower_limit and upper_limit are required")
        quad_val, quad_err = integrate.quad(lambda x: float(fn(x)), params.lower_limit, params.upper_limit)
        abs_err = abs(candidate - quad_val)
        rel_err = abs_err / max(abs(quad_val), 1e-12)
        passed = abs_err <= max(params.tolerance, 10 * quad_err)
        return IndependentVerifierOutput(
            passed=passed,
            method_used="scipy_quad_adaptive_quadrature",
            independent_value=float(quad_val),
            candidate_value=candidate,
            absolute_error=abs_err,
            relative_error=rel_err,
            tolerance=params.tolerance,
            details=f"Adaptive quadrature={quad_val:.10g} (quad_err={quad_err:.2e})",
        )

    if vtype == "derivative_finite_diff":
        pt = params.point if params.point is not None else 1.0
        h = 1e-6
        fd = (float(fn(pt + h)) - float(fn(pt - h))) / (2.0 * h)
        abs_err = abs(candidate - fd)
        rel_err = abs_err / max(abs(fd), 1e-12)
        passed = abs_err <= params.tolerance or rel_err <= params.tolerance
        return IndependentVerifierOutput(
            passed=passed,
            method_used="central_finite_difference",
            independent_value=fd,
            candidate_value=candidate,
            absolute_error=abs_err,
            relative_error=rel_err,
            tolerance=params.tolerance,
            details=f"Central finite difference at {params.variable}={pt}: {fd:.10g}",
        )

    if vtype == "root_residual":
        res_val = abs(float(fn(candidate)))
        passed = res_val <= params.tolerance
        return IndependentVerifierOutput(
            passed=passed,
            method_used="direct_function_residual",
            independent_value=0.0,
            candidate_value=res_val,
            absolute_error=res_val,
            relative_error=res_val,
            tolerance=params.tolerance,
            details=f"|f({candidate})| = {res_val:.3e}",
        )

    raise ValueError(f"Unsupported verification_type: {vtype}")
