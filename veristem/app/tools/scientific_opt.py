import math
from typing import Any

import numpy as np
import sympy as sp
from pydantic import BaseModel, ConfigDict, Field
from scipy import integrate, optimize
from scipy.stats import norm

from app.tools.symbolic_math import safe_parse_sympy


class ScientificOptInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: str = Field(
        ...,
        pattern="^(minimize_scalar|root_brentq|quad_integrate|black_scholes|bond_price)$",
    )
    expression: str | None = None
    variable: str = "x"
    bounds: tuple[float, float] | None = None
    # Black-Scholes parameters
    spot: float | None = Field(default=None, gt=0)
    strike: float | None = Field(default=None, gt=0)
    rate: float | None = None
    volatility: float | None = Field(default=None, gt=0)
    time_to_maturity: float | None = Field(default=None, gt=0)
    option_type: str = Field(default="call", pattern="^(call|put)$")
    # Bond parameters
    face_value: float = 1000.0
    coupon_rate: float = 0.05
    yield_rate: float = 0.05
    periods: int = 10
    frequency: int = 2


class ScientificOptOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: str
    result: dict[str, Any]
    converged: bool = True


def run_scientific_opt(params: ScientificOptInput) -> ScientificOptOutput:
    op = params.operation
    if op == "black_scholes":
        if None in (params.spot, params.strike, params.rate, params.volatility, params.time_to_maturity):
            raise ValueError("spot, strike, rate, volatility, and time_to_maturity are required")
        s = float(params.spot)
        k = float(params.strike)
        r = float(params.rate)
        sigma = float(params.volatility)
        t = float(params.time_to_maturity)
        d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
        d2 = d1 - sigma * math.sqrt(t)
        if params.option_type == "call":
            price = s * float(norm.cdf(d1)) - k * math.exp(-r * t) * float(norm.cdf(d2))
            delta = float(norm.cdf(d1))
        else:
            price = k * math.exp(-r * t) * float(norm.cdf(-d2)) - s * float(norm.cdf(-d1))
            delta = float(norm.cdf(d1)) - 1.0
        gamma = float(norm.pdf(d1)) / (s * sigma * math.sqrt(t))
        vega = s * float(norm.pdf(d1)) * math.sqrt(t)
        return ScientificOptOutput(
            operation=op,
            result={
                "option_type": params.option_type,
                "price": price,
                "d1": d1,
                "d2": d2,
                "delta": delta,
                "gamma": gamma,
                "vega": vega,
            },
        )

    if op == "bond_price":
        c = params.face_value * params.coupon_rate / params.frequency
        y = params.yield_rate / params.frequency
        n = params.periods * params.frequency
        pv_coupons = sum(c / ((1 + y) ** i) for i in range(1, n + 1))
        pv_face = params.face_value / ((1 + y) ** n)
        price = pv_coupons + pv_face
        macaulay_num = sum((i / params.frequency) * (c / ((1 + y) ** i)) for i in range(1, n + 1))
        macaulay_num += (n / params.frequency) * pv_face
        macaulay_dur = macaulay_num / price
        mod_dur = macaulay_dur / (1 + y)
        return ScientificOptOutput(
            operation=op,
            result={"price": price, "macaulay_duration": macaulay_dur, "modified_duration": mod_dur},
        )

    if not params.expression:
        raise ValueError("expression is required")
    sym = sp.Symbol(params.variable, real=True)
    expr = safe_parse_sympy(params.expression, [params.variable])
    fn = sp.lambdify(sym, expr, modules=["numpy"])

    if op == "minimize_scalar":
        res = (
            optimize.minimize_scalar(lambda x: float(fn(x)), bounds=params.bounds, method="bounded")
            if params.bounds
            else optimize.minimize_scalar(lambda x: float(fn(x)))
        )
        return ScientificOptOutput(
            operation=op,
            result={"x_min": float(res.x), "f_min": float(res.fun)},
            converged=bool(res.success),
        )

    if op == "root_brentq":
        if not params.bounds:
            raise ValueError("bounds (a, b) are required for root_brentq")
        a, b = params.bounds
        root = float(optimize.brentq(lambda x: float(fn(x)), a, b))
        return ScientificOptOutput(
            operation=op,
            result={"root": root, "residual": abs(float(fn(root)))},
            converged=True,
        )

    if op == "quad_integrate":
        if not params.bounds:
            raise ValueError("bounds (a, b) are required for quad_integrate")
        a, b = params.bounds
        val, err = integrate.quad(lambda x: float(fn(x)), a, b)
        return ScientificOptOutput(
            operation=op,
            result={"integral": float(val), "abs_error_estimate": float(err)},
            converged=True,
        )

    raise ValueError(f"Unsupported operation: {op}")
