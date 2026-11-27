from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class NumericalComputeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: str = Field(
        ...,
        pattern="^(linear_solve|eigenvalues|svd|poly_roots|trapezoid_integral|matrix_norm|dot_product)$",
    )
    matrix_a: list[list[float]] | None = None
    vector_b: list[float] | None = None
    coefficients: list[float] | None = None
    x_values: list[float] | None = None
    y_values: list[float] | None = None


class NumericalComputeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: str
    result: Any
    condition_number: float | None = None
    residual_norm: float | None = None


def run_numerical_compute(params: NumericalComputeInput) -> NumericalComputeOutput:
    op = params.operation
    if op == "linear_solve":
        if not params.matrix_a or not params.vector_b:
            raise ValueError("matrix_a and vector_b are required")
        a = np.asarray(params.matrix_a, dtype=np.float64)
        b = np.asarray(params.vector_b, dtype=np.float64)
        cond = float(np.linalg.cond(a))
        x = np.linalg.solve(a, b)
        resid = float(np.linalg.norm(a @ x - b))
        return NumericalComputeOutput(
            operation=op,
            result=x.tolist(),
            condition_number=cond,
            residual_norm=resid,
        )

    if op == "eigenvalues":
        if not params.matrix_a:
            raise ValueError("matrix_a is required")
        a = np.asarray(params.matrix_a, dtype=np.float64)
        vals = np.linalg.eigvals(a)
        formatted = [float(v.real) if abs(v.imag) < 1e-12 else f"{v.real:.8g}{v.imag:+.8g}j" for v in vals]
        return NumericalComputeOutput(operation=op, result=formatted, condition_number=float(np.linalg.cond(a)))

    if op == "svd":
        if not params.matrix_a:
            raise ValueError("matrix_a is required")
        a = np.asarray(params.matrix_a, dtype=np.float64)
        _, s, _ = np.linalg.svd(a)
        return NumericalComputeOutput(
            operation=op,
            result={"singular_values": s.tolist()},
            condition_number=float(s[0] / s[-1]) if s[-1] > 1e-15 else float("inf"),
        )

    if op == "poly_roots":
        if not params.coefficients:
            raise ValueError("coefficients are required")
        roots = np.roots(np.asarray(params.coefficients, dtype=np.float64))
        formatted = [float(r.real) if abs(r.imag) < 1e-12 else f"{r.real:.8g}{r.imag:+.8g}j" for r in roots]
        return NumericalComputeOutput(operation=op, result=formatted)

    if op == "trapezoid_integral":
        if not params.x_values or not params.y_values or len(params.x_values) != len(params.y_values):
            raise ValueError("Equal-length x_values and y_values are required")
        val = float(np.trapezoid(np.asarray(params.y_values, dtype=np.float64), np.asarray(params.x_values, dtype=np.float64)))
        return NumericalComputeOutput(operation=op, result=val)

    if op == "matrix_norm":
        if not params.matrix_a:
            raise ValueError("matrix_a is required")
        a = np.asarray(params.matrix_a, dtype=np.float64)
        return NumericalComputeOutput(
            operation=op,
            result={"frobenius": float(np.linalg.norm(a)), "spectral": float(np.linalg.norm(a, ord=2))},
            condition_number=float(np.linalg.cond(a)),
        )

    if op == "dot_product":
        if not params.vector_b or not params.x_values:
            raise ValueError("vector_b and x_values are required")
        v1 = np.asarray(params.vector_b, dtype=np.float64)
        v2 = np.asarray(params.x_values, dtype=np.float64)
        return NumericalComputeOutput(operation=op, result=float(np.dot(v1, v2)))

    raise ValueError(f"Unsupported operation: {op}")
