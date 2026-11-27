from typing import Any

import sympy as sp
from pydantic import BaseModel, ConfigDict, Field
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

_SAFE_SYMPY_LOCALS: dict[str, Any] = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "pi": sp.pi,
    "E": sp.E,
    "e": sp.E,
    "oo": sp.oo,
    "inf": sp.oo,
    "I": sp.I,
    "gamma": sp.gamma,
    "erf": sp.erf,
}


def safe_parse_sympy(expr_str: str, symbol_names: list[str] | None = None) -> sp.Expr:
    local_dict = dict(_SAFE_SYMPY_LOCALS)
    for name in symbol_names or ["x", "y", "z", "t", "s", "r", "n", "k"]:
        if name.isidentifier():
            local_dict[name] = sp.Symbol(name, real=True)
    return parse_expr(expr_str, local_dict=local_dict, transformations=_TRANSFORMS, evaluate=True)


class SymbolicMathInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: str = Field(
        ...,
        pattern="^(simplify|diff|integrate|solve|limit|series|dsolve|matrix_eigen|evaluate)$",
    )
    expression: str = Field(..., min_length=1, max_length=2000)
    variable: str = Field(default="x", max_length=32)
    variables: list[str] = Field(default_factory=lambda: ["x"])
    lower_limit: str | None = None
    upper_limit: str | None = None
    point: str | None = None
    order: int = Field(default=1, ge=1, le=12)
    substitutions: dict[str, float] = Field(default_factory=dict)
    matrix: list[list[float]] | None = None


class SymbolicMathOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: str
    result_str: str
    result_latex: str
    numeric_approx: float | None = None
    solutions: list[str] = Field(default_factory=list)


def run_symbolic_math(params: SymbolicMathInput) -> SymbolicMathOutput:
    var_names = list(dict.fromkeys([params.variable, *params.variables]))
    sym = sp.Symbol(params.variable, real=True)

    if params.operation == "matrix_eigen":
        if not params.matrix:
            raise ValueError("matrix is required for matrix_eigen")
        mat = sp.Matrix(params.matrix)
        eigenvals = mat.eigenvals()
        det = mat.det()
        result = f"eigenvalues={dict(eigenvals)}, det={det}"
        return SymbolicMathOutput(
            operation=params.operation,
            result_str=result,
            result_latex=sp.latex(mat),
            numeric_approx=float(det.evalf()),
            solutions=[str(k) for k in eigenvals.keys()],
        )

    if params.operation == "dsolve":
        t = sp.Symbol(params.variable)
        y = sp.Function("y")
        local_dict = dict(_SAFE_SYMPY_LOCALS)
        local_dict[params.variable] = t
        local_dict["y"] = y
        local_dict["Derivative"] = sp.Derivative
        if "=" in params.expression:
            lhs_s, rhs_s = params.expression.split("=", 1)
            lhs = parse_expr(lhs_s, local_dict=local_dict, transformations=_TRANSFORMS)
            rhs = parse_expr(rhs_s, local_dict=local_dict, transformations=_TRANSFORMS)
            ode = sp.Eq(lhs, rhs)
        else:
            ode = sp.Eq(parse_expr(params.expression, local_dict=local_dict, transformations=_TRANSFORMS), 0)
        sol = sp.dsolve(ode, y(t))
        return SymbolicMathOutput(
            operation=params.operation,
            result_str=str(sol),
            result_latex=sp.latex(sol),
            solutions=[str(sol)],
        )

    if params.operation == "solve":
        if "=" in params.expression:
            lhs_s, rhs_s = params.expression.split("=", 1)
            expr = safe_parse_sympy(lhs_s, var_names) - safe_parse_sympy(rhs_s, var_names)
        else:
            expr = safe_parse_sympy(params.expression, var_names)
        sols = sp.solve(expr, sym)
        sol_strs = [str(s) for s in sols]
        numeric = None
        if sols:
            try:
                val = complex(sols[0].evalf())
                if abs(val.imag) < 1e-12:
                    numeric = float(val.real)
            except Exception:
                numeric = None
        return SymbolicMathOutput(
            operation=params.operation,
            result_str=", ".join(sol_strs) if sol_strs else "No solution",
            result_latex=sp.latex(sols),
            numeric_approx=numeric,
            solutions=sol_strs,
        )

    expr = safe_parse_sympy(params.expression, var_names)

    if params.operation == "simplify":
        res = sp.simplify(expr)
    elif params.operation == "diff":
        res = sp.diff(expr, sym, params.order)
    elif params.operation == "integrate":
        if params.lower_limit is not None and params.upper_limit is not None:
            lo = safe_parse_sympy(params.lower_limit, var_names)
            hi = safe_parse_sympy(params.upper_limit, var_names)
            res = sp.integrate(expr, (sym, lo, hi))
        else:
            res = sp.integrate(expr, sym)
    elif params.operation == "limit":
        pt = safe_parse_sympy(params.point or "0", var_names)
        res = sp.limit(expr, sym, pt)
    elif params.operation == "series":
        pt = safe_parse_sympy(params.point or "0", var_names)
        res = sp.series(expr, sym, pt, n=params.order + 1).removeO()
    elif params.operation == "evaluate":
        sub_map = {sp.Symbol(k, real=True): v for k, v in params.substitutions.items()}
        res = expr.subs(sub_map)
    else:
        raise ValueError(f"Unsupported operation: {params.operation}")

    numeric_val: float | None = None
    try:
        evaluated = res.subs({sp.Symbol(k, real=True): v for k, v in params.substitutions.items()})
        if not evaluated.free_symbols:
            cval = complex(evaluated.evalf())
            if abs(cval.imag) < 1e-12:
                numeric_val = float(cval.real)
    except Exception:
        numeric_val = None

    return SymbolicMathOutput(
        operation=params.operation,
        result_str=str(res),
        result_latex=sp.latex(res),
        numeric_approx=numeric_val,
        solutions=[str(res)],
    )
