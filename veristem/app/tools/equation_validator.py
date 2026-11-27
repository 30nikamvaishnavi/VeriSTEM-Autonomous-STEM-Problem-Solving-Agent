import sympy as sp
from pydantic import BaseModel, ConfigDict, Field

from app.tools.symbolic_math import safe_parse_sympy


class EquationValidatorInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: str = Field(..., pattern="^(check_identity|substitute_and_validate)$")
    lhs_expression: str = Field(..., min_length=1, max_length=2000)
    rhs_expression: str = Field(..., min_length=1, max_length=2000)
    variables: list[str] = Field(default_factory=lambda: ["x"])
    substitutions: dict[str, float] = Field(default_factory=dict)
    tolerance: float = Field(default=1e-7, gt=0)


class EquationValidatorOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    symbolic_difference: str
    numeric_residual: float | None = None
    details: str


def run_equation_validator(params: EquationValidatorInput) -> EquationValidatorOutput:
    var_names = list(dict.fromkeys([*params.variables, *params.substitutions.keys()]))
    lhs = safe_parse_sympy(params.lhs_expression, var_names)
    rhs = safe_parse_sympy(params.rhs_expression, var_names)
    diff_expr = sp.simplify(lhs - rhs)

    if params.operation == "check_identity":
        is_zero = diff_expr == 0 or sp.trigsimp(diff_expr) == 0
        return EquationValidatorOutput(
            valid=bool(is_zero),
            symbolic_difference=str(diff_expr),
            numeric_residual=0.0 if is_zero else None,
            details="Symbolic identity verified" if is_zero else f"Non-zero residual: {diff_expr}",
        )

    sub_map = {sp.Symbol(k, real=True): v for k, v in params.substitutions.items()}
    eval_diff = diff_expr.subs(sub_map)
    try:
        residual = abs(complex(eval_diff.evalf()))
        valid = residual <= params.tolerance
        return EquationValidatorOutput(
            valid=valid,
            symbolic_difference=str(diff_expr),
            numeric_residual=float(residual),
            details=f"Residual={residual:.3e}, tolerance={params.tolerance:.3e}",
        )
    except Exception as exc:
        return EquationValidatorOutput(
            valid=False,
            symbolic_difference=str(diff_expr),
            numeric_residual=None,
            details=f"Could not evaluate residual numerically: {exc}",
        )
