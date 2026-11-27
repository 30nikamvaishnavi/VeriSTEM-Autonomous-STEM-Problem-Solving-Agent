from pint import UnitRegistry
from pydantic import BaseModel, ConfigDict, Field

_UREG = UnitRegistry()


class DimensionalAnalysisInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: str = Field(..., pattern="^(convert|check_equivalence|check_expression_consistency)$")
    value: float | None = None
    from_unit: str | None = None
    to_unit: str | None = None
    lhs_unit: str | None = None
    rhs_unit: str | None = None


class DimensionalAnalysisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: str
    consistent: bool
    converted_value: float | None = None
    result_unit: str
    dimensionality: str
    details: str


def run_dimensional_analysis(params: DimensionalAnalysisInput) -> DimensionalAnalysisOutput:
    op = params.operation
    if op == "convert":
        if params.value is None or not params.from_unit or not params.to_unit:
            raise ValueError("value, from_unit, and to_unit are required")
        q = _UREG.Quantity(params.value, params.from_unit)
        converted = q.to(params.to_unit)
        return DimensionalAnalysisOutput(
            operation=op,
            consistent=True,
            converted_value=float(converted.magnitude),
            result_unit=str(converted.units),
            dimensionality=str(converted.dimensionality),
            details=f"{params.value} {params.from_unit} = {float(converted.magnitude):.8g} {converted.units}",
        )

    if op in ("check_equivalence", "check_expression_consistency"):
        if not params.lhs_unit or not params.rhs_unit:
            raise ValueError("lhs_unit and rhs_unit are required")
        u_lhs = _UREG.Quantity(1.0, params.lhs_unit)
        u_rhs = _UREG.Quantity(1.0, params.rhs_unit)
        consistent = u_lhs.dimensionality == u_rhs.dimensionality
        return DimensionalAnalysisOutput(
            operation=op,
            consistent=consistent,
            converted_value=float(u_lhs.to(params.rhs_unit).magnitude) if consistent else None,
            result_unit=params.rhs_unit,
            dimensionality=f"LHS={u_lhs.dimensionality}, RHS={u_rhs.dimensionality}",
            details="Dimensions match" if consistent else "Dimensional mismatch detected",
        )

    raise ValueError(f"Unsupported operation: {op}")
