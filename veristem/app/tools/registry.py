import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from app.core.exceptions import ToolExecutionError
from app.schemas.models import ToolExecutionRecord
from app.tools.dimensional_analysis import DimensionalAnalysisInput, run_dimensional_analysis
from app.tools.equation_validator import EquationValidatorInput, run_equation_validator
from app.tools.independent_verifier import IndependentVerifierInput, run_independent_verifier
from app.tools.numerical_compute import NumericalComputeInput, run_numerical_compute
from app.tools.python_sandbox import PythonSandboxInput, run_python_sandbox
from app.tools.scientific_opt import ScientificOptInput, run_scientific_opt
from app.tools.statistical_analysis import StatisticalAnalysisInput, run_statistical_analysis
from app.tools.symbolic_math import SymbolicMathInput, run_symbolic_math


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    description: str
    input_schema: type[BaseModel]
    handler: Callable[[Any], BaseModel]


class SafeToolRegistry:
    """Registry of schema-validated computational tools."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        defaults = [
            RegisteredTool(
                name="symbolic_math",
                description="SymPy symbolic differentiation, integration, equation solving, limits, series, ODEs, and matrices.",
                input_schema=SymbolicMathInput,
                handler=run_symbolic_math,
            ),
            RegisteredTool(
                name="numerical_compute",
                description="NumPy linear systems, eigenvalues, SVD, polynomial roots, and quadrature.",
                input_schema=NumericalComputeInput,
                handler=run_numerical_compute,
            ),
            RegisteredTool(
                name="scientific_opt",
                description="SciPy scalar minimization, root finding, adaptive quadrature, Black-Scholes, and bond pricing.",
                input_schema=ScientificOptInput,
                handler=run_scientific_opt,
            ),
            RegisteredTool(
                name="statistical_analysis",
                description="Pandas & SciPy descriptive statistics, linear regression, t-tests, and portfolio risk metrics.",
                input_schema=StatisticalAnalysisInput,
                handler=run_statistical_analysis,
            ),
            RegisteredTool(
                name="dimensional_analysis",
                description="Pint unit conversions and dimensional consistency checks.",
                input_schema=DimensionalAnalysisInput,
                handler=run_dimensional_analysis,
            ),
            RegisteredTool(
                name="equation_validator",
                description="SymPy symbolic identity checking and numerical substitution residual validation.",
                input_schema=EquationValidatorInput,
                handler=run_equation_validator,
            ),
            RegisteredTool(
                name="independent_verifier",
                description="Independent numerical cross-verification via quadrature, finite differences, residuals, and Monte Carlo.",
                input_schema=IndependentVerifierInput,
                handler=run_independent_verifier,
            ),
            RegisteredTool(
                name="python_sandbox",
                description="Isolated subprocess Python execution with AST filtering, resource caps, and no network access.",
                input_schema=PythonSandboxInput,
                handler=run_python_sandbox,
            ),
        ]
        for tool in defaults:
            self._tools[tool.name] = tool

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema.model_json_schema(),
            }
            for t in self._tools.values()
        ]

    def execute(self, tool_name: str, payload: dict[str, Any]) -> ToolExecutionRecord:
        if tool_name not in self._tools:
            raise ToolExecutionError(tool_name, f"Unknown tool '{tool_name}'")
        tool = self._tools[tool_name]
        call_id = f"tc_{uuid.uuid4().hex[:12]}"
        started = time.perf_counter()
        try:
            validated_input = tool.input_schema.model_validate(payload)
            output_model = tool.handler(validated_input)
            duration_ms = round((time.perf_counter() - started) * 1000.0, 3)
            return ToolExecutionRecord(
                tool_call_id=call_id,
                tool_name=tool_name,
                input_payload=validated_input.model_dump(mode="json"),
                output_payload=output_model.model_dump(mode="json"),
                success=True,
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = round((time.perf_counter() - started) * 1000.0, 3)
            return ToolExecutionRecord(
                tool_call_id=call_id,
                tool_name=tool_name,
                input_payload=payload,
                output_payload={},
                success=False,
                error_message=str(exc),
                duration_ms=duration_ms,
            )


_REGISTRY: SafeToolRegistry | None = None


def get_tool_registry() -> SafeToolRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = SafeToolRegistry()
    return _REGISTRY
