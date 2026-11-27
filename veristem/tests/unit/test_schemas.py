import pytest
from pydantic import ValidationError

from app.schemas.models import (
    CreateProblemRequest,
    MathematicalVariable,
    ProblemDomain,
    ProblemSpecification,
    VerificationReport,
    VerificationStatus,
)


def test_problem_specification_validation() -> None:
    spec = ProblemSpecification(
        domain=ProblemDomain.PHYSICS,
        problem_type="kinematics",
        objective="Find horizontal range",
        variables=[
            MathematicalVariable(symbol="v0", name="initial_speed", value=45.0, unit="m/s"),
        ],
        assumptions=["No air resistance"],
    )
    assert spec.domain == ProblemDomain.PHYSICS
    assert spec.variables[0].unit == "m/s"


def test_create_problem_rejects_blank_question() -> None:
    with pytest.raises(ValidationError):
        CreateProblemRequest(question="   ")


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        VerificationReport(
            status=VerificationStatus.PASS,
            evidence_summary="ok",
            unexpected_field="boom",  # type: ignore[call-arg]
        )
