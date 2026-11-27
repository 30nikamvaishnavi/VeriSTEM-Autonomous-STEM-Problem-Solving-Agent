import pytest

from app.core.exceptions import SandboxViolationError
from app.tools.python_sandbox import PythonSandboxInput, run_python_sandbox, validate_python_ast


@pytest.mark.parametrize(
    "malicious_code",
    [
        "import os\nRESULT = os.listdir('.')",
        "import subprocess\nRESULT = subprocess.run(['id'])",
        "import socket\nRESULT = socket.socket()",
        "open('/etc/passwd').read()",
        "eval('1+1')",
        "__import__('os').system('echo hi')",
        "x = ().__class__.__bases__[0]",
    ],
)
def test_sandbox_blocks_unsafe_ast(malicious_code: str) -> None:
    with pytest.raises(SandboxViolationError):
        validate_python_ast(malicious_code)


def test_sandbox_executes_safe_math() -> None:
    out = run_python_sandbox(
        PythonSandboxInput(
            code="import math\nRESULT = round(math.sin(math.pi / 6), 6)",
            timeout_seconds=5,
        )
    )
    assert out.exit_code == 0
    assert out.result == pytest.approx(0.5, rel=1e-6)


def test_sandbox_enforces_timeout() -> None:
    with pytest.raises(SandboxViolationError, match="timed out"):
        run_python_sandbox(
            PythonSandboxInput(
                code="while True:\n    pass",
                timeout_seconds=1,
            )
        )
