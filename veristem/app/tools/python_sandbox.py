import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import get_settings
from app.core.exceptions import SandboxViolationError

ALLOWED_MODULES = frozenset({"math", "cmath", "decimal", "fractions", "statistics", "json", "numpy", "scipy", "sympy"})

FORBIDDEN_CALLS = frozenset({
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "input",
    "breakpoint",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
})

FORBIDDEN_ATTR_PREFIXES = ("__",)


class PythonSandboxInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., min_length=1, max_length=6000)
    timeout_seconds: int = Field(default=10, ge=1, le=30)


class PythonSandboxOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stdout: str
    stderr: str
    result: Any = None
    exit_code: int


def validate_python_ast(code: str) -> None:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SandboxViolationError(f"Syntax error in sandbox code: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                if root_pkg not in ALLOWED_MODULES:
                    raise SandboxViolationError(f"Import of module '{alias.name}' is prohibited")
        elif isinstance(node, ast.ImportFrom):
            root_pkg = (node.module or "").split(".")[0]
            if root_pkg not in ALLOWED_MODULES:
                raise SandboxViolationError(f"Import from module '{node.module}' is prohibited")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                raise SandboxViolationError(f"Call to '{node.func.id}' is prohibited")
        elif isinstance(node, ast.Attribute):
            if any(node.attr.startswith(p) for p in FORBIDDEN_ATTR_PREFIXES):
                raise SandboxViolationError(f"Access to dunder attribute '{node.attr}' is prohibited")


def run_python_sandbox(params: PythonSandboxInput) -> PythonSandboxOutput:
    settings = get_settings()
    validate_python_ast(params.code)
    timeout = min(params.timeout_seconds, settings.sandbox_timeout_seconds)
    mem_bytes = settings.sandbox_memory_mb * 1024 * 1024

    runner_script = """
import json
import sys

try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, ({cpu_limit}, {cpu_limit}))
    resource.setrlimit(resource.RLIMIT_AS, ({mem_limit}, {mem_limit}))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
except Exception:
    pass

# Block socket networking inside the isolated child process
try:
    import socket
    def _blocked(*args, **kwargs):
        raise RuntimeError("Network access is disabled in VeriSTEM sandbox")
    socket.socket = _blocked
except Exception:
    pass

user_code = sys.stdin.read()
namespace = {{"RESULT": None}}
exec(compile(user_code, "<sandbox>", "exec"), namespace, namespace)
print("__VERISTEM_RESULT__:" + json.dumps(namespace.get("RESULT"), default=str))
""".format(cpu_limit=timeout, mem_limit=mem_bytes)

    with tempfile.TemporaryDirectory(prefix="veristem_sandbox_") as tmpdir:
        script_path = Path(tmpdir) / "runner.py"
        script_path.write_text(runner_script, encoding="utf-8")

        safe_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                input=params.code,
                text=True,
                capture_output=True,
                timeout=timeout,
                cwd=tmpdir,
                env=safe_env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxViolationError(f"Sandbox execution timed out after {timeout}s") from exc

    parsed_result: Any = None
    stdout_lines: list[str] = []
    for line in proc.stdout.splitlines():
        if line.startswith("__VERISTEM_RESULT__:"):
            payload = line.split("__VERISTEM_RESULT__:", 1)[1]
            parsed_result = json.loads(payload)
        else:
            stdout_lines.append(line)

    return PythonSandboxOutput(
        stdout="\n".join(stdout_lines)[:4000],
        stderr=proc.stderr[:2000],
        result=parsed_result,
        exit_code=proc.returncode,
    )
