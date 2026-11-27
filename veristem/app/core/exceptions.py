class VeriSTEMError(Exception):
    """Base exception for VeriSTEM errors."""

    def __init__(self, message: str, code: str = "VERISTEM_ERROR", status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class ConfigurationError(VeriSTEMError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="CONFIGURATION_ERROR", status_code=500)


class AuthenticationError(VeriSTEMError):
    def __init__(self, message: str = "Invalid or missing API key") -> None:
        super().__init__(message, code="UNAUTHORIZED", status_code=401)


class RateLimitExceededError(VeriSTEMError):
    def __init__(self, message: str = "Rate limit exceeded") -> None:
        super().__init__(message, code="RATE_LIMIT_EXCEEDED", status_code=429)


class ProblemNotFoundError(VeriSTEMError):
    def __init__(self, problem_id: str) -> None:
        super().__init__(f"Problem '{problem_id}' not found", code="NOT_FOUND", status_code=404)


class SandboxViolationError(VeriSTEMError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="SANDBOX_VIOLATION", status_code=400)


class ToolExecutionError(VeriSTEMError):
    def __init__(self, tool_name: str, message: str) -> None:
        super().__init__(f"Tool '{tool_name}' failed: {message}", code="TOOL_ERROR", status_code=500)


class LLMGatewayError(VeriSTEMError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="LLM_GATEWAY_ERROR", status_code=502)


class InvalidStateTransitionError(VeriSTEMError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="INVALID_STATE", status_code=409)
