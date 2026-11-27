import hmac
from fastapi import Header

from app.core.config import get_settings
from app.core.exceptions import AuthenticationError


async def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """Verify X-API-Key header when VERISTEM_API_KEY is configured."""
    settings = get_settings()
    expected = settings.veristem_api_key.strip()
    if not expected:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key.encode("utf-8"), expected.encode("utf-8")):
        raise AuthenticationError("Invalid or missing X-API-Key header")


def sanitize_problem_text(text: str, max_chars: int = 8000) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Problem text cannot be empty")
    if len(cleaned) > max_chars:
        raise ValueError(f"Problem text exceeds maximum length of {max_chars} characters")
    return cleaned
