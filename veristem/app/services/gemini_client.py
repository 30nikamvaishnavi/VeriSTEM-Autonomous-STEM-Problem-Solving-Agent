import asyncio
from dataclasses import dataclass
from typing import Protocol, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from app.core.config import get_settings
from app.core.exceptions import ConfigurationError, LLMGatewayError
from app.core.logging import get_logger

T = TypeVar("T", bound=BaseModel)
logger = get_logger(__name__)


@dataclass
class StructuredLLMResponse:
    parsed: BaseModel
    tokens_input: int
    tokens_output: int


class LLMGateway(Protocol):
    async def generate_structured(
        self,
        *,
        system_instruction: str,
        prompt: str,
        response_model: type[T],
    ) -> StructuredLLMResponse: ...


class GeminiGateway:
    """Server-side Gemini API client using structured JSON schema output and Pydantic validation."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self.api_key = (api_key if api_key is not None else settings.gemini_api_key).strip()
        self.model = model or settings.gemini_model
        self.timeout_seconds = settings.gemini_timeout_seconds
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        if not self.api_key or self.api_key == "MY_GEMINI_API_KEY":
            raise ConfigurationError("GEMINI_API_KEY is not configured")
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    async def generate_structured(
        self,
        *,
        system_instruction: str,
        prompt: str,
        response_model: type[T],
    ) -> StructuredLLMResponse:
        client = self._get_client()

        def _call() -> types.GenerateContentResponse:
            return client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=response_model,
                ),
            )

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(_call),
                    timeout=self.timeout_seconds,
                )
                raw_text = (response.text or "").strip()
                if not raw_text:
                    raise LLMGatewayError("Gemini returned an empty response body")
                parsed = response_model.model_validate_json(raw_text)
                usage = getattr(response, "usage_metadata", None)
                in_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
                out_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
                return StructuredLLMResponse(parsed=parsed, tokens_input=in_tok, tokens_output=out_tok)
            except (ValidationError, LLMGatewayError) as exc:
                last_err = exc
                logger.warning("llm_validation_retry", attempt=attempt + 1, error=str(exc))
            except Exception as exc:
                last_err = exc
                logger.warning("llm_call_retry", attempt=attempt + 1, error=str(exc))
            await asyncio.sleep(0.5 * (2**attempt))

        raise LLMGatewayError(f"Gemini structured generation failed after retries: {last_err}")
