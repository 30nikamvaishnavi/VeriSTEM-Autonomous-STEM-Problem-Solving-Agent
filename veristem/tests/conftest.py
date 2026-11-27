import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from app.core.config import get_settings
from app.services.gemini_client import StructuredLLMResponse

T = TypeVar("T", bound=BaseModel)


class MockLLMGateway:
    """Deterministic mock Gemini gateway keyed by response model name or scripted queue."""

    def __init__(self, responses: dict[type[BaseModel], list[BaseModel]] | None = None) -> None:
        self.responses: dict[type[BaseModel], list[BaseModel]] = responses or {}
        self.calls: list[str] = []

    async def generate_structured(
        self,
        *,
        system_instruction: str,
        prompt: str,
        response_model: type[T],
    ) -> StructuredLLMResponse:
        self.calls.append(response_model.__name__)
        queue = self.responses.get(response_model)
        if not queue:
            raise AssertionError(f"No mock response queued for {response_model.__name__}")
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        return StructuredLLMResponse(parsed=item, tokens_input=120, tokens_output=85)


@pytest.fixture
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_file = tmp_path / "test_veristem.db"
    ckpt_file = tmp_path / "test_checkpoints.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_file}")
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_DB", str(ckpt_file))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-placeholder")
    monkeypatch.setenv("VERISTEM_API_KEY", "")
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()
