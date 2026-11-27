import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.database.session import init_db
from app.main import create_app


@pytest.mark.asyncio
async def test_api_problem_lifecycle_and_auth(isolated_env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERISTEM_API_KEY", "secret-test-key")
    get_settings.cache_clear()
    await init_db()
    app = create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        health = await client.get("/api/v1/health")
        assert health.status_code == 200
        assert health.json()["database_connected"] is True

        # Unauthenticated request to protected endpoint returns 401
        unauth = await client.post("/api/v1/problems", json={"question": "Differentiate x^3"})
        assert unauth.status_code == 401

        headers = {"X-API-Key": "secret-test-key"}
        created = await client.post(
            "/api/v1/problems",
            json={"question": "Differentiate x^3 at x=2", "domain_hint": "mathematics"},
            headers=headers,
        )
        assert created.status_code == 201
        pid = created.json()["problem_id"]

        status_res = await client.get(f"/api/v1/problems/{pid}/status", headers=headers)
        assert status_res.status_code == 200
        assert status_res.json()["status"] == "PENDING"

        metrics = await client.get("/api/v1/metrics", headers=headers)
        assert metrics.status_code == 200
        assert metrics.json()["total_problems"] >= 1
