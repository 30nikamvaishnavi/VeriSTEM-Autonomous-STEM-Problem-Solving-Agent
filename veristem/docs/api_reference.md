# VeriSTEM API Reference (`/api/v1`)

All protected endpoints accept an optional `X-API-Key` header when `VERISTEM_API_KEY` is configured and attach `X-Correlation-ID` to every response.

| Method | Endpoint | Description | Status Codes |
|---|---|---|---|
| `POST` | `/api/v1/problems` | Create a new STEM problem record | `201`, `401`, `422`, `429` |
| `POST` | `/api/v1/problems/{problem_id}/solve` | Start verification workflow (supports `idempotency_key`, `run_async`) | `202`, `404`, `409`, `422` |
| `GET` | `/api/v1/problems/{problem_id}` | Full problem detail, timeline, tool records, verification & report | `200`, `404` |
| `GET` | `/api/v1/problems/{problem_id}/status` | Lightweight workflow status & clarification prompt | `200`, `404` |
| `GET` | `/api/v1/problems/{problem_id}/report` | Final verified report (`?format=json` or `?format=markdown`) | `200`, `404` |
| `POST` | `/api/v1/problems/{problem_id}/resume` | Resume paused `NEEDS_CLARIFICATION` or `HUMAN_REVIEW` checkpoint | `202`, `404`, `409` |
| `GET` | `/api/v1/health` | Service, database, and Gemini configuration health check | `200` |
| `GET` | `/api/v1/metrics` | Aggregate execution, verification, and tool telemetry | `200` |
