# VeriSTEM System Architecture

## 1. Design Philosophy

VeriSTEM enforces **computational grounding**: language models plan, formalize, and synthesize explanations, while all numeric, symbolic, dimensional, and statistical computations execute in deterministic engines and are independently cross-checked.

## 2. Database Schema

VeriSTEM persists full workflow state across 9 relational tables (SQLite via `aiosqlite` locally, PostgreSQL via `psycopg` in production):

| Table | Purpose | Key Columns |
|---|---|---|
| `problems` | Master problem record & status | `id`, `execution_id`, `question`, `domain_hint`, `status`, `current_stage`, `specification_json`, `correction_attempts` |
| `execution_plans` | Generated DAG execution plans | `id`, `problem_id`, `execution_id`, `plan_json` |
| `agent_states` | Per-agent execution timeline | `id`, `problem_id`, `agent_name`, `stage`, `status`, `result_json`, `duration_ms` |
| `tool_records` | Every computational tool call & I/O | `id`, `problem_id`, `tool_call_id`, `tool_name`, `record_json`, `success`, `duration_ms` |
| `verification_results` | Independent verification verdicts | `id`, `problem_id`, `attempt_number`, `status`, `report_json` |
| `retry_history` | Critic root-cause & correction requests | `id`, `problem_id`, `attempt_number`, `correction_json` |
| `human_review_decisions` | Clarifications & reviewer overrides | `id`, `problem_id`, `decision_json` |
| `final_reports` | Final verified JSON & Markdown reports | `id`, `problem_id`, `verification_status`, `report_json` |
| `idempotency_keys` | Idempotent `/solve` deduplication | `idempotency_key`, `problem_id`, `execution_id` |

LangGraph state checkpoints are additionally persisted in `LANGGRAPH_CHECKPOINT_DB` via `AsyncSqliteSaver`, enabling exact node-level `interrupt()` / `Command(resume=...)` recovery across process restarts.

## 3. Implementation Milestones

1. **Stage 1 — Core & Contracts**: Settings, logging, tracing, rate limiter, Pydantic v2 models (`app/schemas/models.py`).
2. **Stage 2 — Persistence & Tools**: SQLAlchemy ORM & repository (`app/database/`), SafeToolRegistry + 8 computational tools including isolated subprocess sandbox (`app/tools/`).
3. **Stage 3 — Agents & LangGraph**: Gemini structured gateway (`app/services/gemini_client.py`), 6 specialized agents (`app/agents/`), conditional LangGraph workflow (`app/graph/workflow.py`).
4. **Stage 4 — API & UI**: FastAPI v1 async endpoints (`app/api/routes.py`), Streamlit UI (`frontend/streamlit_app.py`), React/Express live workbench.
5. **Stage 5 — Evaluation & Tests**: 32-problem benchmark (`datasets/stem_benchmarks_32.json`), ablation runner, unit/integration/Hypothesis/E2E tests.
6. **Stage 6 — Documentation & Packaging**: Dockerfile, Docker Compose, ADRs, security & API docs.
