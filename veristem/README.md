# VeriSTEM — Autonomous, Verifiable STEM Problem-Solving Platform

**VeriSTEM** is a production-grade multi-agent scientific reasoning and computational verification platform across **Mathematics**, **Physics**, and **Quantitative Finance**.

**Core Engineering Principle:**
> *Never accept an LLM-generated numerical or mathematical answer without independent computational verification.*

---

## 1. System Architecture

VeriSTEM separates **probabilistic reasoning** (Gemini API structured generation) from **deterministic computation & independent verification** (SymPy, NumPy, SciPy, Pandas, Pint, and an isolated Python execution sandbox), orchestrated via a stateful **LangGraph** state machine with SQLite/PostgreSQL persistence.

```
+-----------------------------------------------------------------------------+
|                        Clients (React Workbench / Streamlit)                |
+--------------------------------------+--------------------------------------+
                                       | HTTP / JSON (X-API-Key, X-Correlation-ID)
                                       v
+-----------------------------------------------------------------------------+
|                         FastAPI Async REST API (/api/v1)                    |
|  - Rate Limiter (Sliding Window)   - Idempotency Registry                   |
|  - Pydantic v2 Request Validation  - Background Workflow Executor           |
+--------------------------------------+--------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                     LangGraph Orchestrator (StateGraph)                     |
|  Checkpoints: AsyncSqliteSaver  |  Max Correction Attempts: 3               |
|  Nodes: Understand -> Plan -> Solve -> Verify -> [Critic | HumanReview]     |
+------------------+-----------------------------------+----------------------+
                   |                                   |
                   v                                   v
+----------------------------------+  +---------------------------------------+
|  Gemini Gateway (Structured JSON)|  |         Safe Tool Registry            |
|  - Schema-constrained outputs    |  |  1. symbolic_math (SymPy)             |
|  - Retry & exponential backoff   |  |  2. numerical_compute (NumPy)         |
|  - Token & latency telemetry     |  |  3. scientific_opt (SciPy)            |
+----------------------------------+  |  4. statistical_analysis (Pandas)     |
                                      |  5. dimensional_analysis (Pint)       |
                                      |  6. equation_validator (SymPy)        |
                                      |  7. independent_verifier (Cross-check)|
                                      |  8. python_sandbox (Subprocess -I)    |
                                      +---------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                SQLAlchemy Persistence (SQLite / PostgreSQL)                 |
|  problems | execution_plans | agent_states | tool_records |                 |
|  verification_results | retry_history | human_review_decisions | reports    |
+-----------------------------------------------------------------------------+
```

## 2. Agent Interaction Diagram

```
  [START]
     |
     v
+--------------------+    needs_clarification=True     +-------------------+
| 1. Understanding   | ------------------------------> | Clarify Interrupt |
|    Agent           | <------------------------------ | (Resume via API)  |
+--------------------+       clarification_response    +-------------------+
     |
     | needs_clarification=False
     v
+--------------------+
| 2. Planning Agent  |  Selects methods, dependency DAG, and verification plan
+--------------------+
     |
     v
+--------------------+
| 3. STEM Solver     |  Executes SymPy / NumPy / SciPy / Sandbox tools
+--------------------+  <---------------------------------------+
     |                                                          |
     v                                                          |
+--------------------+    status != PASS & attempts < 3         |
| 4. Independent     | ----------------------------------> +-----------+
|    Verifier Agent  |                                     | 5. Critic |
+--------------------+                                     +-----------+
     |                \
     | status == PASS  \ status != PASS & attempts >= 3
     v                  v
+--------------------+  +-----------------------+
| 6. Explanation &   |  | Human Review Interrupt|
|    Reporting Agent |  | (Approve/Override/Rej)|
+--------------------+  +-----------------------+
     |                           |
     +-------------+-------------+
                   v
                 [END]
```

## 3. Repository Structure

```
veristem/
  app/
    api/          # FastAPI v1 REST endpoints
    agents/       # 6 specialized agents with Gemini structured output
    graph/        # LangGraph StateGraph, typed state, conditional edges
    schemas/      # Strict Pydantic v2 contracts
    tools/        # 8 schema-validated computational tools + isolated sandbox
    services/     # GeminiGateway & WorkflowService
    database/     # SQLAlchemy ORM models, async session, repository
    evaluation/   # 32-problem benchmark & ablation runner (CSV/JSON)
    core/         # Settings, structlog, OpenTelemetry tracing, auth, rate limiter
  frontend/       # Streamlit workbench UI
  datasets/       # 32 ground-truth STEM benchmark problems
  benchmarks/     # Evaluation results output directory
  scripts/        # Evaluation CLI runner
  tests/          # unit/, integration/, property/ (Hypothesis), e2e/
  docs/           # Architecture, Agent Design, API Reference, Security, ADRs
```

## 4. Quickstart (Local & Docker)

### Local Python Setup

```bash
cd veristem
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp ../.env.example .env
# Set GEMINI_API_KEY in .env

# Run FastAPI backend
uvicorn app.main:app --host 0.0.0.0 --port 8000

# In another terminal, run Streamlit workbench
streamlit run frontend/streamlit_app.py --server.port 8501
```

### Docker Compose (PostgreSQL + FastAPI + Streamlit)

```bash
cd veristem
GEMINI_API_KEY=your_key_here docker compose up --build
```

## 5. Testing & Quality Checks

All automated tests mock `GeminiGateway` so they run deterministically with zero API calls:

```bash
cd veristem
pytest -v
ruff check app tests
```

## 6. Evaluation & Benchmarking

Run the 32-problem benchmark across 4 modes (`baseline_single_pass`, `ablation_no_verifier`, `ablation_no_critic`, `full_multi_agent`):

```bash
cd veristem
python scripts/run_evaluation.py
```

Outputs are saved to `benchmarks/results/evaluation_report.json` and `benchmarks/results/evaluation_report.csv`.

## 7. Known Limitations

- Symbolic parsing relies on SymPy `parse_expr` transformations; highly custom notation may require the isolated Python sandbox tool.
- The in-process sliding-window rate limiter applies per worker instance; multi-worker deployments should place Redis or an API gateway in front.
- Live Gemini latency and token costs depend on problem complexity and external API availability.
