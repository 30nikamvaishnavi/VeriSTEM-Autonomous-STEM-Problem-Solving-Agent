# ADR-001: LangGraph StateGraph Orchestration

- **Status**: Accepted
- **Context**: VeriSTEM requires cyclic self-correction (`Verify -> Critic -> Solve -> Verify`), bounded retries (`max_correction_attempts=3`), human-in-the-loop interrupts for missing information or escalated failures, and crash-resilient checkpointing.
- **Decision**: Use LangGraph `StateGraph` with typed `VeriSTEMState`, conditional routing edges, `interrupt()` / `Command(resume=...)`, and `AsyncSqliteSaver` checkpoint persistence paired with SQLAlchemy relational audit tables.
- **Consequences**: Workflows survive process restarts, loops are deterministically bounded, and every agent transition is auditable.
