# ADR-003: Mandatory Independent Computational Verification

- **Status**: Accepted
- **Context**: LLMs frequently produce plausible but numerically incorrect answers or subtle sign/unit errors.
- **Decision**: Separate `STEMSolverAgent` from `IndependentVerificationAgent`. Enforce deterministic code guards that reject `PASS` when zero verification tools succeeded or any tool reports a failing check, and prohibit presenting LLM confidence as proof.
- **Consequences**: False acceptance rate is minimized and every verified output includes reproducible computational evidence.
