# VeriSTEM Agent Design Specification

1. **Problem Understanding Agent (`app/agents/understanding.py`)**
   - Input: Raw question, optional domain hint, optional clarification response.
   - Output: `ProblemSpecification` (domain, variables with units, equations, constraints, assumptions, `needs_clarification`).
2. **Planning Agent (`app/agents/planning.py`)**
   - Input: `ProblemSpecification` + tool catalog.
   - Output: Dependency-aware `ExecutionPlan` with ordered `ExecutionTask` entries and an independent `verification_strategy`.
3. **STEM Solver Agent (`app/agents/solver.py`)**
   - Step A: Generates `SolverToolCallPlan` targeting registered tools (`symbolic_math`, `numerical_compute`, `scientific_opt`, `statistical_analysis`, `python_sandbox`).
   - Step B: Executes tools through `SafeToolRegistry` (never inventing tool outputs).
   - Step C: Synthesizes `SolverSynthesis` strictly from actual `ToolExecutionRecord` payloads.
4. **Independent Verification Agent (`app/agents/verifier.py`)**
   - Plans and runs independent cross-checks (`independent_verifier`, `equation_validator`, `dimensional_analysis`).
   - Deterministic safeguard: Forces `INCONCLUSIVE` if zero verification tools succeed, and forces `FAIL` if any tool check reports `passed=False`, `valid=False`, or `consistent=False`.
5. **Critic and Self-Correction Agent (`app/agents/critic.py`)**
   - Invoked only when verification status is `FAIL` or `INCONCLUSIVE` and `correction_attempts < max_correction_attempts` (default `3`).
   - Produces `CorrectionRequest` detailing `root_cause`, `error_category`, and `corrective_instructions`.
6. **Explanation and Reporting Agent (`app/agents/reporter.py`)**
   - Produces `FinalSolutionReport` separating modeling assumptions from verified tool outputs and verification evidence.
