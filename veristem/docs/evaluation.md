# VeriSTEM Evaluation Methodology

## Benchmark Suite
`datasets/stem_benchmarks_32.json` contains **32 validated ground-truth problems**:
- **Mathematics (12)**: Calculus, linear algebra, ODEs, constrained/unconstrained optimization, probability, statistics, numerical root-finding.
- **Physics (10)**: Kinematics, energy conservation, electromagnetism, circuits, thermodynamics, harmonic oscillation, orbital mechanics.
- **Quantitative Finance (10)**: Black-Scholes pricing & Greeks, put-call parity, portfolio variance & Sharpe ratio, parametric VaR, bond pricing & duration, AR(1) variance, GMV portfolio weights.

## Experimental Configurations
1. `baseline_single_pass`: Direct Gemini structured generation without tool execution or verification.
2. `ablation_no_verifier`: Solver tool execution without independent verification.
3. `ablation_no_critic`: Independent verification enabled, `max_correction_attempts=0`.
4. `full_multi_agent`: Full 6-agent LangGraph workflow with independent verifier and up to 3 critic self-correction loops.

## Metrics Recorded
- Final answer numerical accuracy within problem-specific tolerance
- Mean numerical error
- Unit & dimensional consistency rate
- Verification precision & False Acceptance Rate (FAR)
- Self-correction success rate
- Average, P50, and P95 latency (ms)
- Token consumption and estimated API cost (USD)

Note: Actual benchmark measurements are generated only by executing `python scripts/run_evaluation.py` against a configured Gemini API key. No synthetic benchmark numbers are fabricated in documentation.
