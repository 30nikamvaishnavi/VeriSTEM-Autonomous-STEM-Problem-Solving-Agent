import asyncio
import csv
import json
import statistics
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from app.graph.state import VeriSTEMState
from app.graph.workflow import build_veristem_graph
from app.schemas.models import (
    ProblemSpecification,
    VerificationStatus,
    WorkflowStatus,
    utc_now,
)
from app.services.gemini_client import GeminiGateway, LLMGateway
from app.tools.registry import get_tool_registry

# Gemini 4 Hydrogen reference pricing ($ per 1M tokens) for cost estimation
INPUT_COST_PER_M = 0.15
OUTPUT_COST_PER_M = 0.60


@dataclass
class BenchmarkItem:
    id: str
    domain: str
    category: str
    difficulty: str
    question: str
    expected_answer: str
    expected_numeric: float | None
    expected_unit: str
    tolerance: float
    verification_method: str


def load_benchmark_dataset(path: Path) -> list[BenchmarkItem]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [BenchmarkItem(**item) for item in raw]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(len(sorted_vals) - 1, max(0, int(round((pct / 100.0) * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def _is_numerically_correct(
    predicted: float | None,
    expected: float | None,
    tolerance: float,
) -> tuple[bool, float | None]:
    if predicted is None or expected is None:
        return False, None
    err = abs(predicted - expected)
    rel_ok = err / max(abs(expected), 1e-12) <= max(tolerance, 1e-4)
    return (err <= tolerance or rel_ok), err


async def evaluate_configuration(
    *,
    items: list[BenchmarkItem],
    mode: str,
    llm: LLMGateway,
) -> dict[str, Any]:
    """Run evaluation across items for a given system mode:
    - full_multi_agent: 6-agent LangGraph with verifier + critic (max 3 attempts)
    - ablation_no_critic: verifier enabled, max_correction_attempts=0
    - ablation_no_verifier: solver without independent verification
    - baseline_single_pass: direct LLM structured specification/answer without tools
    """
    registry = get_tool_registry()
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []

    for item in items:
        t0 = time.perf_counter()
        tokens_in = 0
        tokens_out = 0
        predicted_num: float | None = None
        verif_status = VerificationStatus.INCONCLUSIVE.value
        correction_attempts = 0
        unit_ok = True

        if mode == "baseline_single_pass":
            resp = await llm.generate_structured(
                system_instruction="Solve the STEM problem and return ProblemSpecification.",
                prompt=item.question,
                response_model=ProblemSpecification,
            )
            tokens_in = resp.tokens_input
            tokens_out = resp.tokens_output
            verif_status = "SKIPPED"
        else:
            max_retries = 3 if mode == "full_multi_agent" else 0
            checkpointer = MemorySaver()
            graph = build_veristem_graph(llm=llm, registry=registry, checkpointer=checkpointer)
            initial_state: VeriSTEMState = {
                "problem_id": item.id,
                "execution_id": f"eval_{uuid.uuid4().hex[:8]}",
                "question": item.question,
                "domain_hint": item.domain,
                "started_at": utc_now().isoformat(),
                "current_stage": "STARTING",
                "workflow_status": WorkflowStatus.RUNNING.value,
                "max_correction_attempts": max_retries,
                "correction_attempts": 0,
                "correction_history": [],
                "tool_records": [],
                "agent_timeline": [],
            }
            final_state = await graph.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": f"{mode}_{item.id}"}},
            )
            for step in final_state.get("agent_timeline", []):
                tokens_in += int(step.get("tokens_input", 0))
                tokens_out += int(step.get("tokens_output", 0))
            report = final_state.get("final_report") or {}
            predicted_num = report.get("numeric_value")
            verif_status = report.get("verification_status", VerificationStatus.INCONCLUSIVE.value)
            correction_attempts = int(final_state.get("correction_attempts", 0))
            vrep = final_state.get("verification_report") or {}
            if vrep.get("dimensional_consistency_passed") is False:
                unit_ok = False

        duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        latencies.append(duration_ms)
        correct, num_err = _is_numerically_correct(predicted_num, item.expected_numeric, item.tolerance)
        est_cost = (tokens_in / 1e6) * INPUT_COST_PER_M + (tokens_out / 1e6) * OUTPUT_COST_PER_M

        rows.append(
            {
                "mode": mode,
                "problem_id": item.id,
                "domain": item.domain,
                "correct": correct,
                "numeric_error": num_err,
                "verification_status": verif_status,
                "unit_consistent": unit_ok,
                "correction_attempts": correction_attempts,
                "duration_ms": duration_ms,
                "tokens_input": tokens_in,
                "tokens_output": tokens_out,
                "estimated_cost_usd": round(est_cost, 6),
            }
        )

    total = len(rows)
    correct_count = sum(1 for r in rows if r["correct"])
    pass_count = sum(1 for r in rows if r["verification_status"] == VerificationStatus.PASS.value)
    false_accept = sum(
        1 for r in rows if r["verification_status"] == VerificationStatus.PASS.value and not r["correct"]
    )
    corrected_success = sum(1 for r in rows if r["correction_attempts"] > 0 and r["correct"])
    attempted_corrections = sum(1 for r in rows if r["correction_attempts"] > 0)
    errors = [r["numeric_error"] for r in rows if r["numeric_error"] is not None]

    summary = {
        "mode": mode,
        "total_problems": total,
        "accuracy": round(correct_count / total, 4) if total else 0.0,
        "verification_precision": round((pass_count - false_accept) / pass_count, 4) if pass_count else 0.0,
        "false_acceptance_rate": round(false_accept / total, 4) if total else 0.0,
        "self_correction_success_rate": (
            round(corrected_success / attempted_corrections, 4) if attempted_corrections else 0.0
        ),
        "unit_consistency_rate": round(sum(1 for r in rows if r["unit_consistent"]) / total, 4) if total else 0.0,
        "mean_numeric_error": round(statistics.mean(errors), 6) if errors else None,
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p50_latency_ms": round(_percentile(latencies, 50), 2),
        "p95_latency_ms": round(_percentile(latencies, 95), 2),
        "total_tokens": sum(r["tokens_input"] + r["tokens_output"] for r in rows),
        "total_estimated_cost_usd": round(sum(r["estimated_cost_usd"] for r in rows), 5),
    }
    return {"summary": summary, "rows": rows}


def write_evaluation_reports(results: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "evaluation_report.json"
    csv_path = out_dir / "evaluation_report.csv"

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    all_rows = [row for block in results for row in block["rows"]]
    if all_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    return json_path, csv_path


async def run_full_benchmark(dataset_path: Path, output_dir: Path) -> tuple[Path, Path]:
    items = load_benchmark_dataset(dataset_path)
    llm = GeminiGateway()
    modes = [
        "baseline_single_pass",
        "ablation_no_verifier",
        "ablation_no_critic",
        "full_multi_agent",
    ]
    results = []
    for mode in modes:
        results.append(await evaluate_configuration(items=items, mode=mode, llm=llm))
    return write_evaluation_reports(results, output_dir)


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2]
    asyncio.run(
        run_full_benchmark(
            base / "datasets" / "stem_benchmarks_32.json",
            base / "benchmarks" / "results",
        )
    )
