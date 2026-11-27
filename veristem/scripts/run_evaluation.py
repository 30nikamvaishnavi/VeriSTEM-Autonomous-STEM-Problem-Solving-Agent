#!/usr/bin/env python3
"""CLI script to run the VeriSTEM 32-problem evaluation and ablation suite."""

import asyncio
from pathlib import Path

from app.evaluation.runner import run_full_benchmark


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    json_path, csv_path = asyncio.run(
        run_full_benchmark(
            root / "datasets" / "stem_benchmarks_32.json",
            root / "benchmarks" / "results",
        )
    )
    print(f"Evaluation reports written:\n- JSON: {json_path}\n- CSV:  {csv_path}")


if __name__ == "__main__":
    main()
