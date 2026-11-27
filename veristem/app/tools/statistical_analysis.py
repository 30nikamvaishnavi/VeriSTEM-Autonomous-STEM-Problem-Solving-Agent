from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats


class StatisticalAnalysisInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: str = Field(
        ...,
        pattern="^(descriptive_stats|linear_regression|ttest_one_sample|portfolio_metrics)$",
    )
    data: list[float] = Field(default_factory=list)
    x_data: list[float] | None = None
    y_data: list[float] | None = None
    null_mean: float = 0.0
    weights: list[float] | None = None
    expected_returns: list[float] | None = None
    covariance_matrix: list[list[float]] | None = None
    risk_free_rate: float = 0.02
    confidence_level: float = Field(default=0.95, gt=0.5, lt=0.9999)


class StatisticalAnalysisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: str
    metrics: dict[str, Any]


def run_statistical_analysis(params: StatisticalAnalysisInput) -> StatisticalAnalysisOutput:
    op = params.operation
    if op == "descriptive_stats":
        if not params.data:
            raise ValueError("data must be non-empty")
        s = pd.Series(params.data, dtype="float64")
        return StatisticalAnalysisOutput(
            operation=op,
            metrics={
                "count": int(s.count()),
                "mean": float(s.mean()),
                "std_sample": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                "variance_sample": float(s.var(ddof=1)) if len(s) > 1 else 0.0,
                "median": float(s.median()),
                "min": float(s.min()),
                "max": float(s.max()),
                "q25": float(s.quantile(0.25)),
                "q75": float(s.quantile(0.75)),
            },
        )

    if op == "linear_regression":
        if not params.x_data or not params.y_data or len(params.x_data) != len(params.y_data):
            raise ValueError("Equal-length x_data and y_data are required")
        res = stats.linregress(params.x_data, params.y_data)
        return StatisticalAnalysisOutput(
            operation=op,
            metrics={
                "slope": float(res.slope),
                "intercept": float(res.intercept),
                "r_value": float(res.rvalue),
                "r_squared": float(res.rvalue ** 2),
                "p_value": float(res.pvalue),
                "stderr": float(res.stderr),
            },
        )

    if op == "ttest_one_sample":
        if len(params.data) < 2:
            raise ValueError("At least 2 observations required for t-test")
        t_stat, p_val = stats.ttest_1samp(params.data, popmean=params.null_mean)
        return StatisticalAnalysisOutput(
            operation=op,
            metrics={"t_statistic": float(t_stat), "p_value": float(p_val)},
        )

    if op == "portfolio_metrics":
        if not params.weights or not params.expected_returns or not params.covariance_matrix:
            raise ValueError("weights, expected_returns, and covariance_matrix are required")
        w = np.asarray(params.weights, dtype=np.float64)
        mu = np.asarray(params.expected_returns, dtype=np.float64)
        cov = np.asarray(params.covariance_matrix, dtype=np.float64)
        port_ret = float(w @ mu)
        port_var = float(w.T @ cov @ w)
        port_vol = float(np.sqrt(max(port_var, 0.0)))
        sharpe = (port_ret - params.risk_free_rate) / port_vol if port_vol > 0 else 0.0
        z = float(stats.norm.ppf(1.0 - params.confidence_level))
        parametric_var = -(port_ret + z * port_vol)
        return StatisticalAnalysisOutput(
            operation=op,
            metrics={
                "weight_sum": float(np.sum(w)),
                "expected_return": port_ret,
                "variance": port_var,
                "volatility": port_vol,
                "sharpe_ratio": sharpe,
                "parametric_var": parametric_var,
            },
        )

    raise ValueError(f"Unsupported operation: {op}")
