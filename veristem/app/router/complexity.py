import re
import time
from dataclasses import dataclass

from app.schemas.models import ComplexityLevel, ExecutionPath, ProblemDomain


@dataclass(frozen=True)
class ComplexityDecision:
    complexity_level: ComplexityLevel
    execution_path: ExecutionPath
    detected_domain: ProblemDomain
    problem_category: str
    rationale: str
    router_latency_ms: float


_MULTI_AGENT_EXCLUSIONS = [
    re.compile(r"\b(black[- ]?scholes|monte\s*carlo|option\s+pric|var\b|value\s+at\s+risk|sharpe|capm|volatility|portfolio)\b", re.I),
    re.compile(r"\b(projectile|kinematics|velocity|acceleration|momentum|force|newton|torque|pendulum|circuit|resistor|capacitor|inductor|magnetic|electric|thermodynamic|entropy|enthalpy|carnot|fluid|reynolds|navier|schrodinger|lagrangian|hamiltonian)\b", re.I),
    re.compile(r"\b(differential\s+equation|ode\b|pde\b|initial\s+value|boundary\s+value|laplace\s+transform|fourier\s+series|runge[- ]kutta|finite\s+element)\b", re.I),
    re.compile(r"\b(optimize|optimization|maximize|minimize|lagrange\s+multiplier|linear\s+programming|simplex)\b", re.I),
    re.compile(r"\b(hypothesis\s+test|confidence\s+interval|p[- ]value|bayes|posterior|markov|regression|time[- ]series|arima)\b", re.I),
    re.compile(r"\b(prove|proof|derive\s+from\s+first\s+principles|explain\s+why|research|design\s+an\s+experiment)\b", re.I),
    re.compile(r"\b(clarify|assume|unknown\s+parameter|missing)\b", re.I),
]

_RESEARCH_CUES = [
    re.compile(r"\b(navier[- ]stokes|schrodinger|partial\s+differential|pde\b|finite\s+element|stochastic\s+differential|ito'?s?\s+lemma|proof|prove\s+that|derive\s+from\s+first\s+principles|multi[- ]objective)\b", re.I),
]

_ADVANCED_CUES = [
    re.compile(r"\b(monte\s*carlo|black[- ]?scholes|portfolio|lagrange\s+multiplier|differential\s+equation|ode\b|runge[- ]kutta|fourier|laplace|coupled|eigenfunction|bayesian|markov)\b", re.I),
]

_INTEGRAL_PATTERNS = [re.compile(r"\b(definite\s+)?integral\b", re.I), re.compile(r"\bintegrate\b", re.I), re.compile(r"\\int\b", re.I)]
_DERIVATIVE_PATTERNS = [re.compile(r"\b(derivative|differentiate|differentiation)\b", re.I), re.compile(r"\bd/d[a-z]\b", re.I)]
_LINEAR_SYSTEM_PATTERNS = [
    re.compile(r"\b(linear\s+system|system\s+of\s+(linear\s+)?equations)\b", re.I),
    re.compile(r"=\s*[-+]?\d[\d./]*\s*(?:and|,|;)\s*[^=]+=\s*[-+]?\d", re.I),
]
_ALGEBRA_PATTERNS = [re.compile(r"\b(solve|find\s+(?:the\s+)?roots?\s+of|roots?\s+of)\b", re.I)]
_MATRIX_PATTERNS = [re.compile(r"\b(determinant|det\s*\(|trace\s+of|eigenvalues?\s+of|inverse\s+of)\b", re.I)]
_UNIT_OR_NUMERIC_PATTERNS = [
    re.compile(r"\bconvert\s+[-+]?\d+(?:\.\d+)?\s+[a-zA-Z/^*\d-]+\s+(?:to|into)\s+[a-zA-Z/^*\d-]+\b", re.I),
    re.compile(r"^\s*(?:evaluate|compute|calculate|simplify)\s+[-+()\d\s.^*/a-zA-Z_,]+[.?!]?\s*$", re.I),
]


def _infer_domain(question: str, domain_hint: ProblemDomain | None) -> ProblemDomain:
    if domain_hint is not None:
        return domain_hint
    if re.search(r"\b(option|black[- ]?scholes|portfolio|bond|yield|interest|var\b|sharpe)\b", question, re.I):
        return ProblemDomain.QUANTITATIVE_FINANCE
    if re.search(r"\b(velocity|acceleration|force|mass|joule|watt|newton|projectile|voltage|current|thermodynamic|kg|m/s)\b", question, re.I):
        return ProblemDomain.PHYSICS
    if any(
        rx.search(question)
        for rx in (
            *_INTEGRAL_PATTERNS,
            *_DERIVATIVE_PATTERNS,
            *_LINEAR_SYSTEM_PATTERNS,
            *_ALGEBRA_PATTERNS,
            *_MATRIX_PATTERNS,
        )
    ):
        return ProblemDomain.MATHEMATICS
    return ProblemDomain.GENERAL_STEM


def classify_problem_complexity(
    question: str, domain_hint: ProblemDomain | None = None
) -> ComplexityDecision:
    t0 = time.perf_counter()
    normalized = question.strip()
    domain = _infer_domain(normalized, domain_hint)
    has_exclusion = any(rx.search(normalized) for rx in _MULTI_AGENT_EXCLUSIONS)
    word_count = len(normalized.split())

    def _elapsed() -> float:
        return round((time.perf_counter() - t0) * 1000.0, 3)

    if not has_exclusion and word_count <= 45:
        if any(rx.search(normalized) for rx in _INTEGRAL_PATTERNS):
            return ComplexityDecision(
                complexity_level=ComplexityLevel.LEVEL_1_SIMPLE,
                execution_path=ExecutionPath.FAST_PATH,
                detected_domain=ProblemDomain.MATHEMATICS,
                problem_category="calculus_integral",
                rationale="Single integral eligible for deterministic SymPy Fast Path + numerical quadrature.",
                router_latency_ms=_elapsed(),
            )
        if any(rx.search(normalized) for rx in _DERIVATIVE_PATTERNS):
            return ComplexityDecision(
                complexity_level=ComplexityLevel.LEVEL_1_SIMPLE,
                execution_path=ExecutionPath.FAST_PATH,
                detected_domain=ProblemDomain.MATHEMATICS,
                problem_category="calculus_derivative",
                rationale="Direct derivative eligible for SymPy differentiation + finite-difference verification.",
                router_latency_ms=_elapsed(),
            )
        if any(rx.search(normalized) for rx in _LINEAR_SYSTEM_PATTERNS):
            return ComplexityDecision(
                complexity_level=ComplexityLevel.LEVEL_1_SIMPLE,
                execution_path=ExecutionPath.FAST_PATH,
                detected_domain=ProblemDomain.MATHEMATICS,
                problem_category="linear_system",
                rationale="Linear system eligible for SymPy/NumPy direct solve + residual norm verification.",
                router_latency_ms=_elapsed(),
            )
        if any(rx.search(normalized) for rx in _MATRIX_PATTERNS):
            return ComplexityDecision(
                complexity_level=ComplexityLevel.LEVEL_1_SIMPLE,
                execution_path=ExecutionPath.FAST_PATH,
                detected_domain=ProblemDomain.MATHEMATICS,
                problem_category="linear_algebra_matrix",
                rationale="Matrix operation eligible for direct linear algebra + verification.",
                router_latency_ms=_elapsed(),
            )
        if any(rx.search(normalized) for rx in _ALGEBRA_PATTERNS) and "=" in normalized:
            return ComplexityDecision(
                complexity_level=ComplexityLevel.LEVEL_1_SIMPLE,
                execution_path=ExecutionPath.FAST_PATH,
                detected_domain=ProblemDomain.MATHEMATICS,
                problem_category="algebraic_equation",
                rationale="Single algebraic equation eligible for SymPy exact solve + residual check.",
                router_latency_ms=_elapsed(),
            )
        if any(rx.search(normalized) for rx in _UNIT_OR_NUMERIC_PATTERNS):
            return ComplexityDecision(
                complexity_level=ComplexityLevel.LEVEL_1_SIMPLE,
                execution_path=ExecutionPath.FAST_PATH,
                detected_domain=domain,
                problem_category="numerical_or_unit_eval",
                rationale="Direct numerical evaluation or unit conversion eligible for deterministic Fast Path.",
                router_latency_ms=_elapsed(),
            )

    if any(rx.search(normalized) for rx in _RESEARCH_CUES) or word_count > 110:
        return ComplexityDecision(
            complexity_level=ComplexityLevel.LEVEL_4_RESEARCH,
            execution_path=ExecutionPath.MULTI_AGENT_WORKFLOW,
            detected_domain=domain,
            problem_category="research_multistep",
            rationale="Complex or research-grade reasoning requiring full LangGraph multi-agent workflow.",
            router_latency_ms=_elapsed(),
        )

    if any(rx.search(normalized) for rx in _ADVANCED_CUES) or word_count > 55:
        return ComplexityDecision(
            complexity_level=ComplexityLevel.LEVEL_3_ADVANCED,
            execution_path=ExecutionPath.MULTI_AGENT_WORKFLOW,
            detected_domain=domain,
            problem_category="advanced_multistep",
            rationale="Advanced problem requiring planning, solver, and multi-method verification.",
            router_latency_ms=_elapsed(),
        )

    return ComplexityDecision(
        complexity_level=ComplexityLevel.LEVEL_2_INTERMEDIATE,
        execution_path=ExecutionPath.MULTI_AGENT_WORKFLOW,
        detected_domain=domain,
        problem_category="intermediate_stem",
        rationale="Intermediate STEM problem requiring structured parsing and multi-agent verification.",
        router_latency_ms=_elapsed(),
    )
