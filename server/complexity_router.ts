import type { ComplexityLevel, ExecutionPath, ProblemDomain } from '../src/types/veristem.ts';

export interface ComplexityDecision {
  complexity_level: ComplexityLevel;
  execution_path: ExecutionPath;
  detected_domain: ProblemDomain;
  problem_category: string;
  rationale: string;
  router_latency_ms: number;
}

const MULTI_AGENT_EXCLUSIONS: RegExp[] = [
  /\b(black[- ]?scholes|monte\s*carlo|option\s+pric|var\b|value\s+at\s+risk|sharpe|capm|volatility|portfolio)\b/i,
  /\b(projectile|kinematics|velocity|acceleration|momentum|force|newton|torque|pendulum|circuit|resistor|capacitor|inductor|magnetic|electric|thermodynamic|entropy|enthalpy|carnot|fluid|reynolds|navier|schrodinger|lagrangian|hamiltonian)\b/i,
  /\b(differential\s+equation|ode\b|pde\b|initial\s+value|boundary\s+value|laplace\s+transform|fourier\s+series|runge[- ]kutta|finite\s+element)\b/i,
  /\b(optimize|optimization|maximize|minimize|lagrange\s+multiplier|linear\s+programming|simplex)\b/i,
  /\b(hypothesis\s+test|confidence\s+interval|p[- ]value|bayes|posterior|markov|regression|time[- ]series|arima)\b/i,
  /\b(prove|proof|derive\s+from\s+first\s+principles|explain\s+why|research|design\s+an\s+experiment)\b/i,
  /\b(clarify|assume|unknown\s+parameter|missing)\b/i,
];

const RESEARCH_CUES: RegExp[] = [
  /\b(navier[- ]stokes|schrodinger|partial\s+differential|pde\b|finite\s+element|stochastic\s+differential|ito'?s?\s+lemma|proof|prove\s+that|derive\s+from\s+first\s+principles|multi[- ]objective)\b/i,
];

const ADVANCED_CUES: RegExp[] = [
  /\b(monte\s*carlo|black[- ]?scholes|portfolio|lagrange\s+multiplier|differential\s+equation|ode\b|runge[- ]kutta|fourier|laplace|coupled|eigenfunction|bayesian|markov)\b/i,
];

const INTEGRAL_PATTERNS: RegExp[] = [
  /\b(definite\s+)?integral\b/i,
  /\bintegrate\b/i,
  /\\int\b/i,
];

const DERIVATIVE_PATTERNS: RegExp[] = [
  /\b(derivative|differentiate|differentiation)\b/i,
  /\bd\/d[a-z]\b/i,
];

const LINEAR_SYSTEM_PATTERNS: RegExp[] = [
  /\b(linear\s+system|system\s+of\s+(linear\s+)?equations)\b/i,
  /=\s*[-+]?\d[\d./]*\s*(?:and|,|;)\s*[^=]+=\s*[-+]?\d/i,
];

const ALGEBRA_EQUATION_PATTERNS: RegExp[] = [
  /\b(solve|find\s+(?:the\s+)?roots?\s+of|roots?\s+of)\b/i,
  /\bfactor\b/i,
];

const MATRIX_PATTERNS: RegExp[] = [
  /\b(determinant|det\s*\(|trace\s+of|eigenvalues?\s+of|inverse\s+of)\b/i,
];

const UNIT_OR_NUMERIC_PATTERNS: RegExp[] = [
  /\bconvert\s+[-+]?\d+(?:\.\d+)?\s+[a-zA-Z/^*\d-]+\s+(?:to|into)\s+[a-zA-Z/^*\d-]+\b/i,
  /^\s*(?:evaluate|compute|calculate|simplify)\s+[-+()\d\s.^*/a-zA-Z_,]+[.?!]?\s*$/i,
];

function inferDomain(question: string, domainHint?: ProblemDomain | null): ProblemDomain {
  if (domainHint) return domainHint;
  if (/\b(option|black[- ]?scholes|portfolio|bond|yield|interest|var\b|sharpe)\b/i.test(question)) {
    return 'quantitative_finance';
  }
  if (/\b(velocity|acceleration|force|mass|joule|watt|newton|projectile|voltage|current|thermodynamic|kg|m\/s)\b/i.test(question)) {
    return 'physics';
  }
  if (
    INTEGRAL_PATTERNS.some((r) => r.test(question)) ||
    DERIVATIVE_PATTERNS.some((r) => r.test(question)) ||
    LINEAR_SYSTEM_PATTERNS.some((r) => r.test(question)) ||
    ALGEBRA_EQUATION_PATTERNS.some((r) => r.test(question)) ||
    MATRIX_PATTERNS.some((r) => r.test(question))
  ) {
    return 'mathematics';
  }
  return 'general_stem';
}

export function classifyProblemComplexity(
  question: string,
  domainHint?: ProblemDomain | null,
): ComplexityDecision {
  const t0 = performance.now();
  const normalized = question.trim();
  const domain = inferDomain(normalized, domainHint);

  const hasExclusion = MULTI_AGENT_EXCLUSIONS.some((rx) => rx.test(normalized));
  const wordCount = normalized.split(/\s+/).filter(Boolean).length;

  if (!hasExclusion && wordCount <= 45) {
    if (INTEGRAL_PATTERNS.some((rx) => rx.test(normalized))) {
      return {
        complexity_level: 'LEVEL_1_SIMPLE',
        execution_path: 'FAST_PATH',
        detected_domain: 'mathematics',
        problem_category: 'calculus_integral',
        rationale: 'Single-integral expression eligible for deterministic symbolic integration + Python quadrature verification.',
        router_latency_ms: Number((performance.now() - t0).toFixed(3)),
      };
    }

    if (DERIVATIVE_PATTERNS.some((rx) => rx.test(normalized))) {
      return {
        complexity_level: 'LEVEL_1_SIMPLE',
        execution_path: 'FAST_PATH',
        detected_domain: 'mathematics',
        problem_category: 'calculus_derivative',
        rationale: 'Direct differentiation problem eligible for deterministic symbolic diff + finite-difference verification.',
        router_latency_ms: Number((performance.now() - t0).toFixed(3)),
      };
    }

    if (LINEAR_SYSTEM_PATTERNS.some((rx) => rx.test(normalized))) {
      return {
        complexity_level: 'LEVEL_1_SIMPLE',
        execution_path: 'FAST_PATH',
        detected_domain: 'mathematics',
        problem_category: 'linear_system',
        rationale: 'Explicit linear system eligible for direct LU solve + Python residual norm verification.',
        router_latency_ms: Number((performance.now() - t0).toFixed(3)),
      };
    }

    if (MATRIX_PATTERNS.some((rx) => rx.test(normalized))) {
      return {
        complexity_level: 'LEVEL_1_SIMPLE',
        execution_path: 'FAST_PATH',
        detected_domain: 'mathematics',
        problem_category: 'linear_algebra_matrix',
        rationale: 'Matrix calculation eligible for direct algebraic computation + independent Python check.',
        router_latency_ms: Number((performance.now() - t0).toFixed(3)),
      };
    }

    if (ALGEBRA_EQUATION_PATTERNS.some((rx) => rx.test(normalized)) && normalized.includes('=')) {
      return {
        complexity_level: 'LEVEL_1_SIMPLE',
        execution_path: 'FAST_PATH',
        detected_domain: 'mathematics',
        problem_category: 'algebraic_equation',
        rationale: 'Single-equation algebraic problem eligible for exact root solver + residual verification.',
        router_latency_ms: Number((performance.now() - t0).toFixed(3)),
      };
    }

    if (UNIT_OR_NUMERIC_PATTERNS.some((rx) => rx.test(normalized))) {
      return {
        complexity_level: 'LEVEL_1_SIMPLE',
        execution_path: 'FAST_PATH',
        detected_domain: domain,
        problem_category: 'numerical_or_unit_eval',
        rationale: 'Direct numerical evaluation or unit conversion eligible for deterministic solver + cross-check.',
        router_latency_ms: Number((performance.now() - t0).toFixed(3)),
      };
    }
  }

  if (RESEARCH_CUES.some((rx) => rx.test(normalized)) || wordCount > 110) {
    return {
      complexity_level: 'LEVEL_4_RESEARCH',
      execution_path: 'MULTI_AGENT_WORKFLOW',
      detected_domain: domain,
      problem_category: 'research_multistep',
      rationale: 'Open-ended or high-dimensional scientific reasoning requiring full multi-agent orchestration.',
      router_latency_ms: Number((performance.now() - t0).toFixed(3)),
    };
  }

  if (ADVANCED_CUES.some((rx) => rx.test(normalized)) || wordCount > 55) {
    return {
      complexity_level: 'LEVEL_3_ADVANCED',
      execution_path: 'MULTI_AGENT_WORKFLOW',
      detected_domain: domain,
      problem_category: 'advanced_multistep',
      rationale: 'Advanced mathematical, physical, or financial model requiring planning, solver, and multi-method verification.',
      router_latency_ms: Number((performance.now() - t0).toFixed(3)),
    };
  }

  return {
    complexity_level: 'LEVEL_2_INTERMEDIATE',
    execution_path: 'MULTI_AGENT_WORKFLOW',
    detected_domain: domain,
    problem_category: 'intermediate_stem',
    rationale: 'Intermediate multi-step STEM problem requiring semantic parsing, planning, and verified computation.',
    router_latency_ms: Number((performance.now() - t0).toFixed(3)),
  };
}
