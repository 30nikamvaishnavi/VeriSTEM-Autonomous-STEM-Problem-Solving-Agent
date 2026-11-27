import { spawn, spawnSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { create, all } from 'mathjs';
import type { ToolExecutionRecord } from '../src/types/veristem.ts';

const math = create(all, {});

export interface ToolSpec {
  name: string;
  description: string;
  parameters: string;
}

export const TOOL_CATALOG: ToolSpec[] = [
  {
    name: 'symbolic_math',
    description:
      'Symbolic differentiation, algebraic simplification, equation evaluation, and matrix determinants/eigenvalues via mathjs.',
    parameters:
      '{ "operation": "diff|simplify|evaluate|matrix_det|linear_solve", "expression": "string", "variable": "x", "substitutions": {"x": 2}, "matrix_a": [[...]], "vector_b": [...] }',
  },
  {
    name: 'dimensional_analysis',
    description: 'Unit conversion and dimensional consistency verification via mathjs unit engine.',
    parameters:
      '{ "operation": "convert|check_equivalence", "value": 72, "from_unit": "km/hour", "to_unit": "m/s", "lhs_unit": "m/s^2", "rhs_unit": "N/kg" }',
  },
  {
    name: 'independent_verifier',
    description:
      'Independent numerical verification in isolated Python 3.11: adaptive Simpson quadrature, central finite differences, Black-Scholes vs antithetic Monte Carlo, root residuals, and tolerance checks.',
    parameters:
      '{ "verification_type": "integral_quadrature|derivative_finite_diff|root_residual|black_scholes_mc|numeric_tolerance", "expression": "x**2 * math.exp(-x)", "variable": "x", "lower_limit": 0, "upper_limit": 2, "point": 1.0, "candidate_value": 0.646647, "expected_reference": 0.646647, "tolerance": 1e-4, "spot": 100, "strike": 105, "rate": 0.05, "volatility": 0.2, "time_to_maturity": 1.0, "option_type": "call" }',
  },
  {
    name: 'python_sandbox',
    description:
      'Isolated Python 3.11 subprocess sandbox (math, cmath, decimal, fractions, statistics) with AST security guard, memory/CPU caps, no network, and RESULT variable capture.',
    parameters: '{ "code": "import math\\nRESULT = math.sin(math.pi/4)" }',
  },
];

const RUNNER_SOURCE = `
import ast
import json
import sys

try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (6, 6))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
except Exception:
    pass

try:
    import socket
    def _blocked(*args, **kwargs):
        raise RuntimeError("Network disabled in VeriSTEM sandbox")
    socket.socket = _blocked
except Exception:
    pass

ALLOWED_MODULES = {"math", "cmath", "decimal", "fractions", "statistics", "json", "itertools"}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "open", "__import__", "input", "breakpoint", "getattr", "setattr", "delattr", "globals", "locals", "vars"}

user_code = sys.stdin.read()
tree = ast.parse(user_code, mode="exec")
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name.split(".")[0] not in ALLOWED_MODULES:
                raise RuntimeError(f"Import '{alias.name}' prohibited")
    elif isinstance(node, ast.ImportFrom):
        mod = (node.module or "").split(".")[0]
        if mod not in ALLOWED_MODULES:
            raise RuntimeError(f"Import from '{node.module}' prohibited")
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            raise RuntimeError(f"Call '{node.func.id}' prohibited")
    elif isinstance(node, ast.Attribute):
        if node.attr.startswith("__"):
            raise RuntimeError(f"Dunder attribute '{node.attr}' prohibited")

ns = {"RESULT": None}
exec(compile(tree, "<sandbox>", "exec"), ns, ns)
print("__VERISTEM_RESULT__:" + json.dumps(ns.get("RESULT"), default=str))
`;

function parsePythonOutput(stdoutRaw: string, stderrRaw: string, exitCode: number): {
  stdout: string;
  stderr: string;
  result: unknown;
  exitCode: number;
} {
  let result: unknown = null;
  const stdoutLines: string[] = [];
  for (const line of stdoutRaw.split('\n')) {
    if (line.startsWith('__VERISTEM_RESULT__:')) {
      result = JSON.parse(line.slice('__VERISTEM_RESULT__:'.length));
    } else if (line.length > 0) {
      stdoutLines.push(line);
    }
  }
  return {
    stdout: stdoutLines.join('\n').slice(0, 4000),
    stderr: stderrRaw.slice(0, 2000),
    result,
    exitCode,
  };
}

function runIsolatedPython(code: string, timeoutMs = 8000): {
  stdout: string;
  stderr: string;
  result: unknown;
  exitCode: number;
} {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'veristem_py_'));
  const runnerPath = path.join(tmpDir, 'runner.py');

  try {
    fs.writeFileSync(runnerPath, RUNNER_SOURCE, 'utf-8');
    const proc = spawnSync('python3', ['-I', runnerPath], {
      input: code,
      encoding: 'utf-8',
      timeout: timeoutMs,
      cwd: tmpDir,
      env: {
        PATH: process.env.PATH ?? '/usr/bin:/bin',
        PYTHONNOUSERSITE: '1',
        PYTHONDONTWRITEBYTECODE: '1',
      },
      maxBuffer: 1024 * 1024,
    });

    if (proc.error) {
      throw proc.error;
    }

    return parsePythonOutput(proc.stdout || '', proc.stderr || '', proc.status ?? 1);
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

export async function runIsolatedPythonAsync(code: string, timeoutMs = 8000): Promise<{
  stdout: string;
  stderr: string;
  result: unknown;
  exitCode: number;
}> {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'veristem_py_'));
  const runnerPath = path.join(tmpDir, 'runner.py');
  fs.writeFileSync(runnerPath, RUNNER_SOURCE, 'utf-8');

  return await new Promise((resolve, reject) => {
    const child = spawn('python3', ['-I', runnerPath], {
      cwd: tmpDir,
      env: {
        PATH: process.env.PATH ?? '/usr/bin:/bin',
        PYTHONNOUSERSITE: '1',
        PYTHONDONTWRITEBYTECODE: '1',
      },
    });

    let stdout = '';
    let stderr = '';
    let settled = false;

    const timer = setTimeout(() => {
      if (!settled) {
        settled = true;
        child.kill('SIGKILL');
        fs.rmSync(tmpDir, { recursive: true, force: true });
        reject(new Error(`Python sandbox timed out after ${timeoutMs}ms`));
      }
    }, timeoutMs);

    child.stdout.on('data', (chunk: Buffer) => {
      if (stdout.length < 1024 * 1024) stdout += chunk.toString('utf-8');
    });
    child.stderr.on('data', (chunk: Buffer) => {
      if (stderr.length < 1024 * 1024) stderr += chunk.toString('utf-8');
    });
    child.on('error', (err) => {
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        fs.rmSync(tmpDir, { recursive: true, force: true });
        reject(err);
      }
    });
    child.on('close', (codeExit) => {
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        fs.rmSync(tmpDir, { recursive: true, force: true });
        try {
          resolve(parsePythonOutput(stdout, stderr, codeExit ?? 1));
        } catch (err) {
          reject(err);
        }
      }
    });

    child.stdin.write(code);
    child.stdin.end();
  });
}

function executeSyncMathOrUnits(toolName: string, inputPayload: Record<string, unknown>): Record<string, unknown> {
  if (toolName === 'symbolic_math') {
    const op = String(inputPayload.operation ?? 'evaluate');
    const expr = String(inputPayload.expression ?? '');
    const variable = String(inputPayload.variable ?? 'x');
    const subs = (inputPayload.substitutions as Record<string, number> | undefined) ?? {};

    if (op === 'diff') {
      const d = math.derivative(expr, variable);
      const evaluated = Object.keys(subs).length > 0 ? Number(d.evaluate(subs)) : null;
      return { operation: op, result_str: d.toString(), result_latex: d.toTex(), numeric_approx: evaluated };
    }
    if (op === 'simplify') {
      const s = math.simplify(expr);
      return { operation: op, result_str: s.toString(), result_latex: s.toTex() };
    }
    if (op === 'evaluate') {
      const val = math.evaluate(expr, subs);
      return {
        operation: op,
        result_str: String(val),
        numeric_approx: typeof val === 'number' && Number.isFinite(val) ? val : null,
      };
    }
    if (op === 'matrix_det') {
      const mat = inputPayload.matrix_a as number[][];
      const det = math.det(mat);
      return { operation: op, determinant: det };
    }
    if (op === 'linear_solve') {
      const a = inputPayload.matrix_a as number[][];
      const b = inputPayload.vector_b as number[];
      const sol = math.lusolve(a, b) as number[][];
      const flat = sol.map((r) => (Array.isArray(r) ? r[0] : Number(r)));
      const residual = math.norm(math.subtract(math.multiply(a, flat), b));
      return { operation: op, solution: flat, residual_norm: Number(residual) };
    }
    throw new Error(`Unsupported symbolic_math operation: ${op}`);
  }

  if (toolName === 'dimensional_analysis') {
    const op = String(inputPayload.operation ?? 'convert');
    if (op === 'convert') {
      const val = Number(inputPayload.value);
      const fromU = String(inputPayload.from_unit);
      const toU = String(inputPayload.to_unit);
      const converted = math.unit(val, fromU).toNumber(toU);
      return {
        operation: op,
        consistent: true,
        converted_value: converted,
        result_unit: toU,
        details: `${val} ${fromU} = ${converted} ${toU}`,
      };
    }
    if (op === 'check_equivalence') {
      const lhs = math.unit(1, String(inputPayload.lhs_unit));
      const rhs = math.unit(1, String(inputPayload.rhs_unit));
      const consistent = lhs.equalBase(rhs);
      return {
        operation: op,
        consistent,
        details: consistent ? 'Dimensions match' : `Mismatch: ${lhs.formatUnits()} vs ${rhs.formatUnits()}`,
      };
    }
    throw new Error(`Unsupported dimensional_analysis operation: ${op}`);
  }

  throw new Error(`Unsupported sync tool: ${toolName}`);
}

function buildVerifierPythonCode(inputPayload: Record<string, unknown>): { pyCode: string; label: string } {
  const vtype = String(inputPayload.verification_type);
  const candidate = Number(inputPayload.candidate_value);
  const tol = Number(inputPayload.tolerance ?? 1e-4);

  if (vtype === 'integral_quadrature') {
    const expr = String(inputPayload.expression);
    const variable = String(inputPayload.variable ?? 'x');
    const lo = Number(inputPayload.lower_limit);
    const hi = Number(inputPayload.upper_limit);
    return {
      label: 'Simpson quadrature',
      pyCode: `
import math
def f(${variable}):
    return ${expr}
n = 2000
h = (${hi} - ${lo}) / n
s = f(${lo}) + f(${hi})
for i in range(1, n):
    coef = 4 if i % 2 == 1 else 2
    s += coef * f(${lo} + i * h)
quad = s * h / 3.0
abs_err = abs(${candidate} - quad)
rel_err = abs_err / max(abs(quad), 1e-12)
RESULT = {
    "passed": abs_err <= ${tol} or rel_err <= ${tol},
    "method_used": "composite_simpson_quadrature_n2000",
    "independent_value": quad,
    "candidate_value": ${candidate},
    "absolute_error": abs_err,
    "relative_error": rel_err,
    "tolerance": ${tol},
    "details": f"Composite Simpson (n=2000) = {quad:.15g}"
}
`,
    };
  }

  if (vtype === 'derivative_finite_diff') {
    const expr = String(inputPayload.expression);
    const variable = String(inputPayload.variable ?? 'x');
    const pt = Number(inputPayload.point ?? 1.0);
    return {
      label: 'Finite difference check',
      pyCode: `
import math
def f(${variable}):
    return ${expr}
h = 1e-6
fd = (f(${pt} + h) - f(${pt} - h)) / (2.0 * h)
abs_err = abs(${candidate} - fd)
rel_err = abs_err / max(abs(fd), 1e-12)
RESULT = {
    "passed": abs_err <= ${tol} or rel_err <= ${tol},
    "method_used": "central_finite_difference",
    "independent_value": fd,
    "candidate_value": ${candidate},
    "absolute_error": abs_err,
    "relative_error": rel_err,
    "tolerance": ${tol},
    "details": f"Central finite difference at ${variable}=${pt}: {fd:.12g}"
}
`,
    };
  }

  if (vtype === 'root_residual') {
    const expr = String(inputPayload.expression);
    const variable = String(inputPayload.variable ?? 'x');
    return {
      label: 'Root residual check',
      pyCode: `
import math
def f(${variable}):
    return ${expr}
res = abs(f(${candidate}))
RESULT = {
    "passed": res <= ${tol},
    "method_used": "direct_function_residual",
    "independent_value": 0.0,
    "candidate_value": res,
    "absolute_error": res,
    "relative_error": res,
    "tolerance": ${tol},
    "details": f"|f(${candidate})| = {res:.3e}"
}
`,
    };
  }

  if (vtype === 'black_scholes_mc') {
    const s0 = Number(inputPayload.spot);
    const k = Number(inputPayload.strike);
    const r = Number(inputPayload.rate);
    const sigma = Number(inputPayload.volatility);
    const t = Number(inputPayload.time_to_maturity);
    const optType = String(inputPayload.option_type ?? 'call');
    return {
      label: 'Monte Carlo check',
      pyCode: `
import math
state = 42
def rand_u():
    global state
    state = (1664525 * state + 1013904223) % (2**32)
    return (state + 0.5) / (2**32)

total = 0.0
n_pairs = 25000
for _ in range(n_pairs):
    u1 = rand_u()
    u2 = rand_u()
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    for sign in (1.0, -1.0):
        st = ${s0} * math.exp((${r} - 0.5 * ${sigma}**2) * ${t} + ${sigma} * math.sqrt(${t}) * (sign * z))
        payoff = max(st - ${k}, 0.0) if "${optType}" == "call" else max(${k} - st, 0.0)
        total += payoff
mc = math.exp(-${r} * ${t}) * (total / (2 * n_pairs))
abs_err = abs(${candidate} - mc)
mc_tol = max(${tol}, 0.025 * max(mc, 1.0))
RESULT = {
    "passed": abs_err <= mc_tol,
    "method_used": "antithetic_monte_carlo_50k",
    "independent_value": mc,
    "candidate_value": ${candidate},
    "absolute_error": abs_err,
    "relative_error": abs_err / max(abs(mc), 1e-12),
    "tolerance": mc_tol,
    "details": f"Antithetic Monte Carlo (50,000 paths) = {mc:.6f}"
}
`,
    };
  }

  throw new Error(`Unsupported verification_type: ${vtype}`);
}

export function executeTool(toolName: string, inputPayload: Record<string, unknown>): ToolExecutionRecord {
  const callId = `tc_${crypto.randomBytes(6).toString('hex')}`;
  const t0 = performance.now();

  try {
    let output: Record<string, unknown> = {};

    if (toolName === 'symbolic_math' || toolName === 'dimensional_analysis') {
      output = executeSyncMathOrUnits(toolName, inputPayload);
    } else if (toolName === 'independent_verifier') {
      const vtype = String(inputPayload.verification_type);
      const candidate = Number(inputPayload.candidate_value);
      const tol = Number(inputPayload.tolerance ?? 1e-4);

      if (vtype === 'numeric_tolerance') {
        const ref = Number(inputPayload.expected_reference);
        const absErr = Math.abs(candidate - ref);
        const relErr = absErr / Math.max(Math.abs(ref), 1e-12);
        const passed = absErr <= tol || relErr <= tol;
        output = {
          passed,
          method_used: 'direct_tolerance_comparison',
          independent_value: ref,
          candidate_value: candidate,
          absolute_error: absErr,
          relative_error: relErr,
          tolerance: tol,
          details: `abs_err=${absErr.toExponential(3)}, rel_err=${relErr.toExponential(3)}`,
        };
      } else {
        const { pyCode, label } = buildVerifierPythonCode(inputPayload);
        const pyOut = runIsolatedPython(pyCode);
        if (pyOut.exitCode !== 0 || !pyOut.result) {
          throw new Error(pyOut.stderr || `${label} failed`);
        }
        output = pyOut.result as Record<string, unknown>;
      }
    } else if (toolName === 'python_sandbox') {
      const code = String(inputPayload.code ?? '');
      if (!code.trim()) {
        throw new Error('code must be non-empty');
      }
      const pyOut = runIsolatedPython(code);
      if (pyOut.exitCode !== 0) {
        throw new Error(pyOut.stderr || `Python sandbox exited with code ${pyOut.exitCode}`);
      }
      output = {
        stdout: pyOut.stdout,
        result: pyOut.result,
        exit_code: pyOut.exitCode,
      };
    } else {
      throw new Error(`Unknown tool: ${toolName}`);
    }

    return {
      tool_call_id: callId,
      tool_name: toolName,
      input_payload: inputPayload,
      output_payload: output,
      success: true,
      error_message: null,
      duration_ms: Number((performance.now() - t0).toFixed(2)),
      executed_at: new Date().toISOString(),
    };
  } catch (err) {
    return {
      tool_call_id: callId,
      tool_name: toolName,
      input_payload: inputPayload,
      output_payload: {},
      success: false,
      error_message: err instanceof Error ? err.message : String(err),
      duration_ms: Number((performance.now() - t0).toFixed(2)),
      executed_at: new Date().toISOString(),
    };
  }
}

export async function executeToolAsync(
  toolName: string,
  inputPayload: Record<string, unknown>,
): Promise<ToolExecutionRecord> {
  const callId = `tc_${crypto.randomBytes(6).toString('hex')}`;
  const t0 = performance.now();

  try {
    let output: Record<string, unknown> = {};

    if (toolName === 'symbolic_math' || toolName === 'dimensional_analysis') {
      output = executeSyncMathOrUnits(toolName, inputPayload);
    } else if (toolName === 'independent_verifier') {
      const vtype = String(inputPayload.verification_type);
      const candidate = Number(inputPayload.candidate_value);
      const tol = Number(inputPayload.tolerance ?? 1e-4);

      if (vtype === 'numeric_tolerance') {
        const ref = Number(inputPayload.expected_reference);
        const absErr = Math.abs(candidate - ref);
        const relErr = absErr / Math.max(Math.abs(ref), 1e-12);
        const passed = absErr <= tol || relErr <= tol;
        output = {
          passed,
          method_used: 'direct_tolerance_comparison',
          independent_value: ref,
          candidate_value: candidate,
          absolute_error: absErr,
          relative_error: relErr,
          tolerance: tol,
          details: `abs_err=${absErr.toExponential(3)}, rel_err=${relErr.toExponential(3)}`,
        };
      } else {
        const { pyCode, label } = buildVerifierPythonCode(inputPayload);
        const pyOut = await runIsolatedPythonAsync(pyCode);
        if (pyOut.exitCode !== 0 || !pyOut.result) {
          throw new Error(pyOut.stderr || `${label} failed`);
        }
        output = pyOut.result as Record<string, unknown>;
      }
    } else if (toolName === 'python_sandbox') {
      const code = String(inputPayload.code ?? '');
      if (!code.trim()) {
        throw new Error('code must be non-empty');
      }
      const pyOut = await runIsolatedPythonAsync(code);
      if (pyOut.exitCode !== 0) {
        throw new Error(pyOut.stderr || `Python sandbox exited with code ${pyOut.exitCode}`);
      }
      output = {
        stdout: pyOut.stdout,
        result: pyOut.result,
        exit_code: pyOut.exitCode,
      };
    } else {
      throw new Error(`Unknown tool: ${toolName}`);
    }

    return {
      tool_call_id: callId,
      tool_name: toolName,
      input_payload: inputPayload,
      output_payload: output,
      success: true,
      error_message: null,
      duration_ms: Number((performance.now() - t0).toFixed(2)),
      executed_at: new Date().toISOString(),
    };
  } catch (err) {
    return {
      tool_call_id: callId,
      tool_name: toolName,
      input_payload: inputPayload,
      output_payload: {},
      success: false,
      error_message: err instanceof Error ? err.message : String(err),
      duration_ms: Number((performance.now() - t0).toFixed(2)),
      executed_at: new Date().toISOString(),
    };
  }
}

function stableStringify(obj: unknown): string {
  if (obj === null || typeof obj !== 'object') {
    return JSON.stringify(obj);
  }
  if (Array.isArray(obj)) {
    return `[${obj.map(stableStringify).join(',')}]`;
  }
  const entries = Object.entries(obj as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b));
  return `{${entries.map(([k, v]) => `${JSON.stringify(k)}:${stableStringify(v)}`).join(',')}}`;
}

export async function executeToolsParallelWithCache(
  calls: Array<{ task_id: string; tool_name: string; arguments_json: string }>,
  cache: Map<string, ToolExecutionRecord>,
): Promise<ToolExecutionRecord[]> {
  const inflight = new Map<string, Promise<ToolExecutionRecord>>();

  const tasks = calls.map(async (call) => {
    let parsedArgs: Record<string, unknown>;
    try {
      parsedArgs = JSON.parse(call.arguments_json) as Record<string, unknown>;
    } catch (err) {
      return {
        tool_call_id: `tc_err_${call.task_id}`,
        tool_name: call.tool_name,
        input_payload: { raw: call.arguments_json },
        output_payload: {},
        success: false,
        error_message: err instanceof Error ? err.message : String(err),
        duration_ms: 0,
        executed_at: new Date().toISOString(),
      } satisfies ToolExecutionRecord;
    }

    const cacheKey = `${call.tool_name}:${stableStringify(parsedArgs)}`;
    const cached = cache.get(cacheKey);
    if (cached) {
      return {
        ...cached,
        tool_call_id: `tc_cached_${crypto.randomBytes(4).toString('hex')}`,
        duration_ms: 0,
        executed_at: new Date().toISOString(),
      };
    }

    let pending = inflight.get(cacheKey);
    if (!pending) {
      pending = executeToolAsync(call.tool_name, parsedArgs).then((res) => {
        if (res.success) {
          cache.set(cacheKey, res);
        }
        return res;
      });
      inflight.set(cacheKey, pending);
    }
    return pending;
  });

  return Promise.all(tasks);
}
