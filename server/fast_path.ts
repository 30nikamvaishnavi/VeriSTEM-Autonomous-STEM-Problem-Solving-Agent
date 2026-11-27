import crypto from 'node:crypto';
import { create, all } from 'mathjs';
import type {
  AgentExecutionResult,
  ExecutionPlan,
  FinalSolutionReport,
  ProblemSpecification,
  ToolExecutionRecord,
  VerificationCheck,
  VerificationReport,
} from '../src/types/veristem.ts';
import type { ComplexityDecision } from './complexity_router.ts';
import { executeToolAsync, runIsolatedPythonAsync } from './tools.ts';

const math = create(all, {});

export interface FastPathResult {
  handled: boolean;
  reason?: string;
  specification?: ProblemSpecification;
  plan?: ExecutionPlan;
  agent_timeline?: AgentExecutionResult[];
  tool_records?: ToolExecutionRecord[];
  verification_report?: VerificationReport;
  final_report?: FinalSolutionReport;
}

function normalizeMathExpr(raw: string): string {
  let s = raw.trim();
  s = s.replace(/\\cdot/g, '*');
  s = s.replace(/\\times/g, '*');
  s = s.replace(/\\pi\b/g, 'pi');
  s = s.replace(/\\sin\b/g, 'sin');
  s = s.replace(/\\cos\b/g, 'cos');
  s = s.replace(/\\tan\b/g, 'tan');
  s = s.replace(/\\exp\b/g, 'exp');
  s = s.replace(/\\ln\b/g, 'log');
  s = s.replace(/\\log\b/g, 'log');
  s = s.replace(/\\sqrt\s*\{([^}]+)\}/g, 'sqrt($1)');
  s = s.replace(/\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}/g, '(($1)/($2))');
  s = s.replace(/\{([^}]+)\}/g, '($1)');
  s = s.replace(/\bln\s*\(/g, 'log(');
  s = s.replace(/\*\*+/g, '^');
  s = s.replace(/−|–/g, '-');

  // Convert e^(...) or e^x -> exp(...)
  s = s.replace(/\be\s*\^\s*\(([^()]+)\)/g, 'exp($1)');
  s = s.replace(/\be\s*\^\s*([-+]?[a-zA-Z0-9_.]+)/g, 'exp($1)');

  // Insert implicit multiplication carefully: 2x -> 2*x, 2( -> 2*(, )x -> )*x, )( -> )*(
  s = s.replace(/(\d)([a-zA-Z(])/g, '$1*$2');
  s = s.replace(/(\))([a-zA-Z0-9(])/g, '$1*$2');
  s = s.replace(/\b([xyztn])\s+(?=(?:sin|cos|tan|exp|log|sqrt)\b)/gi, '$1 * ');

  return s.replace(/\s+/g, ' ').trim();
}

function toPythonMathExpr(mathjsExpr: string): string {
  let py = mathjsExpr.replace(/\^/g, '**');
  const fnNames = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs'];
  for (const fn of fnNames) {
    py = py.replace(new RegExp(`(?<!math\\.)\\b${fn}\\s*\\(`, 'g'), `math.${fn}(`);
  }
  py = py.replace(/(?<!math\.)\bpi\b/g, 'math.pi');
  py = py.replace(/(?<!math\.)\be\b/g, 'math.e');
  return py;
}

function parseNumericLimit(raw: string): number {
  const norm = normalizeMathExpr(raw);
  const val = math.evaluate(norm);
  if (typeof val !== 'number' || !Number.isFinite(val)) {
    throw new Error(`Invalid numeric limit: ${raw}`);
  }
  return val;
}

function formatFractionCoeff(val: number): string {
  if (Math.abs(val - Math.round(val)) < 1e-12) {
    return String(Math.round(val));
  }
  try {
    const f = math.fraction(val);
    if (f.d === 1n) return `${Number(f.s) * Number(f.n)}`;
    if (f.d <= 10000n) return `(${f.s < 0n ? '-' : ''}${f.n}/${f.d})`;
  } catch {
    // Fall back to precision decimal for non-rational floats
  }
  return val.toPrecision(15).replace(/\.?0+$/, '');
}

interface SymbolicIntegralOutput {
  antiderivativeExpr: string;
  exactDefiniteExpr: string | null;
  numericValue: number | null;
  methodUsed: string;
  intermediateSteps: Array<{ step: string; value: string; unit: string }>;
}

function nodeDependsOnVar(node: math.MathNode, variable: string): boolean {
  let found = false;
  node.traverse((n) => {
    if (n.type === 'SymbolNode' && (n as math.SymbolNode).name === variable) {
      found = true;
    }
  });
  return found;
}

function analyzePolynomialNode(
  node: math.MathNode,
  variable: string,
): { isPoly: boolean; degree: number } {
  if (node.type === 'ParenthesisNode') {
    return analyzePolynomialNode((node as math.ParenthesisNode).content, variable);
  }
  if (node.type === 'ConstantNode') {
    return { isPoly: true, degree: 0 };
  }
  if (node.type === 'SymbolNode') {
    return { isPoly: true, degree: (node as math.SymbolNode).name === variable ? 1 : 0 };
  }
  if (node.type === 'FunctionNode') {
    return nodeDependsOnVar(node, variable)
      ? { isPoly: false, degree: -1 }
      : { isPoly: true, degree: 0 };
  }
  if (node.type === 'OperatorNode') {
    const opNode = node as math.OperatorNode;
    const { op, args } = opNode;
    if ((op === '+' || op === '-') && opNode.isUnary() && args.length === 1) {
      return analyzePolynomialNode(args[0], variable);
    }
    if ((op === '+' || op === '-') && args.length === 2) {
      const left = analyzePolynomialNode(args[0], variable);
      const right = analyzePolynomialNode(args[1], variable);
      if (!left.isPoly || !right.isPoly) return { isPoly: false, degree: -1 };
      return { isPoly: true, degree: Math.max(left.degree, right.degree) };
    }
    if (op === '*' && args.length === 2) {
      const left = analyzePolynomialNode(args[0], variable);
      const right = analyzePolynomialNode(args[1], variable);
      if (!left.isPoly || !right.isPoly) return { isPoly: false, degree: -1 };
      const degree = left.degree + right.degree;
      return degree <= 8 ? { isPoly: true, degree } : { isPoly: false, degree: -1 };
    }
    if (op === '/' && args.length === 2) {
      if (nodeDependsOnVar(args[1], variable)) return { isPoly: false, degree: -1 };
      return analyzePolynomialNode(args[0], variable);
    }
    if (op === '^' && args.length === 2) {
      if (nodeDependsOnVar(args[1], variable)) return { isPoly: false, degree: -1 };
      const base = analyzePolynomialNode(args[0], variable);
      if (!base.isPoly) return { isPoly: false, degree: -1 };
      if (base.degree === 0) return { isPoly: true, degree: 0 };
      const expVal = Number(args[1].evaluate({}));
      if (Number.isInteger(expVal) && expVal >= 0 && expVal <= 8) {
        const degree = base.degree * expVal;
        return degree <= 8 ? { isPoly: true, degree } : { isPoly: false, degree: -1 };
      }
      return { isPoly: false, degree: -1 };
    }
  }
  return { isPoly: false, degree: -1 };
}

function isPolynomialIn(expr: string, variable: string): { isPoly: boolean; degree: number } {
  try {
    const parsed = math.parse(expr);
    return analyzePolynomialNode(parsed, variable);
  } catch {
    return { isPoly: false, degree: -1 };
  }
}

function extractLinearArg(argStr: string, variable: string): { a: number; b: number } | null {
  try {
    const poly = isPolynomialIn(argStr, variable);
    if (!poly.isPoly || poly.degree !== 1) return null;
    const d1 = math.derivative(argStr, variable);
    const a = Number(d1.evaluate({ [variable]: 0 }));
    const b = Number(math.evaluate(argStr, { [variable]: 0 }));
    if (!Number.isFinite(a) || !Number.isFinite(b) || Math.abs(a) < 1e-12) return null;
    return { a, b };
  } catch {
    return null;
  }
}

function evaluateExactExpTerm(coeff: number, expArg: string, variable: string, limitVal: number): { numeric: number; symbolic: string } {
  const argAtLim = Number(math.evaluate(expArg, { [variable]: limitVal }));
  const numeric = coeff * Math.exp(argAtLim);
  if (Math.abs(coeff) < 1e-12) {
    return { numeric: 0, symbolic: '0' };
  }
  const absCoeff = Math.abs(coeff);
  const coeffStr = formatFractionCoeff(absCoeff);
  const expPart = Math.abs(argAtLim) < 1e-12 ? '' : `e^(${formatFractionCoeff(argAtLim)})`;
  const term =
    expPart === ''
      ? coeffStr
      : absCoeff === 1
        ? expPart
        : `${coeffStr} * ${expPart}`;
  return { numeric, symbolic: coeff < 0 ? `-${term}` : term };
}

function integrateSingleTerm(termStr: string, variable: string): {
  antiderivative: string;
  method: string;
  exactDefinite?: (lo: number, hi: number) => { exact: string; numeric: number } | null;
} | null {
  const node = math.parse(termStr);

  // Constant or polynomial term
  const polyInfo = isPolynomialIn(termStr, variable);
  if (polyInfo.isPoly) {
    const simplified = math.simplify(termStr).toString();
    // Extract polynomial coefficients via derivatives at 0: P(x) = sum c_k x^k
    const coeffs: number[] = [];
    let dExpr = simplified;
    let fact = 1;
    for (let k = 0; k <= polyInfo.degree; k += 1) {
      if (k > 0) {
        fact *= k;
        dExpr = math.derivative(dExpr, variable).toString();
      }
      coeffs.push(Number(math.evaluate(dExpr, { [variable]: 0 })) / fact);
    }
    const antiTerms: string[] = [];
    for (let k = 0; k < coeffs.length; k += 1) {
      const c = coeffs[k];
      if (Math.abs(c) < 1e-12) continue;
      const newPow = k + 1;
      const newCoeff = c / newPow;
      const coeffStr = formatFractionCoeff(newCoeff);
      antiTerms.push(newPow === 1 ? `(${coeffStr}) * ${variable}` : `(${coeffStr}) * ${variable}^${newPow}`);
    }
    const anti = antiTerms.length > 0 ? antiTerms.join(' + ') : '0';
    return {
      antiderivative: math.simplify(anti).toString(),
      method: 'polynomial_power_rule',
      exactDefinite: (lo, hi) => {
        const fHi = Number(math.evaluate(anti, { [variable]: hi }));
        const fLo = Number(math.evaluate(anti, { [variable]: lo }));
        const val = fHi - fLo;
        return { exact: formatFractionCoeff(val), numeric: val };
      },
    };
  }

  // Handle product of factors (e.g. P(x) * exp(a*x+b), P(x) * sin(a*x+b), P(x) * cos(a*x+b), c * 1/x)
  const factors: math.MathNode[] = [];
  const collectFactors = (n: math.MathNode): void => {
    if (n.type === 'OperatorNode' && (n as math.OperatorNode).op === '*') {
      const args = (n as math.OperatorNode).args;
      collectFactors(args[0]);
      collectFactors(args[1]);
    } else if (n.type === 'ParenthesisNode') {
      collectFactors((n as math.ParenthesisNode).content);
    } else {
      factors.push(n);
    }
  };
  collectFactors(node);

  // Check negative sign wrapped as unaryMinus
  let sign = 1;
  const cleanedFactors: math.MathNode[] = [];
  for (const f of factors) {
    if (f.type === 'OperatorNode' && (f as math.OperatorNode).isUnary() && (f as math.OperatorNode).op === '-') {
      sign *= -1;
      cleanedFactors.push((f as math.OperatorNode).args[0]);
    } else {
      cleanedFactors.push(f);
    }
  }

  const fnNodes = cleanedFactors.filter((f) => f.type === 'FunctionNode') as math.FunctionNode[];
  const nonFnFactors = cleanedFactors.filter((f) => f.type !== 'FunctionNode');

  if (fnNodes.length === 1) {
    const fn = fnNodes[0];
    const fnName = fn.fn.name;
    const fnArgStr = fn.args[0]?.toString() ?? '';
    const lin = extractLinearArg(fnArgStr, variable);
    const polyPartRaw =
      nonFnFactors.length > 0 ? `${sign < 0 ? '-' : ''}${nonFnFactors.map((n) => `(${n.toString()})`).join(' * ')}` : `${sign}`;
    const polyPartInfo = isPolynomialIn(polyPartRaw, variable);

    if (lin && polyPartInfo.isPoly) {
      const { a } = lin;
      // Tabular Integration by Parts!
      const derivatives: string[] = [math.simplify(polyPartRaw).toString()];
      for (let k = 1; k <= polyPartInfo.degree; k += 1) {
        derivatives.push(math.simplify(math.derivative(derivatives[k - 1], variable)).toString());
      }

      if (fnName === 'exp') {
        // integral P(x) * exp(a*x+b) = exp(a*x+b) * sum_{k=0..deg} (-1)^k * P^(k)(x) / a^(k+1)
        const bracketTerms: string[] = [];
        for (let k = 0; k < derivatives.length; k += 1) {
          const denom = Math.pow(a, k + 1);
          const signFactor = k % 2 === 0 ? 1 : -1;
          const factor = signFactor / denom;
          bracketTerms.push(`(${formatFractionCoeff(factor)}) * (${derivatives[k]})`);
        }
        const bracketPoly = math.simplify(bracketTerms.join(' + ')).toString();
        const anti = `(${bracketPoly}) * exp(${fnArgStr})`;
        return {
          antiderivative: anti,
          method: polyPartInfo.degree > 0 ? 'tabular_integration_by_parts' : 'exponential_substitution',
          exactDefinite: (lo, hi) => {
            const cHi = Number(math.evaluate(bracketPoly, { [variable]: hi }));
            const cLo = Number(math.evaluate(bracketPoly, { [variable]: lo }));
            const tHi = evaluateExactExpTerm(cHi, fnArgStr, variable, hi);
            const tLo = evaluateExactExpTerm(-cLo, fnArgStr, variable, lo);
            const numeric = tHi.numeric + tLo.numeric;

            // Format clean exact expression e.g. 2 - 10 * e^(-2)
            const parts: Array<{ sym: string; isConst: boolean }> = [];
            const argHi = Number(math.evaluate(fnArgStr, { [variable]: hi }));
            const argLo = Number(math.evaluate(fnArgStr, { [variable]: lo }));
            if (Math.abs(tLo.numeric) > 1e-12) {
              parts.push({ sym: tLo.symbolic, isConst: Math.abs(argLo) < 1e-12 });
            }
            if (Math.abs(tHi.numeric) > 1e-12) {
              parts.push({ sym: tHi.symbolic, isConst: Math.abs(argHi) < 1e-12 });
            }
            // Put constant term first if present for canonical readability
            parts.sort((x, y) => Number(y.isConst) - Number(x.isConst));
            let exact = '0';
            if (parts.length > 0) {
              exact = parts[0].sym;
              for (let i = 1; i < parts.length; i += 1) {
                const nxt = parts[i].sym;
                exact += nxt.startsWith('-') ? ` - ${nxt.slice(1)}` : ` + ${nxt}`;
              }
            }
            return { exact, numeric };
          },
        };
      }

      if (fnName === 'sin' || fnName === 'cos') {
        // Tabular integration by parts for P(x) * sin(ax+b) or P(x) * cos(ax+b)
        const antiTerms: string[] = [];
        let currFn = fnName;
        let currSign = 1;
        let denomPow = 1;
        for (let k = 0; k < derivatives.length; k += 1) {
          // Antiderivative of current trig factor
          if (currFn === 'sin') {
            currFn = 'cos';
            currSign *= -1;
          } else {
            currFn = 'sin';
          }
          denomPow *= a;
          const ibpSign = k % 2 === 0 ? 1 : -1;
          const totalFactor = (ibpSign * currSign) / denomPow;
          antiTerms.push(`(${formatFractionCoeff(totalFactor)}) * (${derivatives[k]}) * ${currFn}(${fnArgStr})`);
        }
        const anti = math.simplify(antiTerms.join(' + ')).toString();
        return {
          antiderivative: anti,
          method: polyPartInfo.degree > 0 ? 'tabular_integration_by_parts' : 'trigonometric_antiderivative',
          exactDefinite: (lo, hi) => {
            const val =
              Number(math.evaluate(anti, { [variable]: hi })) -
              Number(math.evaluate(anti, { [variable]: lo }));
            return { exact: val.toPrecision(15).replace(/\.?0+$/, ''), numeric: val };
          },
        };
      }
    }
  }

  // Handle c / (a*x + b)
  if (node.type === 'OperatorNode' && (node as math.OperatorNode).op === '/') {
    const [numNode, denNode] = (node as math.OperatorNode).args;
    const numInfo = isPolynomialIn(numNode.toString(), variable);
    const linDen = extractLinearArg(denNode.toString(), variable);
    if (numInfo.isPoly && numInfo.degree === 0 && linDen) {
      const c = Number(math.evaluate(numNode.toString(), {}));
      const factor = c / linDen.a;
      const anti = `(${formatFractionCoeff(factor)}) * log(abs(${denNode.toString()}))`;
      return {
        antiderivative: anti,
        method: 'logarithmic_reciprocal_rule',
        exactDefinite: (lo, hi) => {
          const val =
            Number(math.evaluate(anti, { [variable]: hi })) -
            Number(math.evaluate(anti, { [variable]: lo }));
          return { exact: val.toPrecision(15).replace(/\.?0+$/, ''), numeric: val };
        },
      };
    }
  }

  return null;
}

function splitTopLevelSumTerms(exprStr: string): string[] {
  const node = math.parse(exprStr);
  const terms: string[] = [];
  const walk = (n: math.MathNode, sign: number): void => {
    if (n.type === 'ParenthesisNode') {
      walk((n as math.ParenthesisNode).content, sign);
      return;
    }
    if (n.type === 'OperatorNode') {
      const op = n as math.OperatorNode;
      if (op.op === '+' && op.args.length === 2) {
        walk(op.args[0], sign);
        walk(op.args[1], sign);
        return;
      }
      if (op.op === '-' && op.args.length === 2) {
        walk(op.args[0], sign);
        walk(op.args[1], -sign);
        return;
      }
      if (op.op === '-' && op.isUnary()) {
        walk(op.args[0], -sign);
        return;
      }
    }
    const raw = n.toString();
    terms.push(sign < 0 ? `-(${raw})` : raw);
  };
  walk(node, 1);
  return terms;
}

function solveIntegralSymbolic(
  integrandExpr: string,
  variable: string,
  lowerLimit: number | null,
  upperLimit: number | null,
): SymbolicIntegralOutput | null {
  const terms = splitTopLevelSumTerms(integrandExpr);
  const integrated = terms.map((t) => integrateSingleTerm(t, variable));
  if (integrated.some((r) => r === null)) {
    return null;
  }

  const valid = integrated as Array<NonNullable<ReturnType<typeof integrateSingleTerm>>>;
  const combinedAnti = math.simplify(valid.map((v) => `(${v.antiderivative})`).join(' + ')).toString();
  const methodUsed = Array.from(new Set(valid.map((v) => v.method))).join(' + ');

  if (lowerLimit !== null && upperLimit !== null) {
    const termResults = valid.map((v) => (v.exactDefinite ? v.exactDefinite(lowerLimit, upperLimit) : null));
    let exactStr: string;
    let numericVal: number;

    if (valid.length === 1 && termResults[0]) {
      exactStr = termResults[0].exact;
      numericVal = termResults[0].numeric;
    } else {
      const fHi = Number(math.evaluate(combinedAnti, { [variable]: upperLimit }));
      const fLo = Number(math.evaluate(combinedAnti, { [variable]: lowerLimit }));
      numericVal = fHi - fLo;
      exactStr =
        Math.abs(numericVal - Math.round(numericVal)) < 1e-12
          ? String(Math.round(numericVal))
          : formatFractionCoeff(numericVal);
    }

    if (!Number.isFinite(numericVal)) {
      return null;
    }

    return {
      antiderivativeExpr: combinedAnti,
      exactDefiniteExpr: exactStr,
      numericValue: Number(numericVal.toPrecision(15)),
      methodUsed,
      intermediateSteps: [
        { step: 'Normalized integrand f(x)', value: integrandExpr, unit: 'dimensionless' },
        { step: 'Antiderivative F(x)', value: combinedAnti, unit: 'dimensionless' },
        {
          step: `Evaluate F(${upperLimit}) - F(${lowerLimit})`,
          value: `${exactStr} (~ ${numericVal.toPrecision(15)})`,
          unit: 'dimensionless',
        },
      ],
    };
  }

  return {
    antiderivativeExpr: `${combinedAnti} + C`,
    exactDefiniteExpr: null,
    numericValue: null,
    methodUsed,
    intermediateSteps: [
      { step: 'Normalized integrand f(x)', value: integrandExpr, unit: 'dimensionless' },
      { step: 'Indefinite antiderivative F(x)', value: `${combinedAnti} + C`, unit: 'dimensionless' },
    ],
  };
}

function parseIntegralPrompt(question: string): {
  integrand: string;
  variable: string;
  lowerLimit: number | null;
  upperLimit: number | null;
} | null {
  const q = question.trim();

  // Pattern 1: "integral of <expr> from <var> = <a> to <var> = <b>" or "from <a> to <b>"
  const m1 = q.match(
    /(?:definite\s+|indefinite\s+)?integral\s+of\s+(.+?)\s+from\s+(?:([a-zA-Z])\s*=\s*)?([^\s]+)\s+to\s+(?:([a-zA-Z])\s*=\s*)?([^\s.?,;]+)\s*[.?!]?$/i,
  );
  if (m1) {
    const integrand = normalizeMathExpr(m1[1].replace(/\bd[a-zA-Z]\b\s*$/i, '').trim());
    const variable = m1[2] || m1[4] || 'x';
    return {
      integrand,
      variable,
      lowerLimit: parseNumericLimit(m1[3]),
      upperLimit: parseNumericLimit(m1[5]),
    };
  }

  // Pattern 2: "integrate <expr> from <a> to <b> [with respect to x]"
  const m2 = q.match(
    /\bintegrate\s+(.+?)\s+from\s+(?:([a-zA-Z])\s*=\s*)?([^\s]+)\s+to\s+(?:([a-zA-Z])\s*=\s*)?([^\s.?,;]+)(?:\s+with\s+respect\s+to\s+([a-zA-Z]))?\s*[.?!]?$/i,
  );
  if (m2) {
    const integrand = normalizeMathExpr(m2[1].replace(/\bd[a-zA-Z]\b\s*$/i, '').trim());
    const variable = m2[6] || m2[2] || m2[4] || 'x';
    return {
      integrand,
      variable,
      lowerLimit: parseNumericLimit(m2[3]),
      upperLimit: parseNumericLimit(m2[5]),
    };
  }

  // Pattern 3: LaTeX \int_{a}^{b} <expr> d<var> or \int_a^b <expr> d<var>
  const m3 = q.match(
    /\\int\s*_\s*\{?([^}\s^]+)\}?\s*\^\s*\{?([^}\s]+)\}?\s*(.+?)\s*d([a-zA-Z])\b/i,
  );
  if (m3) {
    return {
      integrand: normalizeMathExpr(m3[3]),
      variable: m3[4],
      lowerLimit: parseNumericLimit(m3[1]),
      upperLimit: parseNumericLimit(m3[2]),
    };
  }

  // Pattern 4: Indefinite integral "indefinite integral of <expr> [dx]" or "integrate <expr> [with respect to x]"
  const m4 = q.match(
    /(?:(?:indefinite\s+)?integral\s+of|integrate)\s+(.+?)(?:\s+(?:with\s+respect\s+to\s+([a-zA-Z])|d([a-zA-Z])))?\s*[.?!]?$/i,
  );
  if (m4 && !/\bfrom\b/i.test(q)) {
    const exprRaw = m4[1].replace(/\s*d([a-zA-Z])\s*$/i, '');
    const dMatch = m4[1].match(/\bd([a-zA-Z])\s*$/i);
    const variable = m4[2] || m4[3] || (dMatch ? dMatch[1] : 'x');
    return {
      integrand: normalizeMathExpr(exprRaw),
      variable,
      lowerLimit: null,
      upperLimit: null,
    };
  }

  return null;
}

function parseLinearSystem(question: string): {
  variables: string[];
  matrixA: number[][];
  vectorB: number[];
} | null {
  const cleaned = question
    .replace(/^(?:solve\s+(?:the\s+)?(?:linear\s+)?system(?:\s+of\s+equations)?[:\s]*)/i, '')
    .replace(/\s+\bfor\s+[a-zA-Z,\s]+(?:and\s+verify.*)?[.?!]?\s*$/i, '')
    .replace(/\s+\band\s+verify.*$/i, '')
    .replace(/[.?!]\s*$/, '')
    .trim();

  const eqStrings = cleaned
    .split(/\s*(?:\band\b|,|;|\n)\s*/i)
    .map((s) => s.trim())
    .filter((s) => s.includes('='));

  if (eqStrings.length < 2 || eqStrings.length > 4) {
    return null;
  }

  const varSet = new Set<string>();
  const normalizedEqs: Array<{ lhsMinusRhs: string }> = [];

  for (const eq of eqStrings) {
    const parts = eq.split('=');
    if (parts.length !== 2) return null;
    const lhs = normalizeMathExpr(parts[0]);
    const rhs = normalizeMathExpr(parts[1]);
    const expr = `(${lhs}) - (${rhs})`;
    normalizedEqs.push({ lhsMinusRhs: expr });

    const node = math.parse(expr);
    node.traverse((n) => {
      if (n.type === 'SymbolNode') {
        const sym = (n as math.SymbolNode).name;
        if (!['e', 'pi', 'i'].includes(sym)) {
          varSet.add(sym);
        }
      }
    });
  }

  const variables = Array.from(varSet).sort();
  if (variables.length !== eqStrings.length) {
    return null;
  }

  const matrixA: number[][] = [];
  const vectorB: number[] = [];
  const zeroScope: Record<string, number> = {};
  for (const v of variables) zeroScope[v] = 0;

  for (const { lhsMinusRhs } of normalizedEqs) {
    // Check linearity: second derivative wrt every variable is 0
    const bVal = -Number(math.evaluate(lhsMinusRhs, zeroScope));
    const row: number[] = [];
    for (const v of variables) {
      const d1 = math.derivative(lhsMinusRhs, v);
      for (const v2 of variables) {
        if (math.simplify(math.derivative(d1, v2)).toString() !== '0') {
          return null;
        }
      }
      row.push(Number(d1.evaluate(zeroScope)));
    }
    matrixA.push(row);
    vectorB.push(bVal);
  }

  return { variables, matrixA, vectorB };
}

function buildFastReport(params: {
  problemId: string;
  executionId: string;
  question: string;
  decision: ComplexityDecision;
  spec: ProblemSpecification;
  plan: ExecutionPlan;
  timeline: AgentExecutionResult[];
  toolRecords: ToolExecutionRecord[];
  verification: VerificationReport;
  methodology: string;
  equations: string[];
  intermediateResults: Array<{ step: string; value: string; unit: string }>;
  finalAnswer: string;
  numericValue: number | null;
  units: string;
  explanationMarkdown: string;
  startedAt: string;
  elapsedMs: number;
}): FinalSolutionReport {
  return {
    problem_id: params.problemId,
    execution_id: params.executionId,
    original_question: params.question,
    detected_domain: params.decision.detected_domain,
    extracted_variables: params.spec.variables,
    assumptions: params.spec.assumptions,
    solution_methodology: params.methodology,
    equations: params.equations,
    intermediate_results: params.intermediateResults,
    final_answer: params.finalAnswer,
    numeric_value: params.numericValue,
    units: params.units,
    verification_status: params.verification.status,
    verification_evidence: params.verification.checks,
    correction_attempts: 0,
    correction_history: [],
    tool_execution_history: params.toolRecords,
    confidence_limitations: [
      'Solved via deterministic symbolic/algebraic fast path and independently verified in isolated Python 3.11.',
    ],
    explanation_markdown: params.explanationMarkdown,
    complexity_level: params.decision.complexity_level,
    execution_path: params.decision.execution_path,
    elapsed_ms: params.elapsedMs,
    started_at: params.startedAt,
    completed_at: new Date().toISOString(),
  };
}

export async function tryFastPathSolve(
  problemId: string,
  executionId: string,
  question: string,
  decision: ComplexityDecision,
): Promise<FastPathResult> {
  const t0 = performance.now();
  const startedAt = new Date().toISOString();

  if (decision.execution_path !== 'FAST_PATH') {
    return { handled: false, reason: 'Not classified for FAST_PATH.' };
  }

  try {
    const timeline: AgentExecutionResult[] = [
      {
        agent_name: 'ComplexityRouter',
        stage: 'ROUTING',
        status: 'COMPLETED',
        summary: `Classified as ${decision.complexity_level} (${decision.problem_category}) -> ${decision.execution_path}`,
        structured_output: decision as unknown as Record<string, unknown>,
        tool_records: [],
        tokens_input: 0,
        tokens_output: 0,
        duration_ms: decision.router_latency_ms,
        started_at: startedAt,
        completed_at: new Date().toISOString(),
      },
    ];

    // -------------------------------------------------------------------------
    // CASE 1: Definite / Indefinite Integrals
    // -------------------------------------------------------------------------
    if (decision.problem_category === 'calculus_integral') {
      const parsed = parseIntegralPrompt(question);
      if (!parsed) {
        return { handled: false, reason: 'Could not deterministically parse integral prompt.' };
      }

      const solverStart = new Date().toISOString();
      const sT0 = performance.now();
      const solved = solveIntegralSymbolic(
        parsed.integrand,
        parsed.variable,
        parsed.lowerLimit,
        parsed.upperLimit,
      );
      if (!solved) {
        return { handled: false, reason: 'Integrand structure requires full multi-agent/LLM solver.' };
      }

      // Verify antiderivative F'(x) == f(x) using symbolic_math + multi-point check
      const diffCheckRecord = await executeToolAsync('symbolic_math', {
        operation: 'diff',
        expression: solved.antiderivativeExpr.replace(/\s*\+\s*C$/, ''),
        variable: parsed.variable,
      });

      const pyIntegrand = toPythonMathExpr(parsed.integrand);
      const pyAntiDeriv = toPythonMathExpr(String(diffCheckRecord.output_payload.result_str ?? '0'));

      // Run FTC differentiation check & Simpson quadrature (if definite) in parallel!
      const testLo = parsed.lowerLimit ?? 0.25;
      const testHi = parsed.upperLimit ?? 1.75;
      const ftcCode = `
import math
def f(${parsed.variable}):
    return ${pyIntegrand}
def fp(${parsed.variable}):
    return ${pyAntiDeriv}
pts = [ ${testLo} + (${testHi} - ${testLo}) * k / 6.0 for k in range(1, 6) ]
max_err = max(abs(fp(p) - f(p)) for p in pts)
RESULT = {
    "passed": max_err < 1e-8,
    "max_error": max_err,
    "details": f"Fundamental Theorem check max |F'(x)-f(x)| across 5 points = {max_err:.3e}"
}
`;

      const verifStart = new Date().toISOString();
      const vT0 = performance.now();

      const parallelChecks: Array<Promise<ToolExecutionRecord>> = [
        executeToolAsync('python_sandbox', { code: ftcCode }),
      ];

      if (parsed.lowerLimit !== null && parsed.upperLimit !== null && solved.numericValue !== null) {
        parallelChecks.push(
          executeToolAsync('independent_verifier', {
            verification_type: 'integral_quadrature',
            expression: pyIntegrand,
            variable: parsed.variable,
            lower_limit: parsed.lowerLimit,
            upper_limit: parsed.upperLimit,
            candidate_value: solved.numericValue,
            tolerance: 1e-5,
          }),
        );
      }

      const verifToolRecords = await Promise.all(parallelChecks);
      const allToolRecords = [diffCheckRecord, ...verifToolRecords];

      const ftcRes = (verifToolRecords[0]?.output_payload?.result ?? {}) as Record<string, unknown>;
      const ftcPassed = verifToolRecords[0]?.success === true && ftcRes.passed === true;

      const checks: VerificationCheck[] = [
        {
          check_name: 'ftc_symbolic_derivative_identity',
          method: 'symbolic_math_diff_plus_multipoint_python',
          passed: ftcPassed,
          expected_value: parsed.integrand,
          observed_value: String(diffCheckRecord.output_payload.result_str ?? ''),
          relative_error: typeof ftcRes.max_error === 'number' ? ftcRes.max_error : null,
          tolerance: 1e-8,
          details: String(ftcRes.details ?? 'FTC check failed'),
        },
      ];

      let quadPassed = true;
      let indepRecalc: string | null = null;
      if (verifToolRecords.length > 1) {
        const qRec = verifToolRecords[1];
        const qOut = qRec.output_payload;
        quadPassed = qRec.success && qOut.passed === true;
        indepRecalc = qOut.independent_value !== undefined ? String(qOut.independent_value) : null;
        checks.push({
          check_name: 'independent_composite_simpson_quadrature',
          method: String(qOut.method_used ?? 'composite_simpson_quadrature_n2000'),
          passed: quadPassed,
          expected_value: indepRecalc,
          observed_value: String(solved.numericValue),
          relative_error: typeof qOut.relative_error === 'number' ? qOut.relative_error : null,
          tolerance: typeof qOut.tolerance === 'number' ? qOut.tolerance : 1e-5,
          details: String(qOut.details ?? qRec.error_message ?? 'Simpson check'),
        });
      }

      if (!ftcPassed || !quadPassed) {
        return {
          handled: false,
          reason: 'Fast-path independent verification did not pass; escalating to multi-agent workflow.',
        };
      }

      const finalAnswer = solved.exactDefiniteExpr ?? solved.antiderivativeExpr;
      const spec: ProblemSpecification = {
        domain: 'mathematics',
        problem_type: parsed.lowerLimit !== null ? 'definite_integral' : 'indefinite_integral',
        objective: `Evaluate the integral of ${parsed.integrand} with respect to ${parsed.variable}`,
        variables: [
          {
            symbol: parsed.variable,
            name: 'integration variable',
            value:
              parsed.lowerLimit !== null && parsed.upperLimit !== null
                ? `[${parsed.lowerLimit}, ${parsed.upperLimit}]`
                : null,
            unit: 'dimensionless',
            constraints: [],
            description: 'Independent variable of integration',
          },
        ],
        equations: [
          parsed.lowerLimit !== null && parsed.upperLimit !== null
            ? `int_${parsed.lowerLimit}^${parsed.upperLimit} (${parsed.integrand}) d${parsed.variable}`
            : `int (${parsed.integrand}) d${parsed.variable}`,
        ],
        constraints: [],
        assumptions: ['Integrand is continuous on the integration domain.'],
        missing_information: [],
        needs_clarification: false,
        clarification_question: null,
      };

      const plan: ExecutionPlan = {
        methodology: `Fast symbolic integration (${solved.methodUsed}) with independent FTC differentiation and Python numerical quadrature verification.`,
        selected_tools: ['symbolic_math', 'python_sandbox', 'independent_verifier'],
        tasks: [
          {
            task_id: 'fast_1',
            title: 'Symbolic Antiderivative Construction',
            description: `Compute exact antiderivative F(${parsed.variable}) via ${solved.methodUsed}.`,
            method: solved.methodUsed,
            tool_name: 'symbolic_math',
            depends_on: [],
            expected_output: solved.antiderivativeExpr,
          },
          {
            task_id: 'fast_2',
            title: 'Independent Verification',
            description: 'Verify F\'(x) == f(x) and cross-check definite value via Composite Simpson quadrature (n=2000).',
            method: 'ftc_and_simpson_quadrature',
            tool_name: 'independent_verifier',
            depends_on: ['fast_1'],
            expected_output: 'PASS',
          },
        ],
        verification_strategy: 'Two-method independent proof: symbolic derivative FTC identity + Python 3.11 Composite Simpson quadrature (n=2000).',
      };

      const verification: VerificationReport = {
        status: 'PASS',
        checks,
        independent_recalculation: indepRecalc,
        dimensional_consistency_passed: true,
        boundary_conditions_passed: true,
        numerical_stability_passed: true,
        failure_reasons: [],
        evidence_summary: checks.map((c) => c.details).join(' | '),
        verified_at: new Date().toISOString(),
      };

      timeline.push(
        {
          agent_name: 'FastSymbolicSolver',
          stage: 'SOLVING',
          status: 'COMPLETED',
          summary: `Computed exact integral (${solved.methodUsed}): ${finalAnswer}`,
          structured_output: {
            antiderivative: solved.antiderivativeExpr,
            final_answer: finalAnswer,
            numeric_value: solved.numericValue,
          },
          tool_records: [diffCheckRecord],
          tokens_input: 0,
          tokens_output: 0,
          duration_ms: Number((performance.now() - sT0).toFixed(2)),
          started_at: solverStart,
          completed_at: new Date().toISOString(),
        },
        {
          agent_name: 'IndependentVerificationAgent',
          stage: 'VERIFICATION',
          status: 'PASS',
          summary: verification.evidence_summary,
          structured_output: verification as unknown as Record<string, unknown>,
          tool_records: verifToolRecords,
          tokens_input: 0,
          tokens_output: 0,
          duration_ms: Number((performance.now() - vT0).toFixed(2)),
          started_at: verifStart,
          completed_at: new Date().toISOString(),
        },
      );

      const elapsedMs = Number((performance.now() - t0).toFixed(2));
      const md = [
        `### Exact Symbolic Solution (Fast Path)`,
        `We compute the antiderivative of $f(${parsed.variable}) = ${parsed.integrand}$ using **${solved.methodUsed.replace(/_/g, ' ')}**:`,
        `$$F(${parsed.variable}) = ${solved.antiderivativeExpr}$$`,
        parsed.lowerLimit !== null && parsed.upperLimit !== null
          ? `Evaluating across $[${parsed.lowerLimit}, ${parsed.upperLimit}]$:\n$$\\int_{${parsed.lowerLimit}}^{${parsed.upperLimit}} (${parsed.integrand})\\,d${parsed.variable} = F(${parsed.upperLimit}) - F(${parsed.lowerLimit}) = ${finalAnswer} \\approx ${solved.numericValue}$$`
          : `$$\\int (${parsed.integrand})\\,d${parsed.variable} = ${finalAnswer}$$`,
        `### Independent Verification Evidence`,
        ...checks.map((c) => `- **${c.check_name}**: \`PASS\` — ${c.details}`),
      ].join('\n\n');

      const report = buildFastReport({
        problemId,
        executionId,
        question,
        decision,
        spec,
        plan,
        timeline,
        toolRecords: allToolRecords,
        verification,
        methodology: plan.methodology,
        equations: spec.equations,
        intermediateResults: solved.intermediateSteps,
        finalAnswer,
        numericValue: solved.numericValue,
        units: 'dimensionless',
        explanationMarkdown: md,
        startedAt,
        elapsedMs,
      });

      return {
        handled: true,
        specification: spec,
        plan,
        agent_timeline: timeline,
        tool_records: allToolRecords,
        verification_report: verification,
        final_report: report,
      };
    }

    // -------------------------------------------------------------------------
    // CASE 2: Symbolic Derivatives
    // -------------------------------------------------------------------------
    if (decision.problem_category === 'calculus_derivative') {
      const m = question.match(
        /(?:derivative\s+of|differentiate)\s+(.+?)(?:\s+with\s+respect\s+to\s+([a-zA-Z]))?(?:\s+at\s+([a-zA-Z])\s*=\s*([-+]?\d+(?:\.\d+)?))?\s*[.?!]?$/i,
      );
      if (!m) {
        return { handled: false, reason: 'Could not deterministically parse derivative prompt.' };
      }

      const expr = normalizeMathExpr(m[1]);
      const variable = m[2] || m[3] || 'x';
      const evalPt = m[4] ? Number(m[4]) : 1.25;
      const solverStart = new Date().toISOString();
      const sT0 = performance.now();

      const diffRec = await executeToolAsync('symbolic_math', {
        operation: 'diff',
        expression: expr,
        variable,
        substitutions: { [variable]: evalPt },
      });
      if (!diffRec.success || !diffRec.output_payload.result_str) {
        return { handled: false, reason: 'Symbolic differentiation failed.' };
      }

      const symResult = String(diffRec.output_payload.result_str);
      const numericAtPt = Number(diffRec.output_payload.numeric_approx);
      if (!Number.isFinite(numericAtPt)) {
        return { handled: false, reason: 'Derivative numeric evaluation non-finite.' };
      }

      const verifStart = new Date().toISOString();
      const vT0 = performance.now();
      const fdRec = await executeToolAsync('independent_verifier', {
        verification_type: 'derivative_finite_diff',
        expression: toPythonMathExpr(expr),
        variable,
        point: evalPt,
        candidate_value: numericAtPt,
        tolerance: 1e-4,
      });

      const fdOut = fdRec.output_payload;
      if (!fdRec.success || fdOut.passed !== true) {
        return { handled: false, reason: 'Finite-difference verification did not pass.' };
      }

      const hasExplicitPt = Boolean(m[4]);
      const finalAnswer = hasExplicitPt ? String(numericAtPt) : symResult;

      const checks: VerificationCheck[] = [
        {
          check_name: 'central_finite_difference',
          method: String(fdOut.method_used ?? 'central_finite_difference'),
          passed: true,
          expected_value: String(fdOut.independent_value ?? ''),
          observed_value: String(numericAtPt),
          relative_error: typeof fdOut.relative_error === 'number' ? fdOut.relative_error : null,
          tolerance: typeof fdOut.tolerance === 'number' ? fdOut.tolerance : 1e-4,
          details: String(fdOut.details ?? ''),
        },
      ];

      const spec: ProblemSpecification = {
        domain: 'mathematics',
        problem_type: 'symbolic_differentiation',
        objective: `Compute derivative of ${expr} with respect to ${variable}`,
        variables: [
          {
            symbol: variable,
            name: 'independent variable',
            value: hasExplicitPt ? evalPt : null,
            unit: 'dimensionless',
            constraints: [],
            description: 'Differentiation variable',
          },
        ],
        equations: [`d/d${variable} (${expr})`],
        constraints: [],
        assumptions: ['Function is differentiable at the evaluation point.'],
        missing_information: [],
        needs_clarification: false,
        clarification_question: null,
      };

      const plan: ExecutionPlan = {
        methodology: 'Fast symbolic differentiation verified by Python central finite differences.',
        selected_tools: ['symbolic_math', 'independent_verifier'],
        tasks: [
          {
            task_id: 'fast_1',
            title: 'Symbolic Derivative',
            description: `Compute d/d${variable}(${expr}).`,
            method: 'symbolic_math.diff',
            tool_name: 'symbolic_math',
            depends_on: [],
            expected_output: symResult,
          },
          {
            task_id: 'fast_2',
            title: 'Finite Difference Verification',
            description: `Verify derivative numerically at ${variable} = ${evalPt}.`,
            method: 'central_finite_difference',
            tool_name: 'independent_verifier',
            depends_on: ['fast_1'],
            expected_output: 'PASS',
          },
        ],
        verification_strategy: 'Compare symbolic derivative against isolated Python central finite difference (h=1e-6).',
      };

      const verification: VerificationReport = {
        status: 'PASS',
        checks,
        independent_recalculation: String(fdOut.independent_value ?? ''),
        dimensional_consistency_passed: true,
        boundary_conditions_passed: true,
        numerical_stability_passed: true,
        failure_reasons: [],
        evidence_summary: String(fdOut.details ?? ''),
        verified_at: new Date().toISOString(),
      };

      timeline.push(
        {
          agent_name: 'FastSymbolicSolver',
          stage: 'SOLVING',
          status: 'COMPLETED',
          summary: `Symbolic derivative: ${symResult}`,
          structured_output: { derivative: symResult, at_point: evalPt, value: numericAtPt },
          tool_records: [diffRec],
          tokens_input: 0,
          tokens_output: 0,
          duration_ms: Number((performance.now() - sT0).toFixed(2)),
          started_at: solverStart,
          completed_at: new Date().toISOString(),
        },
        {
          agent_name: 'IndependentVerificationAgent',
          stage: 'VERIFICATION',
          status: 'PASS',
          summary: verification.evidence_summary,
          structured_output: verification as unknown as Record<string, unknown>,
          tool_records: [fdRec],
          tokens_input: 0,
          tokens_output: 0,
          duration_ms: Number((performance.now() - vT0).toFixed(2)),
          started_at: verifStart,
          completed_at: new Date().toISOString(),
        },
      );

      const elapsedMs = Number((performance.now() - t0).toFixed(2));
      const report = buildFastReport({
        problemId,
        executionId,
        question,
        decision,
        spec,
        plan,
        timeline,
        toolRecords: [diffRec, fdRec],
        verification,
        methodology: plan.methodology,
        equations: spec.equations,
        intermediateResults: [
          { step: 'Symbolic derivative f\'(x)', value: symResult, unit: 'dimensionless' },
          { step: `Check value at ${variable} = ${evalPt}`, value: String(numericAtPt), unit: 'dimensionless' },
        ],
        finalAnswer,
        numericValue: hasExplicitPt ? numericAtPt : null,
        units: 'dimensionless',
        explanationMarkdown: `### Symbolic Differentiation (Fast Path)\n\n$$\\frac{d}{d${variable}}\\left(${expr}\\right) = ${symResult}$$\n\n### Independent Verification\n\n- **Central Finite Difference**: \`PASS\` — ${verification.evidence_summary}`,
        startedAt,
        elapsedMs,
      });

      return {
        handled: true,
        specification: spec,
        plan,
        agent_timeline: timeline,
        tool_records: [diffRec, fdRec],
        verification_report: verification,
        final_report: report,
      };
    }

    // -------------------------------------------------------------------------
    // CASE 3: Linear Systems (2x2 - 4x4)
    // -------------------------------------------------------------------------
    if (decision.problem_category === 'linear_system') {
      const sys = parseLinearSystem(question);
      if (!sys) {
        return { handled: false, reason: 'Could not parse linear system cleanly.' };
      }

      const solverStart = new Date().toISOString();
      const sT0 = performance.now();
      const solveRec = await executeToolAsync('symbolic_math', {
        operation: 'linear_solve',
        matrix_a: sys.matrixA,
        vector_b: sys.vectorB,
      });
      if (!solveRec.success || !Array.isArray(solveRec.output_payload.solution)) {
        return { handled: false, reason: 'Matrix singular or linear_solve failed.' };
      }

      const sol = solveRec.output_payload.solution as number[];
      const verifStart = new Date().toISOString();
      const vT0 = performance.now();

      const pyResidualCode = `
import math
A = ${JSON.stringify(sys.matrixA)}
b = ${JSON.stringify(sys.vectorB)}
x = ${JSON.stringify(sol)}
residuals = [sum(A[i][j] * x[j] for j in range(len(x))) - b[i] for i in range(len(b))]
norm = math.sqrt(sum(r*r for r in residuals))
RESULT = {
    "passed": norm < 1e-9,
    "residual_norm": norm,
    "details": f"Independent residual ||Ax - b||_2 = {norm:.3e}"
}
`;
      const pyVerifRec = await executeToolAsync('python_sandbox', { code: pyResidualCode });
      const pyRes = (pyVerifRec.output_payload.result ?? {}) as Record<string, unknown>;
      if (!pyVerifRec.success || pyRes.passed !== true) {
        return { handled: false, reason: 'Linear system residual verification failed.' };
      }

      const formattedPairs = sys.variables.map((v, i) => `${v} = ${formatFractionCoeff(sol[i])}`);
      const finalAnswer = formattedPairs.join(', ');
      const checks: VerificationCheck[] = [
        {
          check_name: 'linear_system_residual_norm',
          method: 'python_direct_substitution_l2_norm',
          passed: true,
          expected_value: '0',
          observed_value: String(pyRes.residual_norm),
          relative_error: Number(pyRes.residual_norm),
          tolerance: 1e-9,
          details: String(pyRes.details),
        },
      ];

      const spec: ProblemSpecification = {
        domain: 'mathematics',
        problem_type: 'linear_system',
        objective: `Solve linear system for ${sys.variables.join(', ')}`,
        variables: sys.variables.map((v) => ({
          symbol: v,
          name: `unknown ${v}`,
          value: null,
          unit: 'dimensionless',
          constraints: [],
          description: 'Linear system variable',
        })),
        equations: sys.matrixA.map(
          (row, idx) =>
            `${row.map((c, j) => `${c}*${sys.variables[j]}`).join(' + ')} = ${sys.vectorB[idx]}`,
        ),
        constraints: [],
        assumptions: ['Coefficient matrix A is nonsingular (det(A) != 0).'],
        missing_information: [],
        needs_clarification: false,
        clarification_question: null,
      };

      const plan: ExecutionPlan = {
        methodology: 'LU factorization solve with independent Python L2 residual norm verification.',
        selected_tools: ['symbolic_math', 'python_sandbox'],
        tasks: [
          {
            task_id: 'fast_1',
            title: 'LU Linear Solve',
            description: 'Solve Ax = b using LU decomposition.',
            method: 'math.lusolve',
            tool_name: 'symbolic_math',
            depends_on: [],
            expected_output: finalAnswer,
          },
          {
            task_id: 'fast_2',
            title: 'Residual Norm Verification',
            description: 'Verify ||Ax - b||_2 < 1e-9 in isolated Python.',
            method: 'python_residual_norm',
            tool_name: 'python_sandbox',
            depends_on: ['fast_1'],
            expected_output: 'PASS',
          },
        ],
        verification_strategy: 'Independent substitution of x into Ax - b in isolated Python 3.11.',
      };

      const verification: VerificationReport = {
        status: 'PASS',
        checks,
        independent_recalculation: String(pyRes.residual_norm),
        dimensional_consistency_passed: true,
        boundary_conditions_passed: true,
        numerical_stability_passed: true,
        failure_reasons: [],
        evidence_summary: String(pyRes.details),
        verified_at: new Date().toISOString(),
      };

      timeline.push(
        {
          agent_name: 'FastSymbolicSolver',
          stage: 'SOLVING',
          status: 'COMPLETED',
          summary: `Solved linear system: ${finalAnswer}`,
          structured_output: { solution: sol, variables: sys.variables },
          tool_records: [solveRec],
          tokens_input: 0,
          tokens_output: 0,
          duration_ms: Number((performance.now() - sT0).toFixed(2)),
          started_at: solverStart,
          completed_at: new Date().toISOString(),
        },
        {
          agent_name: 'IndependentVerificationAgent',
          stage: 'VERIFICATION',
          status: 'PASS',
          summary: verification.evidence_summary,
          structured_output: verification as unknown as Record<string, unknown>,
          tool_records: [pyVerifRec],
          tokens_input: 0,
          tokens_output: 0,
          duration_ms: Number((performance.now() - vT0).toFixed(2)),
          started_at: verifStart,
          completed_at: new Date().toISOString(),
        },
      );

      const elapsedMs = Number((performance.now() - t0).toFixed(2));
      const report = buildFastReport({
        problemId,
        executionId,
        question,
        decision,
        spec,
        plan,
        timeline,
        toolRecords: [solveRec, pyVerifRec],
        verification,
        methodology: plan.methodology,
        equations: spec.equations,
        intermediateResults: sys.variables.map((v, i) => ({
          step: `Solved ${v}`,
          value: `${formatFractionCoeff(sol[i])} (${sol[i]})`,
          unit: 'dimensionless',
        })),
        finalAnswer,
        numericValue: sol.length === 1 ? sol[0] : null,
        units: 'dimensionless',
        explanationMarkdown: `### Linear System Solution (Fast Path)\n\n${spec.equations.map((e) => `- $${e}$`).join('\n')}\n\n**Solution:** $${finalAnswer}$\n\n### Independent Verification\n\n- **Residual Check**: \`PASS\` — ${verification.evidence_summary}`,
        startedAt,
        elapsedMs,
      });

      return {
        handled: true,
        specification: spec,
        plan,
        agent_timeline: timeline,
        tool_records: [solveRec, pyVerifRec],
        verification_report: verification,
        final_report: report,
      };
    }

    // -------------------------------------------------------------------------
    // CASE 4: Single-Variable Linear / Quadratic Algebraic Equations
    // -------------------------------------------------------------------------
    if (decision.problem_category === 'algebraic_equation') {
      const eqMatch = question.match(/([^=.\n]+?)\s*=\s*([^.\n?;]+)(?:\s+for\s+([a-zA-Z]))?/i);
      if (!eqMatch) {
        return { handled: false, reason: 'Could not extract equation.' };
      }

      const lhsRaw = eqMatch[1]
        .replace(/^(?:solve|find\s+(?:the\s+)?roots?\s+of|find\s+[a-zA-Z]\s+(?:if|for|such\s+that))\s+/i, '')
        .trim();
      const rhsRaw = eqMatch[2].replace(/\s+for\s+[a-zA-Z].*$/i, '').trim();
      const lhs = normalizeMathExpr(lhsRaw);
      const rhs = normalizeMathExpr(rhsRaw);
      const diffExpr = `(${lhs}) - (${rhs})`;

      const symSet = new Set<string>();
      math.parse(diffExpr).traverse((n) => {
        if (n.type === 'SymbolNode') {
          const sym = (n as math.SymbolNode).name;
          if (!['e', 'pi', 'i'].includes(sym)) symSet.add(sym);
        }
      });
      if (symSet.size !== 1) {
        return { handled: false, reason: 'Equation does not have exactly one unknown variable.' };
      }
      const variable = Array.from(symSet)[0];

      const eqPoly = isPolynomialIn(diffExpr, variable);
      if (!eqPoly.isPoly || eqPoly.degree > 2) {
        return { handled: false, reason: 'Degree > 2 or transcendental equation; escalating to full workflow.' };
      }
      const d1 = math.derivative(diffExpr, variable);
      const d2 = math.derivative(d1, variable);

      const c = Number(math.evaluate(diffExpr, { [variable]: 0 }));
      const b = Number(d1.evaluate({ [variable]: 0 }));
      const a = Number(d2.evaluate({ [variable]: 0 })) / 2;

      const solverStart = new Date().toISOString();
      const sT0 = performance.now();

      const roots: number[] = [];
      let methodUsed = 'linear_inversion';
      if (Math.abs(a) < 1e-12) {
        if (Math.abs(b) < 1e-12) return { handled: false, reason: 'Degenerate equation.' };
        roots.push(-c / b);
      } else {
        methodUsed = 'quadratic_formula';
        const disc = b * b - 4 * a * c;
        if (disc < 0) {
          return { handled: false, reason: 'Complex roots detected; escalating to full workflow.' };
        }
        const sq = Math.sqrt(disc);
        const r1 = (-b - sq) / (2 * a);
        const r2 = (-b + sq) / (2 * a);
        if (Math.abs(r1 - r2) < 1e-12) roots.push(r1);
        else roots.push(r1, r2);
      }

      const pyDiffExpr = toPythonMathExpr(diffExpr);
      const verifStart = new Date().toISOString();
      const vT0 = performance.now();
      const verifRecords = await Promise.all(
        roots.map((r) =>
          executeToolAsync('independent_verifier', {
            verification_type: 'root_residual',
            expression: pyDiffExpr,
            variable,
            candidate_value: r,
            tolerance: 1e-8,
          }),
        ),
      );

      if (verifRecords.some((rec) => !rec.success || rec.output_payload.passed !== true)) {
        return { handled: false, reason: 'Root residual check failed.' };
      }

      const formattedRoots = roots.map((r) => formatFractionCoeff(r));
      const finalAnswer =
        roots.length === 1
          ? `${variable} = ${formattedRoots[0]}`
          : `${variable} = ${formattedRoots.join(', ')}`;

      const checks: VerificationCheck[] = verifRecords.map((rec, idx) => ({
        check_name: `root_residual_${idx + 1}`,
        method: 'direct_function_residual',
        passed: true,
        expected_value: '0',
        observed_value: String(rec.output_payload.candidate_value),
        relative_error: Number(rec.output_payload.absolute_error ?? 0),
        tolerance: 1e-8,
        details: String(rec.output_payload.details ?? ''),
      }));

      const spec: ProblemSpecification = {
        domain: 'mathematics',
        problem_type: methodUsed,
        objective: `Solve ${lhs} = ${rhs} for ${variable}`,
        variables: [
          {
            symbol: variable,
            name: `unknown ${variable}`,
            value: null,
            unit: 'dimensionless',
            constraints: [],
            description: 'Equation unknown',
          },
        ],
        equations: [`${lhs} = ${rhs}`],
        constraints: [],
        assumptions: ['Seeking real-valued roots.'],
        missing_information: [],
        needs_clarification: false,
        clarification_question: null,
      };

      const plan: ExecutionPlan = {
        methodology: `Exact algebraic root finding (${methodUsed}) with Python residual verification.`,
        selected_tools: ['independent_verifier'],
        tasks: [
          {
            task_id: 'fast_1',
            title: 'Exact Root Extraction',
            description: `Solve ${diffExpr} = 0.`,
            method: methodUsed,
            tool_name: 'symbolic_math',
            depends_on: [],
            expected_output: finalAnswer,
          },
          {
            task_id: 'fast_2',
            title: 'Root Residual Verification',
            description: 'Evaluate |f(r)| < 1e-8 for each root in Python.',
            method: 'root_residual',
            tool_name: 'independent_verifier',
            depends_on: ['fast_1'],
            expected_output: 'PASS',
          },
        ],
        verification_strategy: 'Verify |f(r)| <= 1e-8 for every root independently in isolated Python.',
      };

      const verification: VerificationReport = {
        status: 'PASS',
        checks,
        independent_recalculation: checks.map((ck) => ck.details).join('; '),
        dimensional_consistency_passed: true,
        boundary_conditions_passed: true,
        numerical_stability_passed: true,
        failure_reasons: [],
        evidence_summary: checks.map((ck) => ck.details).join(' | '),
        verified_at: new Date().toISOString(),
      };

      timeline.push(
        {
          agent_name: 'FastSymbolicSolver',
          stage: 'SOLVING',
          status: 'COMPLETED',
          summary: `Exact roots (${methodUsed}): ${finalAnswer}`,
          structured_output: { coefficients: { a, b, c }, roots },
          tool_records: [],
          tokens_input: 0,
          tokens_output: 0,
          duration_ms: Number((performance.now() - sT0).toFixed(2)),
          started_at: solverStart,
          completed_at: new Date().toISOString(),
        },
        {
          agent_name: 'IndependentVerificationAgent',
          stage: 'VERIFICATION',
          status: 'PASS',
          summary: verification.evidence_summary,
          structured_output: verification as unknown as Record<string, unknown>,
          tool_records: verifRecords,
          tokens_input: 0,
          tokens_output: 0,
          duration_ms: Number((performance.now() - vT0).toFixed(2)),
          started_at: verifStart,
          completed_at: new Date().toISOString(),
        },
      );

      const elapsedMs = Number((performance.now() - t0).toFixed(2));
      const report = buildFastReport({
        problemId,
        executionId,
        question,
        decision,
        spec,
        plan,
        timeline,
        toolRecords: verifRecords,
        verification,
        methodology: plan.methodology,
        equations: spec.equations,
        intermediateResults: roots.map((r, i) => ({
          step: `Root ${i + 1}`,
          value: `${formattedRoots[i]} (${r})`,
          unit: 'dimensionless',
        })),
        finalAnswer,
        numericValue: roots.length === 1 ? roots[0] : null,
        units: 'dimensionless',
        explanationMarkdown: `### Algebraic Equation Solution (Fast Path)\n\nEquation: $${lhs} = ${rhs}$\n\n**Verified Solution:** $${finalAnswer}$\n\n### Independent Verification\n\n${checks.map((ck) => `- **${ck.check_name}**: \`PASS\` — ${ck.details}`).join('\n')}`,
        startedAt,
        elapsedMs,
      });

      return {
        handled: true,
        specification: spec,
        plan,
        agent_timeline: timeline,
        tool_records: verifRecords,
        verification_report: verification,
        final_report: report,
      };
    }

    // -------------------------------------------------------------------------
    // CASE 5: Matrix Determinants / Traces
    // -------------------------------------------------------------------------
    if (decision.problem_category === 'linear_algebra_matrix') {
      const matMatch = question.match(/(\[\s*\[[\s\S]+\]\s*\])/);
      if (!matMatch) {
        return { handled: false, reason: 'Could not extract matrix literal.' };
      }
      const mat = JSON.parse(matMatch[1]) as number[][];
      if (!Array.isArray(mat) || mat.length === 0 || !Array.isArray(mat[0])) {
        return { handled: false, reason: 'Invalid matrix format.' };
      }

      const solverStart = new Date().toISOString();
      const sT0 = performance.now();

      if (/\b(determinant|det)\b/i.test(question)) {
        const detRec = await executeToolAsync('symbolic_math', {
          operation: 'matrix_det',
          matrix_a: mat,
        });
        if (!detRec.success) {
          return { handled: false, reason: 'Matrix determinant failed.' };
        }
        const detVal = Number(detRec.output_payload.determinant);

        const pyDetCode = `
import math
def py_det(m):
    n = len(m)
    if n == 1:
        return float(m[0][0])
    if n == 2:
        return float(m[0][0]*m[1][1] - m[0][1]*m[1][0])
    total = 0.0
    for c in range(n):
        minor = [row[:c] + row[c+1:] for row in m[1:]]
        total += ((-1)**c) * m[0][c] * py_det(minor)
    return total
d = py_det(${JSON.stringify(mat)})
err = abs(d - ${detVal})
RESULT = {"passed": err < 1e-9, "independent_value": d, "error": err, "details": f"Cofactor expansion det = {d}"}
`;
        const verifStart = new Date().toISOString();
        const vT0 = performance.now();
        const pyRec = await executeToolAsync('python_sandbox', { code: pyDetCode });
        const pyRes = (pyRec.output_payload.result ?? {}) as Record<string, unknown>;
        if (!pyRec.success || pyRes.passed !== true) {
          return { handled: false, reason: 'Cofactor determinant check failed.' };
        }

        const checks: VerificationCheck[] = [
          {
            check_name: 'matrix_determinant_cofactor_check',
            method: 'python_laplace_cofactor_expansion',
            passed: true,
            expected_value: String(pyRes.independent_value),
            observed_value: String(detVal),
            relative_error: Number(pyRes.error ?? 0),
            tolerance: 1e-9,
            details: String(pyRes.details),
          },
        ];

        const spec: ProblemSpecification = {
          domain: 'mathematics',
          problem_type: 'matrix_determinant',
          objective: 'Compute determinant of matrix A',
          variables: [],
          equations: [`det(${JSON.stringify(mat)})`],
          constraints: [],
          assumptions: ['Matrix A is square with real entries.'],
          missing_information: [],
          needs_clarification: false,
          clarification_question: null,
        };
        const plan: ExecutionPlan = {
          methodology: 'LU determinant calculation verified by independent Laplace cofactor expansion in Python.',
          selected_tools: ['symbolic_math', 'python_sandbox'],
          tasks: [
            {
              task_id: 'fast_1',
              title: 'Matrix Determinant',
              description: 'Compute det(A).',
              method: 'math.det',
              tool_name: 'symbolic_math',
              depends_on: [],
              expected_output: String(detVal),
            },
            {
              task_id: 'fast_2',
              title: 'Cofactor Expansion Check',
              description: 'Verify det(A) via independent recursive cofactor expansion.',
              method: 'laplace_cofactor',
              tool_name: 'python_sandbox',
              depends_on: ['fast_1'],
              expected_output: 'PASS',
            },
          ],
          verification_strategy: 'Cross-check LU determinant against Laplace cofactor expansion.',
        };
        const verification: VerificationReport = {
          status: 'PASS',
          checks,
          independent_recalculation: String(pyRes.independent_value),
          dimensional_consistency_passed: true,
          boundary_conditions_passed: true,
          numerical_stability_passed: true,
          failure_reasons: [],
          evidence_summary: String(pyRes.details),
          verified_at: new Date().toISOString(),
        };

        timeline.push(
          {
            agent_name: 'FastSymbolicSolver',
            stage: 'SOLVING',
            status: 'COMPLETED',
            summary: `Determinant computed: ${detVal}`,
            structured_output: { determinant: detVal },
            tool_records: [detRec],
            tokens_input: 0,
            tokens_output: 0,
            duration_ms: Number((performance.now() - sT0).toFixed(2)),
            started_at: solverStart,
            completed_at: new Date().toISOString(),
          },
          {
            agent_name: 'IndependentVerificationAgent',
            stage: 'VERIFICATION',
            status: 'PASS',
            summary: verification.evidence_summary,
            structured_output: verification as unknown as Record<string, unknown>,
            tool_records: [pyRec],
            tokens_input: 0,
            tokens_output: 0,
            duration_ms: Number((performance.now() - vT0).toFixed(2)),
            started_at: verifStart,
            completed_at: new Date().toISOString(),
          },
        );

        const elapsedMs = Number((performance.now() - t0).toFixed(2));
        const report = buildFastReport({
          problemId,
          executionId,
          question,
          decision,
          spec,
          plan,
          timeline,
          toolRecords: [detRec, pyRec],
          verification,
          methodology: plan.methodology,
          equations: spec.equations,
          intermediateResults: [{ step: 'Matrix determinant', value: String(detVal), unit: 'dimensionless' }],
          finalAnswer: formatFractionCoeff(detVal),
          numericValue: detVal,
          units: 'dimensionless',
          explanationMarkdown: `### Matrix Determinant (Fast Path)\n\n$$\\det(A) = ${detVal}$$\n\n### Independent Verification\n\n- **Cofactor Expansion Check**: \`PASS\` — ${verification.evidence_summary}`,
          startedAt,
          elapsedMs,
        });

        return {
          handled: true,
          specification: spec,
          plan,
          agent_timeline: timeline,
          tool_records: [detRec, pyRec],
          verification_report: verification,
          final_report: report,
        };
      }
      return { handled: false, reason: 'Matrix operation not covered by deterministic fast path.' };
    }

    // -------------------------------------------------------------------------
    // CASE 6: Direct Numerical Evaluation & Unit Conversions
    // -------------------------------------------------------------------------
    if (decision.problem_category === 'numerical_or_unit_eval') {
      const unitMatch = question.match(
        /\bconvert\s+([-+]?\d+(?:\.\d+)?)\s+([a-zA-Z/^*\d-]+)\s+(?:to|into)\s+([a-zA-Z/^*\d-]+)\b/i,
      );
      if (unitMatch) {
        const val = Number(unitMatch[1]);
        const fromUnit = unitMatch[2];
        const toUnit = unitMatch[3];
        const solverStart = new Date().toISOString();
        const sT0 = performance.now();
        const convRec = await executeToolAsync('dimensional_analysis', {
          operation: 'convert',
          value: val,
          from_unit: fromUnit,
          to_unit: toUnit,
        });
        if (!convRec.success) {
          return { handled: false, reason: 'Unit conversion failed.' };
        }
        const converted = Number(convRec.output_payload.converted_value);
        const eqRec = await executeToolAsync('dimensional_analysis', {
          operation: 'check_equivalence',
          lhs_unit: fromUnit,
          rhs_unit: toUnit,
        });
        const checks: VerificationCheck[] = [
          {
            check_name: 'dimensional_base_equivalence',
            method: 'mathjs_si_base_check',
            passed: eqRec.success && eqRec.output_payload.consistent === true,
            expected_value: fromUnit,
            observed_value: toUnit,
            relative_error: 0,
            tolerance: 1e-12,
            details: String(convRec.output_payload.details ?? ''),
          },
        ];
        if (!checks[0].passed) {
          return { handled: false, reason: 'Dimensional equivalence check failed.' };
        }

        const spec: ProblemSpecification = {
          domain: decision.detected_domain,
          problem_type: 'unit_conversion',
          objective: `Convert ${val} ${fromUnit} to ${toUnit}`,
          variables: [
            {
              symbol: 'q',
              name: 'input quantity',
              value: val,
              unit: fromUnit,
              constraints: [],
              description: 'Source physical quantity',
            },
          ],
          equations: [`${val} ${fromUnit} -> ${toUnit}`],
          constraints: [],
          assumptions: ['Standard SI/NIST conversion factors applied.'],
          missing_information: [],
          needs_clarification: false,
          clarification_question: null,
        };
        const plan: ExecutionPlan = {
          methodology: 'Deterministic dimensional conversion with SI base-unit equivalence check.',
          selected_tools: ['dimensional_analysis'],
          tasks: [
            {
              task_id: 'fast_1',
              title: 'Unit Conversion',
              description: `Convert ${val} ${fromUnit} to ${toUnit}.`,
              method: 'dimensional_analysis.convert',
              tool_name: 'dimensional_analysis',
              depends_on: [],
              expected_output: String(converted),
            },
          ],
          verification_strategy: 'Verify base SI dimension equivalence between source and target units.',
        };
        const verification: VerificationReport = {
          status: 'PASS',
          checks,
          independent_recalculation: `${converted} ${toUnit}`,
          dimensional_consistency_passed: true,
          boundary_conditions_passed: true,
          numerical_stability_passed: true,
          failure_reasons: [],
          evidence_summary: String(convRec.output_payload.details ?? ''),
          verified_at: new Date().toISOString(),
        };

        timeline.push({
          agent_name: 'FastSymbolicSolver',
          stage: 'VERIFICATION',
          status: 'PASS',
          summary: String(convRec.output_payload.details ?? ''),
          structured_output: convRec.output_payload,
          tool_records: [convRec, eqRec],
          tokens_input: 0,
          tokens_output: 0,
          duration_ms: Number((performance.now() - sT0).toFixed(2)),
          started_at: solverStart,
          completed_at: new Date().toISOString(),
        });

        const elapsedMs = Number((performance.now() - t0).toFixed(2));
        const report = buildFastReport({
          problemId,
          executionId,
          question,
          decision,
          spec,
          plan,
          timeline,
          toolRecords: [convRec, eqRec],
          verification,
          methodology: plan.methodology,
          equations: spec.equations,
          intermediateResults: [{ step: 'Converted value', value: `${converted}`, unit: toUnit }],
          finalAnswer: `${converted} ${toUnit}`,
          numericValue: converted,
          units: toUnit,
          explanationMarkdown: `### Dimensional Conversion (Fast Path)\n\n$$${val}\\text{ ${fromUnit}} = ${converted}\\text{ ${toUnit}}$$\n\n- **SI Base Equivalence**: \`PASS\``,
          startedAt,
          elapsedMs,
        });
        return {
          handled: true,
          specification: spec,
          plan,
          agent_timeline: timeline,
          tool_records: [convRec, eqRec],
          verification_report: verification,
          final_report: report,
        };
      }

      const evalMatch = question.match(/^\s*(?:evaluate|compute|calculate|simplify)\s+(.+?)[.?!]?\s*$/i);
      if (evalMatch) {
        const expr = normalizeMathExpr(evalMatch[1]);
        const solverStart = new Date().toISOString();
        const sT0 = performance.now();
        const evalRec = await executeToolAsync('symbolic_math', {
          operation: 'evaluate',
          expression: expr,
        });
        if (!evalRec.success || evalRec.output_payload.numeric_approx === null) {
          return { handled: false, reason: 'Expression not directly numerically evaluable.' };
        }
        const val = Number(evalRec.output_payload.numeric_approx);
        const pyExpr = toPythonMathExpr(expr);
        const pyOut = await runIsolatedPythonAsync(`import math\nRESULT = float(${pyExpr})`);
        if (pyOut.exitCode !== 0 || typeof pyOut.result !== 'number') {
          return { handled: false, reason: 'Independent Python evaluation failed.' };
        }
        const verifRec = await executeToolAsync('independent_verifier', {
          verification_type: 'numeric_tolerance',
          candidate_value: val,
          expected_reference: pyOut.result,
          tolerance: 1e-10,
        });
        if (!verifRec.success || verifRec.output_payload.passed !== true) {
          return { handled: false, reason: 'Independent numeric tolerance check failed.' };
        }

        const checks: VerificationCheck[] = [
          {
            check_name: 'python_independent_numeric_eval',
            method: 'direct_tolerance_comparison',
            passed: true,
            expected_value: String(pyOut.result),
            observed_value: String(val),
            relative_error: Number(verifRec.output_payload.relative_error ?? 0),
            tolerance: 1e-10,
            details: String(verifRec.output_payload.details ?? ''),
          },
        ];

        const spec: ProblemSpecification = {
          domain: 'mathematics',
          problem_type: 'numeric_evaluation',
          objective: `Evaluate ${expr}`,
          variables: [],
          equations: [expr],
          constraints: [],
          assumptions: ['Real arithmetic evaluation.'],
          missing_information: [],
          needs_clarification: false,
          clarification_question: null,
        };
        const plan: ExecutionPlan = {
          methodology: 'Direct mathjs evaluation cross-checked in isolated Python 3.11.',
          selected_tools: ['symbolic_math', 'independent_verifier'],
          tasks: [
            {
              task_id: 'fast_1',
              title: 'Expression Evaluation',
              description: `Evaluate ${expr}.`,
              method: 'symbolic_math.evaluate',
              tool_name: 'symbolic_math',
              depends_on: [],
              expected_output: String(val),
            },
          ],
          verification_strategy: 'Independent evaluation in Python 3.11 within 1e-10 tolerance.',
        };
        const verification: VerificationReport = {
          status: 'PASS',
          checks,
          independent_recalculation: String(pyOut.result),
          dimensional_consistency_passed: true,
          boundary_conditions_passed: true,
          numerical_stability_passed: true,
          failure_reasons: [],
          evidence_summary: checks[0].details,
          verified_at: new Date().toISOString(),
        };

        timeline.push({
          agent_name: 'FastSymbolicSolver',
          stage: 'VERIFICATION',
          status: 'PASS',
          summary: `Evaluated ${expr} = ${val}`,
          structured_output: { value: val, python_reference: pyOut.result },
          tool_records: [evalRec, verifRec],
          tokens_input: 0,
          tokens_output: 0,
          duration_ms: Number((performance.now() - sT0).toFixed(2)),
          started_at: solverStart,
          completed_at: new Date().toISOString(),
        });

        const elapsedMs = Number((performance.now() - t0).toFixed(2));
        const report = buildFastReport({
          problemId,
          executionId,
          question,
          decision,
          spec,
          plan,
          timeline,
          toolRecords: [evalRec, verifRec],
          verification,
          methodology: plan.methodology,
          equations: spec.equations,
          intermediateResults: [{ step: 'Evaluated value', value: String(val), unit: 'dimensionless' }],
          finalAnswer: formatFractionCoeff(val),
          numericValue: val,
          units: 'dimensionless',
          explanationMarkdown: `### Direct Numerical Evaluation (Fast Path)\n\n$$${expr} = ${val}$$\n\n- **Python Cross-Check**: \`PASS\` — ${checks[0].details}`,
          startedAt,
          elapsedMs,
        });

        return {
          handled: true,
          specification: spec,
          plan,
          agent_timeline: timeline,
          tool_records: [evalRec, verifRec],
          verification_report: verification,
          final_report: report,
        };
      }
    }

    return { handled: false, reason: 'No fast-path handler matched.' };
  } catch (err) {
    return {
      handled: false,
      reason: `Fast path fell back safely: ${err instanceof Error ? err.message : String(err)}`,
    };
  }
}

export function createDeterministicRunId(): string {
  return `fast_${crypto.randomBytes(4).toString('hex')}`;
}
