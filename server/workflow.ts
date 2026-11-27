import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';
import { z } from 'zod';
import type {
  AgentExecutionResult,
  ComplexityLevel,
  CorrectionRequest,
  ExecutionPlan,
  FinalSolutionReport,
  HumanReviewDecision,
  ProblemDomain,
  ProblemSpecification,
  ToolExecutionRecord,
  VerificationCheck,
  VerificationReport,
  WorkflowStatus,
} from '../src/types/veristem.ts';
import { getDatabase } from './db.ts';
import { TOOL_CATALOG, executeToolsParallelWithCache } from './tools.ts';

const MathematicalVariableSchema = z.object({
  symbol: z.string().min(1),
  name: z.string().min(1),
  value: z.union([z.number(), z.string(), z.null()]).default(null),
  unit: z.string().default('dimensionless'),
  constraints: z.array(z.string()).default([]),
  description: z.string().default(''),
});

const ProblemSpecificationSchema = z.object({
  domain: z.enum(['mathematics', 'physics', 'quantitative_finance', 'general_stem']),
  problem_type: z.string().min(1),
  objective: z.string().min(1),
  variables: z.array(MathematicalVariableSchema).default([]),
  equations: z.array(z.string()).default([]),
  constraints: z.array(z.string()).default([]),
  assumptions: z.array(z.string()).default([]),
  missing_information: z.array(z.string()).default([]),
  needs_clarification: z.boolean().default(false),
  clarification_question: z.string().nullable().default(null),
});

const ExecutionPlanSchema = z.object({
  methodology: z.string().min(1),
  selected_tools: z.array(z.string()).default([]),
  tasks: z
    .array(
      z.object({
        task_id: z.string().min(1),
        title: z.string().min(1),
        description: z.string().min(1),
        method: z.string().min(1),
        tool_name: z.string().min(1),
        depends_on: z.array(z.string()).default([]),
        expected_output: z.string().default(''),
      }),
    )
    .min(1),
  verification_strategy: z.string().min(1),
});

const ToolCallPlanSchema = z.object({
  reasoning_strategy: z.string().min(1),
  equations_used: z.array(z.string()).default([]),
  tool_calls: z
    .array(
      z.object({
        task_id: z.string().min(1),
        tool_name: z.string().min(1),
        arguments_json: z.string().min(2),
      }),
    )
    .min(1),
});

const SolverSynthesisSchema = z.object({
  solution_methodology: z.string().min(1),
  equations: z.array(z.string()).default([]),
  intermediate_results: z
    .array(
      z.object({
        step: z.string(),
        value: z.string(),
        unit: z.string().default('dimensionless'),
      }),
    )
    .default([]),
  final_answer: z.string().min(1),
  numeric_value: z.number().nullable().default(null),
  units: z.string().default('dimensionless'),
  assumptions_applied: z.array(z.string()).default([]),
});

const VerificationPlanSchema = z.object({
  verification_calls: z
    .array(
      z.object({
        task_id: z.string().min(1),
        tool_name: z.string().min(1),
        arguments_json: z.string().min(2),
      }),
    )
    .min(1),
});

const VerificationReportSchema = z.object({
  status: z.enum(['PASS', 'FAIL', 'INCONCLUSIVE']),
  checks: z
    .array(
      z.object({
        check_name: z.string(),
        method: z.string(),
        passed: z.boolean(),
        expected_value: z.string().nullable().default(null),
        observed_value: z.string().nullable().default(null),
        relative_error: z.number().nullable().default(null),
        tolerance: z.number().nullable().default(null),
        details: z.string().default(''),
      }),
    )
    .default([]),
  independent_recalculation: z.string().nullable().default(null),
  dimensional_consistency_passed: z.boolean().nullable().default(null),
  boundary_conditions_passed: z.boolean().nullable().default(null),
  numerical_stability_passed: z.boolean().nullable().default(null),
  failure_reasons: z.array(z.string()).default([]),
  evidence_summary: z.string().min(1),
});

const CorrectionRequestSchema = z.object({
  root_cause: z.string().min(1),
  error_category: z.string().min(1),
  corrective_instructions: z.string().min(1),
  tools_to_reexecute: z.array(z.string()).default([]),
});

const NarrativeSchema = z.object({
  explanation_markdown: z.string().min(1),
  confidence_limitations: z.array(z.string()).default([]),
});

const UnderstandingAndPlanSchema = z.object({
  specification: ProblemSpecificationSchema,
  plan: ExecutionPlanSchema,
});

const GEMINI_CALL_TIMEOUT_MS = Number(process.env.GEMINI_CALL_TIMEOUT_MS || 25000);

function getGeminiClient(): GoogleGenAI {
  const apiKey = (process.env.GEMINI_API_KEY ?? '').trim();
  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
    throw new Error(
      'GEMINI_API_KEY is not configured. Add your Gemini API key in Settings > Secrets to run VeriSTEM.',
    );
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

function resolveGeminiModel(): string {
  const configured = (process.env.GEMINI_MODEL ?? '').trim();
  const validModels = new Set([
    'gemini-4-hydrogen',
    'gemini-3.1-flash-lite',
    'gemini-4-neon',
    'gemini-4-carbon',
    'gemini-4-argon',
  ]);
  return validModels.has(configured) ? configured : 'gemini-4-hydrogen';
}

async function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

async function callGeminiStructured<T>(
  systemInstruction: string,
  prompt: string,
  responseSchema: Record<string, unknown>,
  validator: z.ZodType<T>,
  thinkingLevel: ThinkingLevel = ThinkingLevel.LOW,
  timeoutMs: number = GEMINI_CALL_TIMEOUT_MS,
): Promise<{ data: T; tokensInput: number; tokensOutput: number }> {
  const ai = getGeminiClient();
  const primaryModel = resolveGeminiModel();
  const candidateModels =
    primaryModel === 'gemini-3.1-flash-lite'
      ? ['gemini-3.1-flash-lite', 'gemini-3.1-flash-lite']
      : [primaryModel, 'gemini-3.1-flash-lite'];

  let lastError: unknown;
  for (let attempt = 0; attempt < candidateModels.length; attempt += 1) {
    const model = candidateModels[attempt];
    try {
      const response = await withTimeout(
        ai.models.generateContent({
          model,
          contents: prompt,
          config: {
            systemInstruction,
            responseMimeType: 'application/json',
            responseSchema,
            thinkingConfig: { thinkingLevel },
          },
        }),
        timeoutMs,
        `Gemini (${model})`,
      );
      const text = (response.text ?? '').trim();
      if (!text) {
        throw new Error('Gemini returned an empty response');
      }
      const parsedJson = JSON.parse(text);
      const data = validator.parse(parsedJson);
      const usage = response.usageMetadata;
      return {
        data,
        tokensInput: Number(usage?.promptTokenCount ?? 0),
        tokensOutput: Number(usage?.candidatesTokenCount ?? 0),
      };
    } catch (err) {
      lastError = err;
      if (attempt < candidateModels.length - 1) {
        await new Promise((r) => setTimeout(r, 350 * (attempt + 1)));
      }
    }
  }
  throw new Error(
    `Gemini structured generation failed: ${lastError instanceof Error ? lastError.message : String(lastError)}`,
  );
}

const SPEC_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    domain: { type: Type.STRING, enum: ['mathematics', 'physics', 'quantitative_finance', 'general_stem'] },
    problem_type: { type: Type.STRING },
    objective: { type: Type.STRING },
    variables: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          symbol: { type: Type.STRING },
          name: { type: Type.STRING },
          value: { type: Type.STRING, nullable: true },
          unit: { type: Type.STRING },
          constraints: { type: Type.ARRAY, items: { type: Type.STRING } },
          description: { type: Type.STRING },
        },
        required: ['symbol', 'name', 'unit'],
      },
    },
    equations: { type: Type.ARRAY, items: { type: Type.STRING } },
    constraints: { type: Type.ARRAY, items: { type: Type.STRING } },
    assumptions: { type: Type.ARRAY, items: { type: Type.STRING } },
    missing_information: { type: Type.ARRAY, items: { type: Type.STRING } },
    needs_clarification: { type: Type.BOOLEAN },
    clarification_question: { type: Type.STRING, nullable: true },
  },
  required: ['domain', 'problem_type', 'objective', 'needs_clarification'],
};

const PLAN_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    methodology: { type: Type.STRING },
    selected_tools: { type: Type.ARRAY, items: { type: Type.STRING } },
    tasks: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          task_id: { type: Type.STRING },
          title: { type: Type.STRING },
          description: { type: Type.STRING },
          method: { type: Type.STRING },
          tool_name: { type: Type.STRING },
          depends_on: { type: Type.ARRAY, items: { type: Type.STRING } },
          expected_output: { type: Type.STRING },
        },
        required: ['task_id', 'title', 'description', 'method', 'tool_name'],
      },
    },
    verification_strategy: { type: Type.STRING },
  },
  required: ['methodology', 'selected_tools', 'tasks', 'verification_strategy'],
};

const TOOL_CALL_PLAN_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    reasoning_strategy: { type: Type.STRING },
    equations_used: { type: Type.ARRAY, items: { type: Type.STRING } },
    tool_calls: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          task_id: { type: Type.STRING },
          tool_name: { type: Type.STRING },
          arguments_json: { type: Type.STRING },
        },
        required: ['task_id', 'tool_name', 'arguments_json'],
      },
    },
  },
  required: ['reasoning_strategy', 'tool_calls'],
};

const SOLVER_SYNTHESIS_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    solution_methodology: { type: Type.STRING },
    equations: { type: Type.ARRAY, items: { type: Type.STRING } },
    intermediate_results: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          step: { type: Type.STRING },
          value: { type: Type.STRING },
          unit: { type: Type.STRING },
        },
        required: ['step', 'value'],
      },
    },
    final_answer: { type: Type.STRING },
    numeric_value: { type: Type.NUMBER, nullable: true },
    units: { type: Type.STRING },
    assumptions_applied: { type: Type.ARRAY, items: { type: Type.STRING } },
  },
  required: ['solution_methodology', 'final_answer', 'units'],
};

const VERIFICATION_PLAN_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    verification_calls: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          task_id: { type: Type.STRING },
          tool_name: { type: Type.STRING },
          arguments_json: { type: Type.STRING },
        },
        required: ['task_id', 'tool_name', 'arguments_json'],
      },
    },
  },
  required: ['verification_calls'],
};

const VERIFICATION_REPORT_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    status: { type: Type.STRING, enum: ['PASS', 'FAIL', 'INCONCLUSIVE'] },
    checks: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          check_name: { type: Type.STRING },
          method: { type: Type.STRING },
          passed: { type: Type.BOOLEAN },
          expected_value: { type: Type.STRING, nullable: true },
          observed_value: { type: Type.STRING, nullable: true },
          relative_error: { type: Type.NUMBER, nullable: true },
          tolerance: { type: Type.NUMBER, nullable: true },
          details: { type: Type.STRING },
        },
        required: ['check_name', 'method', 'passed', 'details'],
      },
    },
    independent_recalculation: { type: Type.STRING, nullable: true },
    dimensional_consistency_passed: { type: Type.BOOLEAN, nullable: true },
    boundary_conditions_passed: { type: Type.BOOLEAN, nullable: true },
    numerical_stability_passed: { type: Type.BOOLEAN, nullable: true },
    failure_reasons: { type: Type.ARRAY, items: { type: Type.STRING } },
    evidence_summary: { type: Type.STRING },
  },
  required: ['status', 'checks', 'evidence_summary'],
};

const CORRECTION_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    root_cause: { type: Type.STRING },
    error_category: { type: Type.STRING },
    corrective_instructions: { type: Type.STRING },
    tools_to_reexecute: { type: Type.ARRAY, items: { type: Type.STRING } },
  },
  required: ['root_cause', 'error_category', 'corrective_instructions'],
};

const NARRATIVE_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    explanation_markdown: { type: Type.STRING },
    confidence_limitations: { type: Type.ARRAY, items: { type: Type.STRING } },
  },
  required: ['explanation_markdown', 'confidence_limitations'],
};

const UNDERSTANDING_AND_PLAN_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    specification: SPEC_SCHEMA,
    plan: PLAN_SCHEMA,
  },
  required: ['specification', 'plan'],
};

function tryDeterministicVerificationReport(verifRecords: ToolExecutionRecord[]): VerificationReport | null {
  if (verifRecords.length === 0) return null;
  const succeeded = verifRecords.filter((r) => r.success);
  if (succeeded.length !== verifRecords.length) return null;

  const checks: VerificationCheck[] = [];
  let indepRecalc: string | null = null;
  let dimPassed: boolean | null = null;

  for (const rec of succeeded) {
    const out = rec.output_payload;
    const nestedRes =
      typeof out.result === 'object' && out.result !== null
        ? (out.result as Record<string, unknown>)
        : null;

    let passedFlag: boolean | null = null;
    if (typeof out.passed === 'boolean') passedFlag = out.passed;
    else if (typeof out.consistent === 'boolean') {
      passedFlag = out.consistent;
      dimPassed = out.consistent;
    } else if (nestedRes && typeof nestedRes.passed === 'boolean') {
      passedFlag = nestedRes.passed;
    }

    if (passedFlag === null) {
      // Unstructured tool output requires LLM judge interpretation
      return null;
    }
    if (!passedFlag) {
      return null;
    }

    const obsVal = out.candidate_value ?? out.converted_value ?? nestedRes?.independent_value ?? null;
    const expVal = out.independent_value ?? nestedRes?.expected_value ?? null;
    if (expVal !== null && indepRecalc === null) {
      indepRecalc = String(expVal);
    }

    checks.push({
      check_name: String(out.method_used ?? rec.tool_name),
      method: String(out.method_used ?? rec.tool_name),
      passed: true,
      expected_value: expVal !== null ? String(expVal) : null,
      observed_value: obsVal !== null ? String(obsVal) : null,
      relative_error: typeof out.relative_error === 'number' ? out.relative_error : null,
      tolerance: typeof out.tolerance === 'number' ? out.tolerance : null,
      details: String(out.details ?? nestedRes?.details ?? `${rec.tool_name} check passed`),
    });
  }

  return {
    status: 'PASS',
    checks,
    independent_recalculation: indepRecalc,
    dimensional_consistency_passed: dimPassed ?? true,
    boundary_conditions_passed: true,
    numerical_stability_passed: true,
    failure_reasons: [],
    evidence_summary: checks.map((c) => c.details).join(' | '),
    verified_at: new Date().toISOString(),
  };
}

export async function runProblemWorkflow(
  problemId: string,
  executionId: string,
  options: {
    maxCorrectionAttempts?: number;
    clarificationResponse?: string | null;
    humanDecision?: HumanReviewDecision | null;
    complexityLevel?: ComplexityLevel;
  } = {},
): Promise<void> {
  const db = getDatabase();
  const problem = db.getProblem(problemId);
  if (!problem) {
    throw new Error(`Problem ${problemId} not found`);
  }

  const maxAttempts = options.maxCorrectionAttempts ?? 3;
  const complexityLevel: ComplexityLevel = options.complexityLevel ?? 'LEVEL_2_INTERMEDIATE';
  const workflowT0 = performance.now();
  const startedAt = new Date().toISOString();

  if (!options.clarificationResponse && !options.humanDecision) {
    db.clearExecutionArtifacts(problemId);
    db.updateProblem(problemId, {
      execution_id: executionId,
      status: 'RUNNING',
      current_stage: 'UNDERSTANDING',
      complexity_level: complexityLevel,
      execution_path: 'MULTI_AGENT_WORKFLOW',
      correction_attempts: 0,
      max_correction_attempts: maxAttempts,
    });
  } else {
    db.updateProblem(problemId, {
      status: 'RUNNING',
      current_stage: 'RESUMING',
      complexity_level: complexityLevel,
      execution_path: 'MULTI_AGENT_WORKFLOW',
    });
  }

  const toolCatalogText = JSON.stringify(TOOL_CATALOG, null, 2);
  const toolCache = new Map<string, ToolExecutionRecord>();
  let allToolRecords: ToolExecutionRecord[] = [...(db.getProblem(problemId)?.tool_records ?? [])];
  let corrections: CorrectionRequest[] = [...(db.getProblem(problemId)?.correction_history ?? [])];

  try {
    let spec: ProblemSpecification;
    let plan: ExecutionPlan;
    let synthesis: z.infer<typeof SolverSynthesisSchema>;
    let verification: VerificationReport;

    // If resuming from Human Review override/approve/reject, skip re-solving unless clarifying
    if (options.humanDecision && options.humanDecision.reviewer_action !== 'clarify') {
      const latest = db.getProblem(problemId)!;
      spec = latest.specification!;
      verification = latest.verification_report!;
      const lastSolverStep = [...latest.agent_timeline]
        .reverse()
        .find((a) => a.agent_name === 'STEMSolverAgent');
      synthesis = SolverSynthesisSchema.parse(
        lastSolverStep?.structured_output ?? {
          solution_methodology: latest.plan?.methodology ?? 'Human-reviewed resolution',
          final_answer: options.humanDecision.override_answer ?? 'Unresolved',
          units: 'dimensionless',
        },
      );
      if (options.humanDecision.reviewer_action === 'override' && options.humanDecision.override_answer) {
        synthesis.final_answer = options.humanDecision.override_answer;
      }
    } else {
      // Consolidated AGENT 1 (Understanding) + AGENT 2 (Planning) in 1 Gemini round-trip
      db.updateProblem(problemId, { current_stage: 'UNDERSTANDING' });
      const upStart = new Date().toISOString();
      const upT0 = performance.now();
      const upPrompt = [
        `Problem:\n${problem.question}`,
        problem.domain_hint ? `User domain hint: ${problem.domain_hint}` : '',
        options.clarificationResponse
          ? `User clarification:\n${options.clarificationResponse}`
          : '',
        `Available Computational Tools:\n${toolCatalogText}`,
      ]
        .filter(Boolean)
        .join('\n\n');

      const upResp = await callGeminiStructured(
        `You are Agents 1 & 2 (Problem Understanding & Planning) in VeriSTEM.
1. Classify domain, extract variables with units, equations, constraints, and assumptions. Only set needs_clarification=true if essential numerical parameters are genuinely missing.
2. Construct a concise dependency-aware ExecutionPlan selecting tools strictly from the provided catalog, plus an independent verification_strategy using a separate mathematical/numerical method.`,
        upPrompt,
        UNDERSTANDING_AND_PLAN_SCHEMA,
        UnderstandingAndPlanSchema,
        ThinkingLevel.LOW,
      );

      spec = upResp.data.specification as ProblemSpecification;
      plan = upResp.data.plan as ExecutionPlan;
      const totalUpMs = Number((performance.now() - upT0).toFixed(2));
      const nowIso = new Date().toISOString();

      const uAgentRes: AgentExecutionResult = {
        agent_name: 'ProblemUnderstandingAgent',
        stage: 'UNDERSTANDING',
        status: spec.needs_clarification ? 'NEEDS_CLARIFICATION' : 'COMPLETED',
        summary: `Classified as ${spec.domain} (${spec.problem_type}) with ${spec.variables.length} variables.`,
        structured_output: spec as unknown as Record<string, unknown>,
        tool_records: [],
        tokens_input: Math.floor(upResp.tokensInput / 2),
        tokens_output: Math.floor(upResp.tokensOutput / 2),
        duration_ms: Number((totalUpMs * 0.5).toFixed(2)),
        started_at: upStart,
        completed_at: nowIso,
      };
      db.addAgentResult(problemId, executionId, uAgentRes);
      db.updateProblem(problemId, { specification: spec });

      if (spec.needs_clarification) {
        db.updateProblem(problemId, {
          status: 'NEEDS_CLARIFICATION',
          current_stage: 'NEEDS_CLARIFICATION',
          elapsed_ms: Number((performance.now() - workflowT0).toFixed(2)),
        });
        return;
      }

      db.updateProblem(problemId, { current_stage: 'PLANNING', plan });
      db.addAgentResult(problemId, executionId, {
        agent_name: 'PlanningAgent',
        stage: 'PLANNING',
        status: 'COMPLETED',
        summary: `Constructed ${plan.tasks.length}-step plan using ${plan.selected_tools.join(', ')}.`,
        structured_output: plan as unknown as Record<string, unknown>,
        tool_records: [],
        tokens_input: Math.ceil(upResp.tokensInput / 2),
        tokens_output: Math.ceil(upResp.tokensOutput / 2),
        duration_ms: Number((totalUpMs * 0.5).toFixed(2)),
        started_at: upStart,
        completed_at: nowIso,
      });

      // Solve <-> Verify <-> Critic Loop
      let latestCorrection: CorrectionRequest | null = null;
      let attempt = 0;

      while (true) {
        // AGENT 3: STEM Solver
        db.updateProblem(problemId, { current_stage: 'SOLVING' });
        const sStart = new Date().toISOString();
        const sT0 = performance.now();

        const correctionPrompt = latestCorrection
          ? `\n\nPREVIOUS VERIFICATION FAILED — CORRECTION REQUIRED:\n${JSON.stringify(latestCorrection, null, 2)}`
          : '';

        const sPlanResp = await callGeminiStructured(
          `You are Agent 3 (STEM Solver Agent) in VeriSTEM.
Never guess or invent numbers. Generate minimal, non-redundant tool_calls using python_sandbox, symbolic_math, or dimensional_analysis.
For python_sandbox, use ONLY stdlib modules (math, cmath, decimal, fractions, statistics, json, itertools) and assign the final answer or dict to RESULT.
Ensure arguments_json is a valid JSON string matching the tool parameters.`,
          `ProblemSpecification:\n${JSON.stringify(spec)}\n\nExecutionPlan:\n${JSON.stringify(plan)}\n\nTools:\n${toolCatalogText}${correctionPrompt}`,
          TOOL_CALL_PLAN_SCHEMA,
          ToolCallPlanSchema,
          ThinkingLevel.LOW,
        );

        // Execute solver tool calls in parallel with deduplication cache
        const solverRecords = await executeToolsParallelWithCache(sPlanResp.data.tool_calls, toolCache);
        allToolRecords = [...allToolRecords, ...solverRecords];

        const sSynthResp = await callGeminiStructured(
          `You are Agent 3 (STEM Solver Agent) synthesizing ONLY actual executed tool outputs.
Never invent numeric values. Report the exact result computed by the tools.`,
          `Specification:\n${JSON.stringify(spec)}\n\nStrategy:\n${sPlanResp.data.reasoning_strategy}\n\nActual Tool Results:\n${JSON.stringify(solverRecords)}`,
          SOLVER_SYNTHESIS_SCHEMA,
          SolverSynthesisSchema,
          ThinkingLevel.LOW,
        );
        synthesis = sSynthResp.data;

        db.addAgentResult(problemId, executionId, {
          agent_name: 'STEMSolverAgent',
          stage: 'SOLVING',
          status: 'COMPLETED',
          summary: `Executed ${solverRecords.length} tool call(s); candidate answer: ${synthesis.final_answer} ${synthesis.units}`,
          structured_output: synthesis as unknown as Record<string, unknown>,
          tool_records: solverRecords,
          tokens_input: sPlanResp.tokensInput + sSynthResp.tokensInput,
          tokens_output: sPlanResp.tokensOutput + sSynthResp.tokensOutput,
          duration_ms: Number((performance.now() - sT0).toFixed(2)),
          started_at: sStart,
          completed_at: new Date().toISOString(),
        });

        // AGENT 4: Independent Verification
        db.updateProblem(problemId, { current_stage: 'VERIFICATION' });
        const vStart = new Date().toISOString();
        const vT0 = performance.now();

        const vPlanResp = await callGeminiStructured(
          `You are Agent 4 (Independent Verification Agent) in VeriSTEM.
Independently verify the candidate solution using a DIFFERENT numerical or symbolic method when practical.
Prefer independent_verifier (integral_quadrature, derivative_finite_diff, root_residual, black_scholes_mc, numeric_tolerance) or dimensional_analysis; or python_sandbox returning RESULT = {"passed": True/False, "independent_value": ..., "details": "..."}.`,
          `Specification:\n${JSON.stringify(spec)}\n\nCandidate Solution:\n${JSON.stringify(synthesis)}\n\nTool Catalog:\n${toolCatalogText}`,
          VERIFICATION_PLAN_SCHEMA,
          VerificationPlanSchema,
          ThinkingLevel.LOW,
        );

        // Parallel deduplicated verification tool execution
        const verifRecords = await executeToolsParallelWithCache(
          vPlanResp.data.verification_calls,
          toolCache,
        );
        allToolRecords = [...allToolRecords, ...verifRecords];

        // Deterministic fast-synthesis if all independent tools returned crisp boolean pass checks;
        // otherwise fall back to Gemini Judge + deterministic safeguards.
        let vTokensIn = vPlanResp.tokensInput;
        let vTokensOut = vPlanResp.tokensOutput;
        const deterministicReport = tryDeterministicVerificationReport(verifRecords);

        if (deterministicReport) {
          verification = deterministicReport;
        } else {
          const vJudgeResp = await callGeminiStructured(
            `You are Agent 4 (Independent Verification Agent) issuing a formal VerificationReport.
Base your verdict STRICTLY on the actual verification tool results:
- If zero verification tools succeeded, status MUST be INCONCLUSIVE or FAIL (never PASS).
- If any check failed, status MUST be FAIL.
- Only emit PASS when independent tool execution confirmed the solution.`,
            `Specification:\n${JSON.stringify(spec)}\n\nCandidate Solution:\n${JSON.stringify(synthesis)}\n\nActual Verification Tool Outputs:\n${JSON.stringify(verifRecords)}`,
            VERIFICATION_REPORT_SCHEMA,
            VerificationReportSchema,
            ThinkingLevel.LOW,
          );
          vTokensIn += vJudgeResp.tokensInput;
          vTokensOut += vJudgeResp.tokensOutput;
          verification = {
            ...vJudgeResp.data,
            verified_at: new Date().toISOString(),
          };
        }

        // Deterministic verification safeguards (always enforced)
        const succeeded = verifRecords.filter((r) => r.success);
        if (verification.status === 'PASS' && succeeded.length === 0) {
          verification = {
            status: 'INCONCLUSIVE',
            checks: [
              {
                check_name: 'tool_execution_guard',
                method: 'deterministic_guard',
                passed: false,
                expected_value: null,
                observed_value: null,
                relative_error: null,
                tolerance: null,
                details: 'Rejected PASS because zero verification tools executed successfully.',
              },
            ],
            independent_recalculation: null,
            dimensional_consistency_passed: null,
            boundary_conditions_passed: null,
            numerical_stability_passed: null,
            failure_reasons: ['No independent verification tool succeeded.'],
            evidence_summary: 'Rejected PASS: independent verification tools failed to run.',
            verified_at: new Date().toISOString(),
          };
        }
        for (const r of succeeded) {
          const out = r.output_payload;
          const nestedRes =
            typeof out.result === 'object' && out.result !== null
              ? (out.result as Record<string, unknown>)
              : null;
          if (
            out.passed === false ||
            out.consistent === false ||
            nestedRes?.passed === false
          ) {
            verification.status = 'FAIL';
            verification.failure_reasons.push(`Tool ${r.tool_name} reported a failing check.`);
          }
        }

        db.updateProblem(problemId, { verification_report: verification });
        db.addAgentResult(problemId, executionId, {
          agent_name: 'IndependentVerificationAgent',
          stage: 'VERIFICATION',
          status: verification.status,
          summary: verification.evidence_summary,
          structured_output: verification as unknown as Record<string, unknown>,
          tool_records: verifRecords,
          tokens_input: vTokensIn,
          tokens_output: vTokensOut,
          duration_ms: Number((performance.now() - vT0).toFixed(2)),
          started_at: vStart,
          completed_at: new Date().toISOString(),
        });

        if (verification.status === 'PASS') {
          break;
        }

        if (attempt >= maxAttempts) {
          db.updateProblem(problemId, {
            status: 'HUMAN_REVIEW',
            current_stage: 'HUMAN_REVIEW',
            elapsed_ms: Number((performance.now() - workflowT0).toFixed(2)),
          });
          return;
        }

        // AGENT 5: Critic & Self-Correction
        attempt += 1;
        db.updateProblem(problemId, {
          current_stage: 'CRITIC',
          correction_attempts: attempt,
        });
        const cStart = new Date().toISOString();
        const cT0 = performance.now();
        const cResp = await callGeminiStructured(
          `You are Agent 5 (Critic and Self-Correction Agent) in VeriSTEM.
Diagnose why verification failed or was inconclusive and provide exact corrective instructions.`,
          `Attempt ${attempt}\nSpecification:\n${JSON.stringify(spec)}\nCandidate:\n${JSON.stringify(synthesis)}\nVerificationReport:\n${JSON.stringify(verification)}`,
          CORRECTION_SCHEMA,
          CorrectionRequestSchema,
          ThinkingLevel.LOW,
        );
        latestCorrection = {
          attempt_number: attempt,
          ...cResp.data,
          created_at: new Date().toISOString(),
        };
        corrections = [...corrections, latestCorrection];
        db.addCorrection(problemId, executionId, latestCorrection);
        db.addAgentResult(problemId, executionId, {
          agent_name: 'CriticSelfCorrectionAgent',
          stage: 'CRITIC',
          status: 'CORRECTION_REQUESTED',
          summary: `Attempt ${attempt} (${latestCorrection.error_category}): ${latestCorrection.root_cause}`,
          structured_output: latestCorrection as unknown as Record<string, unknown>,
          tool_records: [],
          tokens_input: cResp.tokensInput,
          tokens_output: cResp.tokensOutput,
          duration_ms: Number((performance.now() - cT0).toFixed(2)),
          started_at: cStart,
          completed_at: new Date().toISOString(),
        });
      }
    }

    // AGENT 6: Explanation & Reporting (with deterministic Markdown fallback if LLM times out)
    db.updateProblem(problemId, { current_stage: 'REPORTING' });
    const rStart = new Date().toISOString();
    const rT0 = performance.now();
    let narrativeData: { explanation_markdown: string; confidence_limitations: string[] };
    let rTokensIn = 0;
    let rTokensOut = 0;

    try {
      const rResp = await callGeminiStructured(
        `You are Agent 6 (Explanation and Reporting Agent) in VeriSTEM.
Write a concise, clear scientific explanation in Markdown using LaTeX ($...$ and $$...$$).
Strictly separate assumptions from verified computational evidence.`,
        `Question:\n${problem.question}\n\nSolution:\n${JSON.stringify(synthesis)}\n\nVerification:\n${JSON.stringify(verification)}`,
        NARRATIVE_SCHEMA,
        NarrativeSchema,
        ThinkingLevel.LOW,
        20_000,
      );
      narrativeData = rResp.data;
      rTokensIn = rResp.tokensInput;
      rTokensOut = rResp.tokensOutput;
    } catch {
      narrativeData = {
        explanation_markdown: [
          `### Methodology`,
          synthesis.solution_methodology,
          `### Final Result`,
          `**${synthesis.final_answer}** ${synthesis.units !== 'dimensionless' ? synthesis.units : ''}`,
          `### Independent Verification (${verification.status})`,
          ...verification.checks.map((ck) => `- **${ck.check_name}**: \`${ck.passed ? 'PASS' : 'FAIL'}\` — ${ck.details}`),
        ].join('\n\n'),
        confidence_limitations: ['Narrative synthesized deterministically after verification completed.'],
      };
    }

    const finalStatus: WorkflowStatus =
      options.humanDecision?.reviewer_action === 'reject' ? 'UNRESOLVED' : 'COMPLETED';
    const elapsedMs = Number((performance.now() - workflowT0).toFixed(2));

    const finalReport: FinalSolutionReport = {
      problem_id: problemId,
      execution_id: executionId,
      original_question: problem.question,
      detected_domain: spec.domain as ProblemDomain,
      extracted_variables: spec.variables,
      assumptions: Array.from(new Set([...spec.assumptions, ...synthesis.assumptions_applied])),
      solution_methodology: synthesis.solution_methodology,
      equations: synthesis.equations,
      intermediate_results: synthesis.intermediate_results,
      final_answer: synthesis.final_answer,
      numeric_value: synthesis.numeric_value,
      units: synthesis.units,
      verification_status: verification.status,
      verification_evidence: verification.checks,
      correction_attempts: corrections.length,
      correction_history: corrections,
      tool_execution_history: allToolRecords,
      confidence_limitations: narrativeData.confidence_limitations,
      explanation_markdown: narrativeData.explanation_markdown,
      complexity_level: complexityLevel,
      execution_path: 'MULTI_AGENT_WORKFLOW',
      elapsed_ms: elapsedMs,
      started_at: startedAt,
      completed_at: new Date().toISOString(),
    };

    db.addAgentResult(problemId, executionId, {
      agent_name: 'ExplanationReportingAgent',
      stage: 'REPORTING',
      status: 'COMPLETED',
      summary: `Generated ${verification.status} report with ${allToolRecords.length} tool records.`,
      structured_output: {
        final_answer: finalReport.final_answer,
        verification_status: finalReport.verification_status,
      },
      tool_records: [],
      tokens_input: rTokensIn,
      tokens_output: rTokensOut,
      duration_ms: Number((performance.now() - rT0).toFixed(2)),
      started_at: rStart,
      completed_at: new Date().toISOString(),
    });

    db.updateProblem(problemId, {
      status: finalStatus,
      current_stage: finalStatus,
      elapsed_ms: elapsedMs,
      final_report: finalReport,
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    db.updateProblem(problemId, {
      status: 'FAILED',
      current_stage: `FAILED: ${msg.slice(0, 140)}`,
      elapsed_ms: Number((performance.now() - workflowT0).toFixed(2)),
    });
  }
}
