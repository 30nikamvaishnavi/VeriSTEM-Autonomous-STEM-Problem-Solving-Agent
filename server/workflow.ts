import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';
import { z } from 'zod';
import type {
  AgentExecutionResult,
  CorrectionRequest,
  ExecutionPlan,
  FinalSolutionReport,
  HumanReviewDecision,
  ProblemDomain,
  ProblemSpecification,
  ToolExecutionRecord,
  VerificationReport,
  WorkflowStatus,
} from '../src/types/veristem.ts';
import { getDatabase } from './db.ts';
import { TOOL_CATALOG, executeTool } from './tools.ts';

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

async function callGeminiStructured<T>(
  systemInstruction: string,
  prompt: string,
  responseSchema: Record<string, unknown>,
  validator: z.ZodType<T>,
  thinkingLevel: ThinkingLevel = ThinkingLevel.LOW,
): Promise<{ data: T; tokensInput: number; tokensOutput: number }> {
  const ai = getGeminiClient();
  const primaryModel = resolveGeminiModel();
  const candidateModels =
    primaryModel === 'gemini-3.1-flash-lite'
      ? ['gemini-3.1-flash-lite', 'gemini-3.1-flash-lite']
      : [primaryModel, 'gemini-3.1-flash-lite', 'gemini-3.1-flash-lite'];

  let lastError: unknown;
  for (let attempt = 0; attempt < candidateModels.length; attempt += 1) {
    const model = candidateModels[attempt];
    try {
      const response = await ai.models.generateContent({
        model,
        contents: prompt,
        config: {
          systemInstruction,
          responseMimeType: 'application/json',
          responseSchema,
          thinkingConfig: { thinkingLevel },
        },
      });
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
      await new Promise((r) => setTimeout(r, 700 * (attempt + 1)));
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

export async function runProblemWorkflow(
  problemId: string,
  executionId: string,
  options: {
    maxCorrectionAttempts?: number;
    clarificationResponse?: string | null;
    humanDecision?: HumanReviewDecision | null;
  } = {},
): Promise<void> {
  const db = getDatabase();
  const problem = db.getProblem(problemId);
  if (!problem) {
    throw new Error(`Problem ${problemId} not found`);
  }

  const maxAttempts = options.maxCorrectionAttempts ?? 3;
  const startedAt = new Date().toISOString();

  if (!options.clarificationResponse && !options.humanDecision) {
    db.clearExecutionArtifacts(problemId);
    db.updateProblem(problemId, {
      execution_id: executionId,
      status: 'RUNNING',
      current_stage: 'UNDERSTANDING',
      correction_attempts: 0,
      max_correction_attempts: maxAttempts,
    });
  } else {
    db.updateProblem(problemId, {
      status: 'RUNNING',
      current_stage: 'RESUMING',
    });
  }

  const toolCatalogText = JSON.stringify(TOOL_CATALOG, null, 2);
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
      synthesis = SolverSynthesisSchema.parse(lastSolverStep?.structured_output ?? {
        solution_methodology: latest.plan?.methodology ?? 'Human-reviewed resolution',
        final_answer: options.humanDecision.override_answer ?? 'Unresolved',
        units: 'dimensionless',
      });
      if (options.humanDecision.reviewer_action === 'override' && options.humanDecision.override_answer) {
        synthesis.final_answer = options.humanDecision.override_answer;
      }
    } else {
      // AGENT 1: Problem Understanding
      db.updateProblem(problemId, { current_stage: 'UNDERSTANDING' });
      const uStart = new Date().toISOString();
      const uT0 = performance.now();
      const uPrompt = [
        `Problem:\n${problem.question}`,
        problem.domain_hint ? `User domain hint: ${problem.domain_hint}` : '',
        options.clarificationResponse
          ? `User clarification:\n${options.clarificationResponse}`
          : '',
      ]
        .filter(Boolean)
        .join('\n\n');

      const uResp = await callGeminiStructured(
        `You are Agent 1 (Problem Understanding Agent) in VeriSTEM.
Classify the domain, extract variables with units, equations, constraints, and assumptions.
Only set needs_clarification=true if essential numerical inputs are genuinely missing so the problem has no unique answer.`,
        uPrompt,
        SPEC_SCHEMA,
        ProblemSpecificationSchema,
      );
      spec = uResp.data as ProblemSpecification;
      const uAgentRes: AgentExecutionResult = {
        agent_name: 'ProblemUnderstandingAgent',
        stage: 'UNDERSTANDING',
        status: spec.needs_clarification ? 'NEEDS_CLARIFICATION' : 'COMPLETED',
        summary: `Classified as ${spec.domain} (${spec.problem_type}) with ${spec.variables.length} variables.`,
        structured_output: spec as unknown as Record<string, unknown>,
        tool_records: [],
        tokens_input: uResp.tokensInput,
        tokens_output: uResp.tokensOutput,
        duration_ms: Number((performance.now() - uT0).toFixed(2)),
        started_at: uStart,
        completed_at: new Date().toISOString(),
      };
      db.addAgentResult(problemId, executionId, uAgentRes);
      db.updateProblem(problemId, { specification: spec });

      if (spec.needs_clarification) {
        db.updateProblem(problemId, {
          status: 'NEEDS_CLARIFICATION',
          current_stage: 'NEEDS_CLARIFICATION',
        });
        return;
      }

      // AGENT 2: Planning
      db.updateProblem(problemId, { current_stage: 'PLANNING' });
      const pStart = new Date().toISOString();
      const pT0 = performance.now();
      const pResp = await callGeminiStructured(
        `You are Agent 2 (Planning Agent) in VeriSTEM.
Decompose the problem into dependency-aware tasks and select computational tools strictly from the catalog.
Define an independent verification_strategy using a separate mathematical/numerical method.`,
        `ProblemSpecification:\n${JSON.stringify(spec, null, 2)}\n\nAvailable Tools:\n${toolCatalogText}`,
        PLAN_SCHEMA,
        ExecutionPlanSchema,
      );
      plan = pResp.data as ExecutionPlan;
      db.addAgentResult(problemId, executionId, {
        agent_name: 'PlanningAgent',
        stage: 'PLANNING',
        status: 'COMPLETED',
        summary: `Constructed ${plan.tasks.length}-step plan using ${plan.selected_tools.join(', ')}.`,
        structured_output: plan as unknown as Record<string, unknown>,
        tool_records: [],
        tokens_input: pResp.tokensInput,
        tokens_output: pResp.tokensOutput,
        duration_ms: Number((performance.now() - pT0).toFixed(2)),
        started_at: pStart,
        completed_at: new Date().toISOString(),
      });
      db.updateProblem(problemId, { plan });

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
Never guess or invent numbers. Generate tool_calls using python_sandbox, symbolic_math, or dimensional_analysis.
For python_sandbox, write clean Python 3 using math/cmath/decimal/fractions/statistics and assign the final answer or dict to RESULT.
Ensure arguments_json is a valid JSON string matching the tool parameters.`,
          `ProblemSpecification:\n${JSON.stringify(spec, null, 2)}\n\nExecutionPlan:\n${JSON.stringify(plan, null, 2)}\n\nTools:\n${toolCatalogText}${correctionPrompt}`,
          TOOL_CALL_PLAN_SCHEMA,
          ToolCallPlanSchema,
        );

        const solverRecords: ToolExecutionRecord[] = [];
        for (const call of sPlanResp.data.tool_calls) {
          try {
            const parsedArgs = JSON.parse(call.arguments_json) as Record<string, unknown>;
            solverRecords.push(executeTool(call.tool_name, parsedArgs));
          } catch (err) {
            solverRecords.push({
              tool_call_id: `tc_err_${call.task_id}`,
              tool_name: call.tool_name,
              input_payload: { raw: call.arguments_json },
              output_payload: {},
              success: false,
              error_message: err instanceof Error ? err.message : String(err),
              duration_ms: 0,
              executed_at: new Date().toISOString(),
            });
          }
        }
        allToolRecords = [...allToolRecords, ...solverRecords];

        const sSynthResp = await callGeminiStructured(
          `You are Agent 3 (STEM Solver Agent) synthesizing ONLY actual executed tool outputs.
Never invent numeric values. Report the exact result computed by the tools.`,
          `Specification:\n${JSON.stringify(spec, null, 2)}\n\nStrategy:\n${sPlanResp.data.reasoning_strategy}\n\nActual Tool Results:\n${JSON.stringify(solverRecords, null, 2)}`,
          SOLVER_SYNTHESIS_SCHEMA,
          SolverSynthesisSchema,
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
Use independent_verifier (integral_quadrature, derivative_finite_diff, root_residual, black_scholes_mc, numeric_tolerance), dimensional_analysis, symbolic_math, or python_sandbox.
For python_sandbox, use ONLY Python stdlib modules (math, cmath, decimal, fractions, statistics, json, itertools — never numpy/scipy) and set RESULT = {"passed": True/False, "independent_value": ..., "details": "..."}.`,
          `Specification:\n${JSON.stringify(spec, null, 2)}\n\nCandidate Solution:\n${JSON.stringify(synthesis, null, 2)}\n\nTool Catalog:\n${toolCatalogText}`,
          VERIFICATION_PLAN_SCHEMA,
          VerificationPlanSchema,
        );

        const verifRecords: ToolExecutionRecord[] = [];
        for (const call of vPlanResp.data.verification_calls) {
          try {
            const parsedArgs = JSON.parse(call.arguments_json) as Record<string, unknown>;
            verifRecords.push(executeTool(call.tool_name, parsedArgs));
          } catch (err) {
            verifRecords.push({
              tool_call_id: `tc_verr_${call.task_id}`,
              tool_name: call.tool_name,
              input_payload: { raw: call.arguments_json },
              output_payload: {},
              success: false,
              error_message: err instanceof Error ? err.message : String(err),
              duration_ms: 0,
              executed_at: new Date().toISOString(),
            });
          }
        }
        allToolRecords = [...allToolRecords, ...verifRecords];

        const vJudgeResp = await callGeminiStructured(
          `You are Agent 4 (Independent Verification Agent) issuing a formal VerificationReport.
Base your verdict STRICTLY on the actual verification tool results:
- If zero verification tools succeeded, status MUST be INCONCLUSIVE or FAIL (never PASS).
- If any check failed, status MUST be FAIL.
- Only emit PASS when independent tool execution confirmed the solution.`,
          `Specification:\n${JSON.stringify(spec, null, 2)}\n\nCandidate Solution:\n${JSON.stringify(synthesis, null, 2)}\n\nActual Verification Tool Outputs:\n${JSON.stringify(verifRecords, null, 2)}`,
          VERIFICATION_REPORT_SCHEMA,
          VerificationReportSchema,
        );

        verification = {
          ...vJudgeResp.data,
          verified_at: new Date().toISOString(),
        };

        // Deterministic verification safeguards
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
          tokens_input: vPlanResp.tokensInput + vJudgeResp.tokensInput,
          tokens_output: vPlanResp.tokensOutput + vJudgeResp.tokensOutput,
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
          `Attempt ${attempt}\nSpecification:\n${JSON.stringify(spec, null, 2)}\nCandidate:\n${JSON.stringify(synthesis, null, 2)}\nVerificationReport:\n${JSON.stringify(verification, null, 2)}`,
          CORRECTION_SCHEMA,
          CorrectionRequestSchema,
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

    // AGENT 6: Explanation & Reporting
    db.updateProblem(problemId, { current_stage: 'REPORTING' });
    const rStart = new Date().toISOString();
    const rT0 = performance.now();
    const rResp = await callGeminiStructured(
      `You are Agent 6 (Explanation and Reporting Agent) in VeriSTEM.
Write a clear, structured scientific explanation in Markdown using LaTeX ($...$ and $$...$$).
Strictly separate assumptions from verified computational evidence. Never present LLM confidence as mathematical proof.`,
      `Question:\n${problem.question}\n\nSpecification:\n${JSON.stringify(spec, null, 2)}\n\nSolution:\n${JSON.stringify(synthesis, null, 2)}\n\nVerification:\n${JSON.stringify(verification, null, 2)}`,
      NARRATIVE_SCHEMA,
      NarrativeSchema,
    );

    const finalStatus: WorkflowStatus =
      options.humanDecision?.reviewer_action === 'reject' ? 'UNRESOLVED' : 'COMPLETED';

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
      confidence_limitations: rResp.data.confidence_limitations,
      explanation_markdown: rResp.data.explanation_markdown,
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
      tokens_input: rResp.tokensInput,
      tokens_output: rResp.tokensOutput,
      duration_ms: Number((performance.now() - rT0).toFixed(2)),
      started_at: rStart,
      completed_at: new Date().toISOString(),
    });

    db.updateProblem(problemId, {
      status: finalStatus,
      current_stage: finalStatus,
      final_report: finalReport,
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    db.updateProblem(problemId, {
      status: 'FAILED',
      current_stage: `FAILED: ${msg.slice(0, 140)}`,
    });
  }
}
