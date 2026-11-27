export type ProblemDomain =
  | 'mathematics'
  | 'physics'
  | 'quantitative_finance'
  | 'general_stem';

export type VerificationStatus = 'PASS' | 'FAIL' | 'INCONCLUSIVE';

export type WorkflowStatus =
  | 'PENDING'
  | 'RUNNING'
  | 'NEEDS_CLARIFICATION'
  | 'HUMAN_REVIEW'
  | 'COMPLETED'
  | 'UNRESOLVED'
  | 'FAILED';

export interface MathematicalVariable {
  symbol: string;
  name: string;
  value: number | string | null;
  unit: string;
  constraints: string[];
  description: string;
}

export interface ProblemSpecification {
  domain: ProblemDomain;
  problem_type: string;
  objective: string;
  variables: MathematicalVariable[];
  equations: string[];
  constraints: string[];
  assumptions: string[];
  missing_information: string[];
  needs_clarification: boolean;
  clarification_question: string | null;
}

export interface ExecutionTask {
  task_id: string;
  title: string;
  description: string;
  method: string;
  tool_name: string;
  depends_on: string[];
  expected_output: string;
}

export interface ExecutionPlan {
  methodology: string;
  selected_tools: string[];
  tasks: ExecutionTask[];
  verification_strategy: string;
}

export interface ToolExecutionRecord {
  tool_call_id: string;
  tool_name: string;
  input_payload: Record<string, unknown>;
  output_payload: Record<string, unknown>;
  success: boolean;
  error_message: string | null;
  duration_ms: number;
  executed_at: string;
}

export interface AgentExecutionResult {
  agent_name: string;
  stage: string;
  status: string;
  summary: string;
  structured_output: Record<string, unknown>;
  tool_records: ToolExecutionRecord[];
  tokens_input: number;
  tokens_output: number;
  duration_ms: number;
  started_at: string;
  completed_at: string;
}

export interface VerificationCheck {
  check_name: string;
  method: string;
  passed: boolean;
  expected_value: string | null;
  observed_value: string | null;
  relative_error: number | null;
  tolerance: number | null;
  details: string;
}

export interface VerificationReport {
  status: VerificationStatus;
  checks: VerificationCheck[];
  independent_recalculation: string | null;
  dimensional_consistency_passed: boolean | null;
  boundary_conditions_passed: boolean | null;
  numerical_stability_passed: boolean | null;
  failure_reasons: string[];
  evidence_summary: string;
  verified_at: string;
}

export interface CorrectionRequest {
  attempt_number: number;
  root_cause: string;
  error_category: string;
  corrective_instructions: string;
  tools_to_reexecute: string[];
  created_at: string;
}

export interface HumanReviewDecision {
  reviewer_action: 'approve' | 'clarify' | 'override' | 'reject';
  notes: string;
  clarification_response: string | null;
  override_answer: string | null;
  decided_at: string;
}

export interface FinalSolutionReport {
  problem_id: string;
  execution_id: string;
  original_question: string;
  detected_domain: ProblemDomain;
  extracted_variables: MathematicalVariable[];
  assumptions: string[];
  solution_methodology: string;
  equations: string[];
  intermediate_results: Array<Record<string, unknown>>;
  final_answer: string;
  numeric_value: number | null;
  units: string;
  verification_status: VerificationStatus;
  verification_evidence: VerificationCheck[];
  correction_attempts: number;
  correction_history: CorrectionRequest[];
  tool_execution_history: ToolExecutionRecord[];
  confidence_limitations: string[];
  explanation_markdown: string;
  started_at: string;
  completed_at: string;
}

export interface ProblemRecord {
  problem_id: string;
  execution_id: string | null;
  question: string;
  domain_hint: ProblemDomain | null;
  status: WorkflowStatus;
  current_stage: string;
  correction_attempts: number;
  specification: ProblemSpecification | null;
  plan: ExecutionPlan | null;
  agent_timeline: AgentExecutionResult[];
  tool_records: ToolExecutionRecord[];
  verification_report: VerificationReport | null;
  correction_history: CorrectionRequest[];
  human_review: HumanReviewDecision | null;
  final_report: FinalSolutionReport | null;
  created_at: string;
  updated_at: string;
}

export interface MetricsSummary {
  total_problems: number;
  completed_problems: number;
  passed_verifications: number;
  failed_verifications: number;
  unresolved_problems: number;
  average_correction_attempts: number;
  average_duration_ms: number;
  total_tool_executions: number;
}

export interface BenchmarkCase {
  id: string;
  domain: ProblemDomain;
  category: string;
  difficulty: 'standard' | 'advanced' | 'frontier';
  question: string;
  expected_answer: string;
  expected_numeric: number | null;
  expected_unit: string;
  tolerance: number;
  verification_method: string;
}
