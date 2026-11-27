import { AlertTriangle, CheckCircle2, Clock, RotateCcw, XCircle } from 'lucide-react';
import type {
  AgentExecutionResult,
  ComplexityLevel,
  ExecutionPath,
  VerificationStatus,
  WorkflowStatus,
} from '../types/veristem';

interface WorkflowGraphProps {
  currentStage: string;
  workflowStatus: WorkflowStatus | null;
  verificationStatus: VerificationStatus | null;
  complexityLevel?: ComplexityLevel | null;
  executionPath?: ExecutionPath | null;
  elapsedMs?: number | null;
  correctionAttempts: number;
  timeline: AgentExecutionResult[];
}

const STAGES = [
  { key: 'UNDERSTANDING', label: '01. Understanding', agent: 'ProblemUnderstandingAgent' },
  { key: 'PLANNING', label: '02. Planning', agent: 'PlanningAgent' },
  { key: 'SOLVING', label: '03. STEM Solver', agent: 'STEMSolverAgent', altAgent: 'FastSymbolicSolver' },
  { key: 'VERIFICATION', label: '04. Independent Verifier', agent: 'IndependentVerificationAgent' },
  { key: 'CRITIC', label: '05. Critic Loop', agent: 'CriticSelfCorrectionAgent' },
  { key: 'REPORTING', label: '06. Report Synthesis', agent: 'ExplanationReportingAgent' },
];

function formatElapsed(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return '—';
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(1)} ms`;
}

export function WorkflowGraph({
  currentStage,
  workflowStatus,
  verificationStatus,
  complexityLevel,
  executionPath,
  elapsedMs,
  correctionAttempts,
  timeline,
}: WorkflowGraphProps) {
  const latestByAgent = new Map<string, AgentExecutionResult>();
  for (const item of timeline) {
    latestByAgent.set(item.agent_name, item);
  }

  const isFastPath = executionPath === 'FAST_PATH';

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-5">
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-200">
        <div>
          <h3 className="text-base font-semibold text-slate-900">
            Execution Pipeline &amp; Complexity Routing
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            {isFastPath
              ? 'Fast Path active: deterministic symbolic computation + independent Python 3.11 verification'
              : 'Bounded LangGraph multi-agent workflow (max 3 critic retries) with parallel tool execution'}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3 text-xs font-mono tabular-nums text-slate-600">
          {complexityLevel && (
            <>
              <span className="text-slate-900 font-semibold">{complexityLevel}</span>
              <span aria-hidden="true">·</span>
            </>
          )}
          {executionPath && (
            <>
              <span className={isFastPath ? 'text-blue-700 font-semibold' : 'text-slate-800 font-semibold'}>
                {executionPath}
              </span>
              <span aria-hidden="true">·</span>
            </>
          )}
          <span>Stage: {currentStage || 'IDLE'}</span>
          {elapsedMs !== null && elapsedMs !== undefined && (
            <>
              <span aria-hidden="true">·</span>
              <span className="text-slate-900 font-semibold">{formatElapsed(elapsedMs)}</span>
            </>
          )}
          {!isFastPath && (
            <>
              <span aria-hidden="true">·</span>
              <span>Retries: {correctionAttempts}/3</span>
            </>
          )}
          {verificationStatus && (
            <>
              <span aria-hidden="true">·</span>
              <span
                className={
                  verificationStatus === 'PASS'
                    ? 'text-emerald-700 font-semibold'
                    : verificationStatus === 'FAIL'
                      ? 'text-rose-700 font-semibold'
                      : 'text-amber-700 font-semibold'
                }
              >
                Verdict: {verificationStatus}
              </span>
            </>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-3 pt-4">
        {STAGES.map((stage) => {
          const exec = latestByAgent.get(stage.agent) ?? (stage.altAgent ? latestByAgent.get(stage.altAgent) : undefined);
          const isActive = currentStage === stage.key && workflowStatus === 'RUNNING';
          const isFastBypassed =
            isFastPath &&
            workflowStatus === 'COMPLETED' &&
            !exec &&
            (stage.key === 'UNDERSTANDING' || stage.key === 'PLANNING' || stage.key === 'CRITIC' || stage.key === 'REPORTING');
          const isCriticBypassed =
            !isFastPath &&
            stage.key === 'CRITIC' &&
            correctionAttempts === 0 &&
            workflowStatus === 'COMPLETED';

          let statusLabel = 'Pending';
          let statusTone = 'text-slate-400';
          let Icon = Clock;

          if (isActive) {
            statusLabel = 'Running';
            statusTone = 'text-blue-600';
            Icon = RotateCcw;
          } else if (exec) {
            if (exec.status === 'FAIL') {
              statusLabel = 'Failed';
              statusTone = 'text-rose-700';
              Icon = XCircle;
            } else if (exec.status === 'INCONCLUSIVE' || exec.status === 'NEEDS_CLARIFICATION') {
              statusLabel = exec.status === 'NEEDS_CLARIFICATION' ? 'Clarify' : 'Inconclusive';
              statusTone = 'text-amber-700';
              Icon = AlertTriangle;
            } else {
              statusLabel = 'Done';
              statusTone = 'text-emerald-700';
              Icon = CheckCircle2;
            }
          } else if (isFastBypassed) {
            statusLabel = stage.key === 'REPORTING' ? 'Direct' : 'Fast-bypass';
            statusTone = 'text-slate-400';
          } else if (isCriticBypassed) {
            statusLabel = 'Not needed';
            statusTone = 'text-slate-400';
          }

          return (
            <div
              key={stage.key}
              className={`p-3 rounded-md border transition-colors ${
                isActive
                  ? 'border-blue-600 bg-blue-50/40'
                  : exec
                    ? 'border-slate-200 bg-slate-50/60'
                    : 'border-slate-200 bg-white'
              }`}
            >
              <div className="text-xs font-semibold text-slate-900 truncate">{stage.label}</div>
              <div className={`flex items-center gap-1.5 text-xs font-medium mt-2 ${statusTone}`}>
                <Icon className={`w-3.5 h-3.5 shrink-0 ${isActive ? 'animate-spin' : ''}`} />
                <span>{statusLabel}</span>
              </div>
              <div className="text-[11px] font-mono tabular-nums text-slate-500 mt-1.5">
                {exec ? `${exec.duration_ms.toFixed(1)} ms` : '—'}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
