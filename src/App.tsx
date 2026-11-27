import { useCallback, useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Play,
  RotateCcw,
  XCircle,
} from 'lucide-react';
import { BenchmarkExplorer } from './components/BenchmarkExplorer';
import { ReportView } from './components/ReportView';
import { ToolEvidenceList } from './components/ToolEvidenceList';
import { VerificationPanel } from './components/VerificationPanel';
import { WorkflowGraph } from './components/WorkflowGraph';
import type {
  BenchmarkCase,
  MetricsSummary,
  ProblemDomain,
  ProblemRecord,
} from './types/veristem';

type ActiveTab = 'workbench' | 'workflow' | 'benchmarks' | 'history';

const CURATED_PRESETS: Array<{ label: string; domain: ProblemDomain; question: string }> = [
  {
    label: 'Definite Integral & Quadrature',
    domain: 'mathematics',
    question: 'Evaluate the definite integral of x^2 * exp(-x) from x = 0 to x = 2.',
  },
  {
    label: 'Projectile Kinematics (SI Units)',
    domain: 'physics',
    question:
      'A projectile is launched from ground level at 45 m/s at 35 degrees above horizontal (g = 9.81 m/s^2). Compute the horizontal range in meters.',
  },
  {
    label: 'Black-Scholes Option & Monte Carlo',
    domain: 'quantitative_finance',
    question:
      'Compute the Black-Scholes European call price for S=100, K=105, r=0.05, sigma=0.20, T=1.0 year, and verify against Monte Carlo simulation.',
  },
  {
    label: 'Linear System & Residual Norm',
    domain: 'mathematics',
    question: 'Solve the linear system 4*x + y = 9 and 2*x + 3*y = 8 for x and y, and verify the residual norm.',
  },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('workbench');
  const [question, setQuestion] = useState(CURATED_PRESETS[0].question);
  const [domainHint, setDomainHint] = useState<'auto' | ProblemDomain>('auto');
  const [maxAttempts, setMaxAttempts] = useState(3);
  const [forceMultiAgent, setForceMultiAgent] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [activeProblem, setActiveProblem] = useState<ProblemRecord | null>(null);
  const [history, setHistory] = useState<ProblemRecord[]>([]);
  const [metrics, setMetrics] = useState<MetricsSummary | null>(null);
  const [benchmarks, setBenchmarks] = useState<BenchmarkCase[]>([]);
  const [expandedAgents, setExpandedAgents] = useState<Record<number, boolean>>({});

  // Resume / clarification state
  const [resumeAction, setResumeAction] = useState<'clarify' | 'approve' | 'override' | 'reject'>('clarify');
  const [resumeText, setResumeText] = useState('');
  const [overrideAnswer, setOverrideAnswer] = useState('');
  const [resuming, setResuming] = useState(false);

  const composerRef = useRef<HTMLTextAreaElement | null>(null);
  const sessionTokenRef = useRef<string | null>(null);

  const ensureSessionToken = useCallback(async (forceRefresh = false): Promise<string> => {
    if (sessionTokenRef.current && !forceRefresh) {
      return sessionTokenRef.current;
    }
    const res = await fetch('/api/web/session', { method: 'POST' });
    if (!res.ok) {
      throw new Error('Could not establish workbench session');
    }
    const data = (await res.json()) as { token: string };
    sessionTokenRef.current = data.token;
    return data.token;
  }, []);

  const workbenchFetch = useCallback(
    async (path: string, init: RequestInit = {}): Promise<Response> => {
      const token = await ensureSessionToken(false);
      const headers = new Headers(init.headers ?? {});
      headers.set('Authorization', `Bearer ${token}`);
      headers.set('X-Requested-With', 'VeriSTEM-Workbench');
      let res = await fetch(path, { ...init, headers });
      if (res.status === 401) {
        const freshToken = await ensureSessionToken(true);
        headers.set('Authorization', `Bearer ${freshToken}`);
        res = await fetch(path, { ...init, headers });
      }
      return res;
    },
    [ensureSessionToken],
  );

  const refreshHistoryAndMetrics = useCallback(async () => {
    try {
      const [histRes, metRes] = await Promise.all([
        workbenchFetch('/api/web/problems'),
        workbenchFetch('/api/web/metrics'),
      ]);
      if (histRes.ok) {
        const list = (await histRes.json()) as ProblemRecord[];
        setHistory(list);
      }
      if (metRes.ok) {
        const met = (await metRes.json()) as MetricsSummary;
        setMetrics(met);
      }
    } catch {
      // Transient fetch failure handled silently during background refresh
    }
  }, [workbenchFetch]);

  useEffect(() => {
    void refreshHistoryAndMetrics();
    fetch('/api/v1/benchmarks')
      .then((r) => (r.ok ? r.json() : []))
      .then((data: BenchmarkCase[]) => setBenchmarks(data))
      .catch(() => {});
  }, [refreshHistoryAndMetrics]);

  // Poll active problem while workflow is running
  useEffect(() => {
    if (!activeProblem || activeProblem.status !== 'RUNNING') {
      return;
    }
    const timer = window.setInterval(async () => {
      try {
        const res = await workbenchFetch(`/api/web/problems/${activeProblem.problem_id}`);
        if (res.ok) {
          const updated = (await res.json()) as ProblemRecord;
          setActiveProblem(updated);
          if (updated.status !== 'RUNNING') {
            void refreshHistoryAndMetrics();
          }
        }
      } catch {
        // Retry on next interval
      }
    }, 1200);
    return () => window.clearInterval(timer);
  }, [activeProblem, refreshHistoryAndMetrics, workbenchFetch]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || submitting) return;
    setErrorMsg(null);
    setSubmitting(true);

    try {
      const createRes = await workbenchFetch('/api/web/problems', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: question.trim(),
          domain_hint: domainHint === 'auto' ? null : domainHint,
        }),
      });
      const createData = await createRes.json();
      if (!createRes.ok) {
        throw new Error(createData.message || 'Failed to create problem');
      }

      const problemId = createData.problem_id as string;
      const solveRes = await workbenchFetch(`/api/web/problems/${problemId}/solve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          max_correction_attempts: maxAttempts,
          run_async: true,
          force_multi_agent: forceMultiAgent,
        }),
      });
      if (!solveRes.ok) {
        const errData = await solveRes.json();
        throw new Error(errData.message || 'Failed to start solver workflow');
      }

      const detailRes = await workbenchFetch(`/api/web/problems/${problemId}`);
      if (detailRes.ok) {
        const detail = (await detailRes.json()) as ProblemRecord;
        setActiveProblem(detail);
        setActiveTab('workbench');
      }
      void refreshHistoryAndMetrics();
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Unexpected submission error');
    } finally {
      setSubmitting(false);
    }
  };

  const handleResume = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!activeProblem || resuming) return;
    setResuming(true);
    setErrorMsg(null);
    try {
      const actionToSend =
        activeProblem.status === 'NEEDS_CLARIFICATION' ? 'clarify' : resumeAction;
      const res = await workbenchFetch(`/api/web/problems/${activeProblem.problem_id}/resume`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: actionToSend,
          clarification_response: resumeText || null,
          notes: resumeText,
          override_answer: actionToSend === 'override' ? overrideAnswer || null : null,
        }),
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.message || 'Failed to resume workflow');
      }
      const detailRes = await workbenchFetch(`/api/web/problems/${activeProblem.problem_id}`);
      if (detailRes.ok) {
        setActiveProblem((await detailRes.json()) as ProblemRecord);
      }
      setResumeText('');
      setOverrideAnswer('');
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Failed to resume workflow');
    } finally {
      setResuming(false);
    }
  };

  const handleLoadFromBenchmark = (q: string, dom: ProblemDomain) => {
    setQuestion(q);
    setDomainHint(dom);
    setActiveTab('workbench');
    window.setTimeout(() => composerRef.current?.focus(), 60);
  };

  const handleNewProblem = () => {
    setQuestion('');
    setDomainHint('auto');
    setActiveProblem(null);
    setErrorMsg(null);
    setActiveTab('workbench');
    window.setTimeout(() => composerRef.current?.focus(), 60);
  };

  const selectProblemFromHistory = async (problemId: string) => {
    const res = await workbenchFetch(`/api/web/problems/${problemId}`);
    if (res.ok) {
      const detail = (await res.json()) as ProblemRecord;
      setActiveProblem(detail);
      setQuestion(detail.question);
      setActiveTab('workbench');
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
      {/* Top Bar Contract: 3 zones separated by gap-8 */}
      <header className="sticky top-0 z-30 bg-white border-b border-slate-200">
        <div className="max-w-[1440px] mx-auto px-6 h-16 flex items-center justify-between gap-8">
          <a
            href="#top"
            onClick={(e) => {
              e.preventDefault();
              setActiveTab('workbench');
            }}
            className="text-lg font-bold tracking-tight text-slate-900 whitespace-nowrap shrink-0"
          >
            VeriSTEM
          </a>

          <nav className="flex items-center gap-6 text-sm font-medium">
            {(
              [
                ['workbench', 'Workbench'],
                ['workflow', 'Workflow'],
                ['benchmarks', 'Benchmarks'],
                ['history', 'History'],
              ] as Array<[ActiveTab, string]>
            ).map(([tabKey, label]) => (
              <button
                key={tabKey}
                type="button"
                onClick={() => setActiveTab(tabKey)}
                className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 ${
                  activeTab === tabKey
                    ? 'text-blue-600 border-blue-600'
                    : 'text-slate-600 border-transparent hover:text-slate-900'
                }`}
              >
                {label}
              </button>
            ))}
          </nav>

          <div className="shrink-0">
            <button
              type="button"
              onClick={handleNewProblem}
              className="px-4 py-2 text-xs font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 transition-colors whitespace-nowrap shrink-0"
            >
              New Problem
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-[1440px] w-full mx-auto px-6 py-8">
        {activeTab === 'workbench' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* Left Column: Problem Input & Presets */}
            <section className="lg:col-span-4 bg-white border border-slate-200 rounded-lg p-5 space-y-5">
              <div>
                <h1 className="text-lg font-semibold text-slate-900" style={{ textWrap: 'balance' }}>
                  Verifiable STEM Problem Solver
                </h1>
                <p className="text-xs text-slate-600 mt-1 leading-relaxed">
                  Every numerical and symbolic claim is independently checked via SymPy-style math, dimensional analysis, and isolated Python 3.11 execution.
                </p>
              </div>

              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label htmlFor="problem-input" className="block text-xs font-medium text-slate-700 mb-1.5">
                    Problem Statement
                  </label>
                  <textarea
                    id="problem-input"
                    ref={composerRef}
                    rows={5}
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="Enter a mathematics, physics, or quantitative finance problem..."
                    className="w-full rounded-md border border-slate-200 bg-slate-50/50 px-3 py-2.5 text-sm text-slate-900 placeholder:text-slate-400 focus:border-blue-600 focus:bg-white focus:outline-none"
                  />
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label htmlFor="domain-select" className="block text-xs font-medium text-slate-700 mb-1.5">
                      Domain
                    </label>
                    <select
                      id="domain-select"
                      value={domainHint}
                      onChange={(e) => setDomainHint(e.target.value as 'auto' | ProblemDomain)}
                      className="w-full rounded-md border border-slate-200 bg-white px-2.5 py-2 text-xs text-slate-900 focus:border-blue-600 focus:outline-none"
                    >
                      <option value="auto">Auto-detect</option>
                      <option value="mathematics">Mathematics</option>
                      <option value="physics">Physics</option>
                      <option value="quantitative_finance">Quantitative Finance</option>
                      <option value="general_stem">General STEM</option>
                    </select>
                  </div>

                  <div>
                    <label htmlFor="retry-select" className="block text-xs font-medium text-slate-700 mb-1.5">
                      Max Critic Retries
                    </label>
                    <select
                      id="retry-select"
                      value={maxAttempts}
                      onChange={(e) => setMaxAttempts(Number(e.target.value))}
                      className="w-full rounded-md border border-slate-200 bg-white px-2.5 py-2 text-xs font-mono text-slate-900 focus:border-blue-600 focus:outline-none"
                    >
                      <option value={1}>1 attempt</option>
                      <option value={2}>2 attempts</option>
                      <option value={3}>3 attempts</option>
                    </select>
                  </div>
                </div>

                <label className="flex items-center gap-2 text-xs text-slate-600 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={forceMultiAgent}
                    onChange={(e) => setForceMultiAgent(e.target.checked)}
                    className="rounded border-slate-300 text-blue-600 focus:ring-blue-600"
                  />
                  <span>Force full multi-agent workflow (bypass fast path)</span>
                </label>

                {errorMsg && (
                  <div className="p-3 bg-rose-50 border border-rose-200 rounded-md text-xs text-rose-700">
                    {errorMsg}
                  </div>
                )}

                <button
                  type="submit"
                  disabled={submitting || !question.trim() || activeProblem?.status === 'RUNNING'}
                  className="w-full inline-flex items-center justify-center gap-2 px-4 py-2.5 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {submitting || activeProblem?.status === 'RUNNING' ? (
                    <>
                      <RotateCcw className="w-4 h-4 animate-spin" />
                      <span>Verifying Workflow...</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4" />
                      <span>Solve &amp; Verify</span>
                    </>
                  )}
                </button>
              </form>

              <div className="pt-4 border-t border-slate-200 space-y-2.5">
                <div className="text-xs font-medium text-slate-700">Validated Examples</div>
                <div className="space-y-1.5">
                  {CURATED_PRESETS.map((preset) => (
                    <button
                      key={preset.label}
                      type="button"
                      onClick={() => {
                        setQuestion(preset.question);
                        setDomainHint(preset.domain);
                      }}
                      className="w-full text-left px-3 py-2 rounded-md border border-slate-200 hover:border-blue-600 hover:bg-slate-50 transition-colors"
                    >
                      <div className="text-xs font-medium text-slate-900">{preset.label}</div>
                      <div className="text-[11px] text-slate-500 mt-0.5 truncate">
                        {preset.domain.replace('_', ' ')} · {preset.question}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </section>

            {/* Right Column: Live Multi-Agent Execution & Report */}
            <section className="lg:col-span-8 space-y-6">
              <WorkflowGraph
                currentStage={activeProblem?.current_stage ?? 'IDLE'}
                workflowStatus={activeProblem?.status ?? null}
                verificationStatus={activeProblem?.verification_report?.status ?? null}
                complexityLevel={activeProblem?.complexity_level ?? null}
                executionPath={activeProblem?.execution_path ?? null}
                elapsedMs={activeProblem?.elapsed_ms ?? null}
                correctionAttempts={activeProblem?.correction_attempts ?? 0}
                timeline={activeProblem?.agent_timeline ?? []}
              />

              {activeProblem && activeProblem.status === 'FAILED' && (
                <div className="bg-rose-50 border border-rose-200 rounded-lg p-4 flex items-start gap-3">
                  <XCircle className="w-5 h-5 text-rose-700 shrink-0 mt-0.5" />
                  <div className="text-sm text-rose-800">
                    <div className="font-semibold">Workflow Execution Failed</div>
                    <p className="mt-1 font-mono text-xs">{activeProblem.current_stage}</p>
                  </div>
                </div>
              )}

              {activeProblem &&
                (activeProblem.status === 'NEEDS_CLARIFICATION' ||
                  activeProblem.status === 'HUMAN_REVIEW') && (
                  <form
                    onSubmit={handleResume}
                    className="bg-amber-50 border border-amber-200 rounded-lg p-5 space-y-4"
                  >
                    <div className="flex items-start gap-3">
                      <AlertTriangle className="w-5 h-5 text-amber-700 shrink-0 mt-0.5" />
                      <div>
                        <h3 className="text-sm font-semibold text-amber-900">
                          {activeProblem.status === 'NEEDS_CLARIFICATION'
                            ? 'Clarification Required Before Solving'
                            : 'Human Review Escalation (Max Correction Attempts Reached)'}
                        </h3>
                        <p className="text-xs text-amber-800 mt-1">
                          {activeProblem.specification?.clarification_question ||
                            'Independent verification could not pass automatically. Provide clarification, approve, override, or mark unresolved.'}
                        </p>
                      </div>
                    </div>

                    {activeProblem.status === 'HUMAN_REVIEW' && (
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                        <div>
                          <label className="block text-xs font-medium text-amber-900 mb-1">
                            Reviewer Decision
                          </label>
                          <select
                            value={resumeAction}
                            onChange={(e) =>
                              setResumeAction(
                                e.target.value as 'clarify' | 'approve' | 'override' | 'reject',
                              )
                            }
                            className="w-full rounded-md border border-amber-300 bg-white px-2.5 py-2 text-xs text-slate-900"
                          >
                            <option value="clarify">Clarify and Re-run</option>
                            <option value="approve">Approve Current Solution</option>
                            <option value="override">Override Final Answer</option>
                            <option value="reject">Mark Unresolved</option>
                          </select>
                        </div>
                        {resumeAction === 'override' && (
                          <div>
                            <label className="block text-xs font-medium text-amber-900 mb-1">
                              Override Answer
                            </label>
                            <input
                              type="text"
                              value={overrideAnswer}
                              onChange={(e) => setOverrideAnswer(e.target.value)}
                              placeholder="Enter verified value..."
                              className="w-full rounded-md border border-amber-300 bg-white px-2.5 py-2 text-xs text-slate-900"
                            />
                          </div>
                        )}
                      </div>
                    )}

                    <div>
                      <textarea
                        rows={2}
                        value={resumeText}
                        onChange={(e) => setResumeText(e.target.value)}
                        placeholder="Provide missing parameters or review notes..."
                        className="w-full rounded-md border border-amber-300 bg-white px-3 py-2 text-xs text-slate-900"
                      />
                    </div>

                    <button
                      type="submit"
                      disabled={resuming}
                      className="px-4 py-2 text-xs font-medium text-white bg-amber-700 rounded-md hover:bg-amber-800 disabled:opacity-50"
                    >
                      {resuming ? 'Resuming...' : 'Resume Workflow'}
                    </button>
                  </form>
                )}

              <ReportView report={activeProblem?.final_report ?? null} />

              <VerificationPanel
                verification={activeProblem?.verification_report ?? null}
                corrections={activeProblem?.correction_history ?? []}
              />

              {/* Agent Timeline Accordion */}
              <div className="bg-white border border-slate-200 rounded-lg p-5">
                <div className="flex items-center justify-between pb-3 border-b border-slate-200">
                  <h3 className="text-base font-semibold text-slate-900">Agent Execution Timeline</h3>
                  <span className="text-xs font-mono tabular-nums text-slate-500">
                    {activeProblem?.agent_timeline.length ?? 0} step(s)
                  </span>
                </div>

                {!activeProblem || activeProblem.agent_timeline.length === 0 ? (
                  <p className="text-sm text-slate-500 mt-3">
                    Submit a problem to inspect each agent’s structured JSON contract, duration, and token consumption.
                  </p>
                ) : (
                  <div className="divide-y divide-slate-100 mt-2">
                    {activeProblem.agent_timeline.map((step, idx) => {
                      const isOpen = Boolean(expandedAgents[idx]);
                      return (
                        <div key={`${step.agent_name}_${idx}`} className="py-2.5">
                          <button
                            type="button"
                            onClick={() =>
                              setExpandedAgents((prev) => ({ ...prev, [idx]: !prev[idx] }))
                            }
                            className="w-full flex items-center justify-between gap-3 text-left py-1 px-1.5 rounded hover:bg-slate-50 transition-colors"
                          >
                            <div className="flex items-center gap-2.5 min-w-0">
                              {isOpen ? (
                                <ChevronDown className="w-4 h-4 text-slate-400 shrink-0" />
                              ) : (
                                <ChevronRight className="w-4 h-4 text-slate-400 shrink-0" />
                              )}
                              <span className="text-xs font-semibold text-slate-900">
                                {step.agent_name}
                              </span>
                              <span className="text-xs text-slate-500 truncate">{step.summary}</span>
                            </div>
                            <div className="flex items-center gap-3 text-xs font-mono tabular-nums text-slate-500 shrink-0">
                              <span>{step.tokens_input + step.tokens_output} tok</span>
                              <span>{step.duration_ms.toFixed(0)} ms</span>
                            </div>
                          </button>
                          {isOpen && (
                            <pre className="mt-2 ml-6 p-3 bg-slate-50 border border-slate-200 rounded font-mono text-[11px] text-slate-800 overflow-x-auto">
                              {JSON.stringify(step.structured_output, null, 2)}
                            </pre>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              <ToolEvidenceList records={activeProblem?.tool_records ?? []} />
            </section>
          </div>
        )}

        {activeTab === 'workflow' && (
          <div className="space-y-6">
            <div>
              <h2 className="text-xl font-semibold text-slate-900" style={{ textWrap: 'balance' }}>
                Multi-Agent Architecture &amp; Computational Sandbox
              </h2>
              <p className="text-sm text-slate-600 mt-1">
                Inspection of the active problem’s execution plan, variable extraction, and deterministic tool contracts.
              </p>
            </div>

            <WorkflowGraph
              currentStage={activeProblem?.current_stage ?? 'IDLE'}
              workflowStatus={activeProblem?.status ?? null}
              verificationStatus={activeProblem?.verification_report?.status ?? null}
              complexityLevel={activeProblem?.complexity_level ?? null}
              executionPath={activeProblem?.execution_path ?? null}
              elapsedMs={activeProblem?.elapsed_ms ?? null}
              correctionAttempts={activeProblem?.correction_attempts ?? 0}
              timeline={activeProblem?.agent_timeline ?? []}
            />

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-3">
                <h3 className="text-base font-semibold text-slate-900">
                  Typed Problem Specification
                </h3>
                {activeProblem?.specification ? (
                  <pre className="p-3 bg-slate-50 border border-slate-200 rounded font-mono text-xs text-slate-800 overflow-x-auto">
                    {JSON.stringify(activeProblem.specification, null, 2)}
                  </pre>
                ) : (
                  <p className="text-sm text-slate-500">
                    No active problem specification loaded. Run a problem in Workbench or pick one from History.
                  </p>
                )}
              </div>

              <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-3">
                <h3 className="text-base font-semibold text-slate-900">
                  Dependency-Aware Execution Plan
                </h3>
                {activeProblem?.plan ? (
                  <div className="space-y-3">
                    <p className="text-xs text-slate-700">{activeProblem.plan.methodology}</p>
                    <div className="divide-y divide-slate-100 border-t border-slate-100">
                      {activeProblem.plan.tasks.map((t) => (
                        <div key={t.task_id} className="py-2.5 text-xs">
                          <div className="flex items-center justify-between font-medium text-slate-900">
                            <span>
                              {t.task_id} · {t.title}
                            </span>
                            <span className="font-mono text-blue-600">{t.tool_name}</span>
                          </div>
                          <p className="text-slate-600 mt-1">{t.description}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <p className="text-sm text-slate-500">
                    No execution plan available yet.
                  </p>
                )}
              </div>
            </div>

            <ToolEvidenceList records={activeProblem?.tool_records ?? []} />
          </div>
        )}

        {activeTab === 'benchmarks' && (
          <BenchmarkExplorer benchmarks={benchmarks} onLoadProblem={handleLoadFromBenchmark} />
        )}

        {activeTab === 'history' && (
          <div className="space-y-6">
            <div>
              <h2 className="text-xl font-semibold text-slate-900" style={{ textWrap: 'balance' }}>
                Persisted Execution History &amp; Telemetry
              </h2>
              <p className="text-sm text-slate-600 mt-1">
                All problem specifications, verification verdicts, tool records, and final reports stored in SQLite.
              </p>
            </div>

            {metrics && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-white border border-slate-200 rounded-lg p-4">
                  <div className="text-xs text-slate-500">Total Problems</div>
                  <div className="text-2xl font-mono font-semibold tabular-nums text-slate-900 mt-1">
                    {metrics.total_problems}
                  </div>
                </div>
                <div className="bg-white border border-slate-200 rounded-lg p-4">
                  <div className="text-xs text-slate-500">Verified PASS</div>
                  <div className="text-2xl font-mono font-semibold tabular-nums text-emerald-700 mt-1">
                    {metrics.passed_verifications}
                  </div>
                </div>
                <div className="bg-white border border-slate-200 rounded-lg p-4">
                  <div className="text-xs text-slate-500">Avg. Solve Time</div>
                  <div className="text-2xl font-mono font-semibold tabular-nums text-slate-900 mt-1">
                    {metrics.average_duration_ms >= 1000
                      ? `${(metrics.average_duration_ms / 1000).toFixed(2)} s`
                      : `${metrics.average_duration_ms.toFixed(1)} ms`}
                  </div>
                </div>
                <div className="bg-white border border-slate-200 rounded-lg p-4">
                  <div className="text-xs text-slate-500">Avg. Correction Loops</div>
                  <div className="text-2xl font-mono font-semibold tabular-nums text-slate-900 mt-1">
                    {metrics.average_correction_attempts.toFixed(2)}
                  </div>
                </div>
              </div>
            )}

            <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
              {history.length === 0 ? (
                <div className="p-8 text-center space-y-3">
                  <p className="text-sm text-slate-600">
                    No persisted problem executions yet. Solve a problem in the Workbench to record verified outputs.
                  </p>
                  <button
                    type="button"
                    onClick={() => setActiveTab('workbench')}
                    className="px-4 py-2 text-xs font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700"
                  >
                    Open Workbench
                  </button>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-slate-200 bg-slate-50 text-slate-500">
                        <th className="py-3 px-4 font-medium">Problem ID</th>
                        <th className="py-3 px-4 font-medium">Question</th>
                        <th className="py-3 px-4 font-medium">Path</th>
                        <th className="py-3 px-4 font-medium">Status</th>
                        <th className="py-3 px-4 font-medium">Verification</th>
                        <th className="py-3 px-4 font-medium text-right">Elapsed</th>
                        <th className="py-3 px-4 font-medium text-right">Action</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {history.map((item) => {
                        const vStatus = item.verification_report?.status;
                        return (
                          <tr key={item.problem_id} className="hover:bg-slate-50/70">
                            <td className="py-3 px-4 font-mono font-medium text-slate-900 whitespace-nowrap">
                              {item.problem_id}
                            </td>
                            <td className="py-3 px-4 text-slate-700 max-w-md truncate">
                              {item.question}
                            </td>
                            <td className="py-3 px-4 font-mono text-slate-700 whitespace-nowrap">
                              {item.execution_path ?? 'MULTI_AGENT_WORKFLOW'}
                            </td>
                            <td className="py-3 px-4 font-mono text-slate-700 whitespace-nowrap">
                              {item.status}
                            </td>
                            <td className="py-3 px-4 font-medium whitespace-nowrap">
                              {vStatus === 'PASS' ? (
                                <span className="inline-flex items-center gap-1 text-emerald-700">
                                  <CheckCircle2 className="w-3.5 h-3.5" />
                                  PASS
                                </span>
                              ) : vStatus === 'FAIL' ? (
                                <span className="inline-flex items-center gap-1 text-rose-700">
                                  <XCircle className="w-3.5 h-3.5" />
                                  FAIL
                                </span>
                              ) : (
                                <span className="text-slate-400">{vStatus ?? '—'}</span>
                              )}
                            </td>
                            <td className="py-3 px-4 font-mono tabular-nums text-right text-slate-700 whitespace-nowrap">
                              {item.elapsed_ms !== null && item.elapsed_ms !== undefined
                                ? item.elapsed_ms >= 1000
                                  ? `${(item.elapsed_ms / 1000).toFixed(2)} s`
                                  : `${item.elapsed_ms.toFixed(1)} ms`
                                : '—'}
                            </td>
                            <td className="py-3 px-4 text-right whitespace-nowrap">
                              <button
                                type="button"
                                onClick={() => selectProblemFromHistory(item.problem_id)}
                                className="text-xs font-medium text-blue-600 hover:text-blue-700"
                              >
                                Inspect
                              </button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
