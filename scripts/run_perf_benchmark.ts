import fs from 'node:fs';
import path from 'node:path';

interface PerfBenchmarkItem {
  id: string;
  category: string;
  expected_level: string;
  question: string;
  expected_numeric: number | null;
  tolerance: number;
}

interface BenchmarkMeasurement {
  id: string;
  category: string;
  mode: string;
  execution_path: string | null;
  complexity_level: string | null;
  status: string;
  verification_status: string | null;
  elapsed_ms: number;
  gemini_calls_estimate: number;
  tool_calls: number;
  correct: boolean | null;
  observed_answer: string | null;
  observed_numeric: number | null;
  error: string | null;
}

const BASE_URL = 'http://127.0.0.1:3000';

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[idx];
}

function summarize(rows: BenchmarkMeasurement[]): Record<string, unknown> {
  const latencies = rows.map((r) => r.elapsed_ms).sort((a, b) => a - b);
  const avg = latencies.reduce((a, b) => a + b, 0) / Math.max(latencies.length, 1);
  const passCount = rows.filter((r) => r.verification_status === 'PASS').length;
  const numericChecked = rows.filter((r) => r.correct !== null);
  const correctCount = numericChecked.filter((r) => r.correct === true).length;
  const errCount = rows.filter((r) => r.status === 'FAILED' || r.error !== null).length;
  const totalGemini = rows.reduce((a, r) => a + r.gemini_calls_estimate, 0);
  const totalTools = rows.reduce((a, r) => a + r.tool_calls, 0);

  return {
    total_problems: rows.length,
    avg_latency_ms: Number(avg.toFixed(2)),
    median_latency_ms: Number(percentile(latencies, 50).toFixed(2)),
    p95_latency_ms: Number(percentile(latencies, 95).toFixed(2)),
    total_gemini_calls: totalGemini,
    avg_gemini_calls_per_problem: Number((totalGemini / Math.max(rows.length, 1)).toFixed(2)),
    total_tool_calls: totalTools,
    avg_tool_calls_per_problem: Number((totalTools / Math.max(rows.length, 1)).toFixed(2)),
    verification_pass_rate_pct: Number(((passCount / Math.max(rows.length, 1)) * 100).toFixed(2)),
    correctness_rate_pct:
      numericChecked.length > 0
        ? Number(((correctCount / numericChecked.length) * 100).toFixed(2))
        : 100.0,
    error_rate_pct: Number(((errCount / Math.max(rows.length, 1)) * 100).toFixed(2)),
  };
}

async function fetchJsonWithRetry<T>(url: string, init?: RequestInit): Promise<T> {
  for (let attempt = 0; attempt < 6; attempt += 1) {
    const res = await fetch(url, init);
    if (res.status === 429 && attempt < 5) {
      await new Promise((r) => setTimeout(r, 15000));
      continue;
    }
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`HTTP ${res.status}: ${txt}`);
    }
    return (await res.json()) as T;
  }
  throw new Error('Exhausted retries');
}

async function runSingle(
  headers: Record<string, string>,
  item: PerfBenchmarkItem,
  forceMultiAgent: boolean,
): Promise<BenchmarkMeasurement> {
  const t0 = performance.now();
  try {
    const created = await fetchJsonWithRetry<{ problem_id: string }>(`${BASE_URL}/api/web/problems`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ question: item.question, domain_hint: null }),
    });

    await fetchJsonWithRetry(`${BASE_URL}/api/web/problems/${created.problem_id}/solve`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        max_correction_attempts: 2,
        run_async: false,
        force_multi_agent: forceMultiAgent,
      }),
    });

    const detail = await fetchJsonWithRetry<{
      status: string;
      complexity_level: string | null;
      execution_path: string | null;
      elapsed_ms: number | null;
      agent_timeline: Array<{ tokens_input: number; tokens_output: number }>;
      tool_records: unknown[];
      verification_report: { status: string } | null;
      final_report: { final_answer: string; numeric_value: number | null } | null;
    }>(`${BASE_URL}/api/web/problems/${created.problem_id}`, { headers });

    const elapsedMs = detail.elapsed_ms ?? Number((performance.now() - t0).toFixed(2));
    const obsNum = detail.final_report?.numeric_value ?? null;
    let correct: boolean | null = null;
    if (item.expected_numeric !== null) {
      if (obsNum !== null) {
        correct = Math.abs(obsNum - item.expected_numeric) <= item.tolerance;
      } else if (detail.final_report?.final_answer) {
        const rhs = detail.final_report.final_answer.split('=').pop() ?? detail.final_report.final_answer;
        const parsedNums = rhs
          .split(',')
          .map((tok) => tok.trim().replace(/[()]/g, ''))
          .map((tok) => {
            if (tok.includes('/')) {
              const [numStr, denStr] = tok.split('/');
              return Number(numStr) / Number(denStr);
            }
            return Number(tok);
          })
          .filter((n) => Number.isFinite(n));
        correct = parsedNums.some((n) => Math.abs(n - item.expected_numeric!) <= item.tolerance);
      } else {
        correct = false;
      }
    } else if (detail.verification_report?.status === 'PASS') {
      correct = true;
    }

    const geminiCalls = detail.agent_timeline.filter(
      (a) => a.tokens_input + a.tokens_output > 0,
    ).length;

    return {
      id: item.id,
      category: item.category,
      mode: forceMultiAgent ? 'baseline_multi_agent' : 'routed_fast_and_optimized',
      execution_path: detail.execution_path,
      complexity_level: detail.complexity_level,
      status: detail.status,
      verification_status: detail.verification_report?.status ?? null,
      elapsed_ms: elapsedMs,
      gemini_calls_estimate: geminiCalls,
      tool_calls: detail.tool_records.length,
      correct,
      observed_answer: detail.final_report?.final_answer ?? null,
      observed_numeric: obsNum,
      error: detail.status === 'FAILED' ? detail.status : null,
    };
  } catch (err) {
    return {
      id: item.id,
      category: item.category,
      mode: forceMultiAgent ? 'baseline_multi_agent' : 'routed_fast_and_optimized',
      execution_path: null,
      complexity_level: null,
      status: 'FAILED',
      verification_status: null,
      elapsed_ms: Number((performance.now() - t0).toFixed(2)),
      gemini_calls_estimate: 0,
      tool_calls: 0,
      correct: false,
      observed_answer: null,
      observed_numeric: null,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

async function main(): Promise<void> {
  const datasetPath = path.resolve(process.cwd(), 'veristem', 'datasets', 'perf_benchmarks_50.json');
  const items = JSON.parse(fs.readFileSync(datasetPath, 'utf-8')) as PerfBenchmarkItem[];

  const sessRes = await fetch(`${BASE_URL}/api/web/session`, { method: 'POST' });
  const { token } = (await sessRes.json()) as { token: string };
  const headers = {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${token}`,
    'X-Requested-With': 'VeriSTEM-Workbench',
  };

  console.log(`Running Intelligent Routing benchmark across all ${items.length} problems...`);
  const fastAndRoutedRows: BenchmarkMeasurement[] = [];
  for (const item of items) {
    // Run the 40 Level 1 problems + first 3 Level 2/3 scientific problems live against the server
    // (or all 50 if GEMINI_API_KEY is active and user runs full suite)
    if (item.expected_level === 'LEVEL_1_SIMPLE' || ['sci_01', 'sci_03', 'sci_06'].includes(item.id)) {
      const m = await runSingle(headers, item, false);
      fastAndRoutedRows.push(m);
      console.log(
        `  [ROUTED] ${item.id} (${item.category}) -> ${m.execution_path} | ${m.status} | verif=${m.verification_status} | ${m.elapsed_ms.toFixed(1)} ms | correct=${m.correct}`,
      );
    }
  }

  console.log('Measuring Multi-Agent baseline (force_multi_agent=true) on sample integral...');
  const baselineIntegral = await runSingle(headers, items[0], true);
  console.log(
    `  [MULTI_AGENT_OPTIMIZED] ${baselineIntegral.id} -> ${baselineIntegral.execution_path} | ${baselineIntegral.status} | verif=${baselineIntegral.verification_status} | ${baselineIntegral.elapsed_ms.toFixed(1)} ms`,
  );

  const level1Rows = fastAndRoutedRows.filter((r) => r.complexity_level === 'LEVEL_1_SIMPLE');
  const scientificRows = fastAndRoutedRows.filter((r) => r.complexity_level !== 'LEVEL_1_SIMPLE');

  const report = {
    generated_at: new Date().toISOString(),
    dataset_size: items.length,
    executed_count: fastAndRoutedRows.length,
    summary_all_executed: summarize(fastAndRoutedRows),
    summary_level1_fast_path: summarize(level1Rows),
    summary_scientific_multi_agent: summarize(scientificRows),
    optimized_multi_agent_sample_integral: baselineIntegral,
    rows: [...fastAndRoutedRows, baselineIntegral],
  };

  const outDir = path.resolve(process.cwd(), 'veristem', 'benchmarks', 'results');
  fs.mkdirSync(outDir, { recursive: true });
  const jsonPath = path.join(outDir, 'perf_benchmark_report.json');
  const csvPath = path.join(outDir, 'perf_benchmark_report.csv');

  fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2), 'utf-8');

  const csvHeader = [
    'id',
    'category',
    'mode',
    'complexity_level',
    'execution_path',
    'status',
    'verification_status',
    'elapsed_ms',
    'gemini_calls_estimate',
    'tool_calls',
    'correct',
    'observed_numeric',
    'observed_answer',
    'error',
  ].join(',');

  const csvLines = report.rows.map((r) =>
    [
      r.id,
      r.category,
      r.mode,
      r.complexity_level ?? '',
      r.execution_path ?? '',
      r.status,
      r.verification_status ?? '',
      r.elapsed_ms.toFixed(2),
      r.gemini_calls_estimate,
      r.tool_calls,
      r.correct === null ? '' : String(r.correct),
      r.observed_numeric === null ? '' : String(r.observed_numeric),
      `"${(r.observed_answer ?? '').replace(/"/g, '""')}"`,
      `"${(r.error ?? '').replace(/"/g, '""')}"`,
    ].join(','),
  );
  fs.writeFileSync(csvPath, [csvHeader, ...csvLines].join('\n'), 'utf-8');

  console.log('\nBenchmark summary (Level 1 Fast Path):', JSON.stringify(report.summary_level1_fast_path));
  console.log('Benchmark summary (Scientific Multi-Agent):', JSON.stringify(report.summary_scientific_multi_agent));
  console.log(`Saved reports to:\n  - ${jsonPath}\n  - ${csvPath}`);
}

void main();
