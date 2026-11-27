import { DatabaseSync } from 'node:sqlite';
import path from 'node:path';
import type {
  AgentExecutionResult,
  CorrectionRequest,
  ExecutionPlan,
  FinalSolutionReport,
  HumanReviewDecision,
  MetricsSummary,
  ProblemDomain,
  ProblemRecord,
  ProblemSpecification,
  ToolExecutionRecord,
  VerificationReport,
  WorkflowStatus,
} from '../src/types/veristem.ts';

const DB_PATH = path.resolve(process.cwd(), 'veristem_web.db');

export class VeriSTEMDatabase {
  private db: DatabaseSync;

  constructor(dbPath: string = DB_PATH) {
    this.db = new DatabaseSync(dbPath);
    this.db.exec('PRAGMA journal_mode = WAL;');
    this.db.exec('PRAGMA foreign_keys = ON;');
    this.initSchema();
  }

  private initSchema(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS problems (
        id TEXT PRIMARY KEY,
        execution_id TEXT,
        question TEXT NOT NULL,
        domain_hint TEXT,
        status TEXT NOT NULL DEFAULT 'PENDING',
        current_stage TEXT NOT NULL DEFAULT 'CREATED',
        specification_json TEXT,
        plan_json TEXT,
        verification_json TEXT,
        human_review_json TEXT,
        final_report_json TEXT,
        correction_attempts INTEGER NOT NULL DEFAULT 0,
        max_correction_attempts INTEGER NOT NULL DEFAULT 3,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS agent_states (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        problem_id TEXT NOT NULL,
        execution_id TEXT NOT NULL,
        result_json TEXT NOT NULL,
        duration_ms REAL NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE
      );

      CREATE TABLE IF NOT EXISTS tool_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        problem_id TEXT NOT NULL,
        execution_id TEXT NOT NULL,
        record_json TEXT NOT NULL,
        success INTEGER NOT NULL DEFAULT 1,
        duration_ms REAL NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE
      );

      CREATE TABLE IF NOT EXISTS retry_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        problem_id TEXT NOT NULL,
        execution_id TEXT NOT NULL,
        correction_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE
      );

      CREATE TABLE IF NOT EXISTS idempotency_keys (
        idempotency_key TEXT PRIMARY KEY,
        problem_id TEXT NOT NULL,
        execution_id TEXT NOT NULL,
        created_at TEXT NOT NULL
      );
    `);
  }

  public isHealthy(): boolean {
    try {
      const row = this.db.prepare('SELECT 1 AS ok').get() as { ok?: number } | undefined;
      return row?.ok === 1;
    } catch {
      return false;
    }
  }

  public createProblem(
    problemId: string,
    question: string,
    domainHint: ProblemDomain | null,
  ): ProblemRecord {
    const now = new Date().toISOString();
    this.db
      .prepare(
        `INSERT INTO problems (id, question, domain_hint, status, current_stage, created_at, updated_at)
         VALUES (?, ?, ?, 'PENDING', 'CREATED', ?, ?)`,
      )
      .run(problemId, question, domainHint, now, now);
    return this.getProblem(problemId)!;
  }

  public getIdempotent(key: string): { problem_id: string; execution_id: string } | null {
    const row = this.db
      .prepare('SELECT problem_id, execution_id FROM idempotency_keys WHERE idempotency_key = ?')
      .get(key) as { problem_id: string; execution_id: string } | undefined;
    return row ?? null;
  }

  public saveIdempotent(key: string, problemId: string, executionId: string): void {
    this.db
      .prepare(
        `INSERT OR IGNORE INTO idempotency_keys (idempotency_key, problem_id, execution_id, created_at)
         VALUES (?, ?, ?, ?)`,
      )
      .run(key, problemId, executionId, new Date().toISOString());
  }

  public updateProblem(
    problemId: string,
    patch: {
      execution_id?: string;
      status?: WorkflowStatus;
      current_stage?: string;
      specification?: ProblemSpecification;
      plan?: ExecutionPlan;
      verification_report?: VerificationReport;
      human_review?: HumanReviewDecision;
      final_report?: FinalSolutionReport;
      correction_attempts?: number;
      max_correction_attempts?: number;
    },
  ): void {
    const sets: string[] = ['updated_at = ?'];
    const values: Array<string | number | null> = [new Date().toISOString()];

    if (patch.execution_id !== undefined) {
      sets.push('execution_id = ?');
      values.push(patch.execution_id);
    }
    if (patch.status !== undefined) {
      sets.push('status = ?');
      values.push(patch.status);
    }
    if (patch.current_stage !== undefined) {
      sets.push('current_stage = ?');
      values.push(patch.current_stage);
    }
    if (patch.specification !== undefined) {
      sets.push('specification_json = ?');
      values.push(JSON.stringify(patch.specification));
    }
    if (patch.plan !== undefined) {
      sets.push('plan_json = ?');
      values.push(JSON.stringify(patch.plan));
    }
    if (patch.verification_report !== undefined) {
      sets.push('verification_json = ?');
      values.push(JSON.stringify(patch.verification_report));
    }
    if (patch.human_review !== undefined) {
      sets.push('human_review_json = ?');
      values.push(JSON.stringify(patch.human_review));
    }
    if (patch.final_report !== undefined) {
      sets.push('final_report_json = ?');
      values.push(JSON.stringify(patch.final_report));
    }
    if (patch.correction_attempts !== undefined) {
      sets.push('correction_attempts = ?');
      values.push(patch.correction_attempts);
    }
    if (patch.max_correction_attempts !== undefined) {
      sets.push('max_correction_attempts = ?');
      values.push(patch.max_correction_attempts);
    }

    values.push(problemId);
    this.db.prepare(`UPDATE problems SET ${sets.join(', ')} WHERE id = ?`).run(...values);
  }

  public clearExecutionArtifacts(problemId: string): void {
    this.db.prepare('DELETE FROM agent_states WHERE problem_id = ?').run(problemId);
    this.db.prepare('DELETE FROM tool_records WHERE problem_id = ?').run(problemId);
    this.db.prepare('DELETE FROM retry_history WHERE problem_id = ?').run(problemId);
  }

  public addAgentResult(problemId: string, executionId: string, res: AgentExecutionResult): void {
    const now = new Date().toISOString();
    this.db
      .prepare(
        `INSERT INTO agent_states (problem_id, execution_id, result_json, duration_ms, created_at)
         VALUES (?, ?, ?, ?, ?)`,
      )
      .run(problemId, executionId, JSON.stringify(res), res.duration_ms, now);

    for (const tool of res.tool_records) {
      this.db
        .prepare(
          `INSERT INTO tool_records (problem_id, execution_id, record_json, success, duration_ms, created_at)
           VALUES (?, ?, ?, ?, ?, ?)`,
        )
        .run(problemId, executionId, JSON.stringify(tool), tool.success ? 1 : 0, tool.duration_ms, now);
    }
  }

  public addCorrection(problemId: string, executionId: string, corr: CorrectionRequest): void {
    this.db
      .prepare(
        `INSERT INTO retry_history (problem_id, execution_id, correction_json, created_at)
         VALUES (?, ?, ?, ?)`,
      )
      .run(problemId, executionId, JSON.stringify(corr), new Date().toISOString());
  }

  public getProblem(problemId: string): ProblemRecord | null {
    const row = this.db.prepare('SELECT * FROM problems WHERE id = ?').get(problemId) as
      | Record<string, unknown>
      | undefined;
    if (!row) {
      return null;
    }

    const agentRows = this.db
      .prepare('SELECT result_json FROM agent_states WHERE problem_id = ? ORDER BY id ASC')
      .all(problemId) as Array<{ result_json: string }>;
    const toolRows = this.db
      .prepare('SELECT record_json FROM tool_records WHERE problem_id = ? ORDER BY id ASC')
      .all(problemId) as Array<{ record_json: string }>;
    const retryRows = this.db
      .prepare('SELECT correction_json FROM retry_history WHERE problem_id = ? ORDER BY id ASC')
      .all(problemId) as Array<{ correction_json: string }>;

    return {
      problem_id: String(row.id),
      execution_id: row.execution_id ? String(row.execution_id) : null,
      question: String(row.question),
      domain_hint: (row.domain_hint as ProblemDomain | null) ?? null,
      status: row.status as WorkflowStatus,
      current_stage: String(row.current_stage),
      correction_attempts: Number(row.correction_attempts ?? 0),
      specification: row.specification_json ? JSON.parse(String(row.specification_json)) : null,
      plan: row.plan_json ? JSON.parse(String(row.plan_json)) : null,
      agent_timeline: agentRows.map((r) => JSON.parse(r.result_json) as AgentExecutionResult),
      tool_records: toolRows.map((r) => JSON.parse(r.record_json) as ToolExecutionRecord),
      verification_report: row.verification_json ? JSON.parse(String(row.verification_json)) : null,
      correction_history: retryRows.map((r) => JSON.parse(r.correction_json) as CorrectionRequest),
      human_review: row.human_review_json ? JSON.parse(String(row.human_review_json)) : null,
      final_report: row.final_report_json ? JSON.parse(String(row.final_report_json)) : null,
      created_at: String(row.created_at),
      updated_at: String(row.updated_at),
    };
  }

  public listProblems(limit = 25): ProblemRecord[] {
    const rows = this.db
      .prepare('SELECT id FROM problems ORDER BY updated_at DESC LIMIT ?')
      .all(limit) as Array<{ id: string }>;
    return rows
      .map((r) => this.getProblem(r.id))
      .filter((r): r is ProblemRecord => r !== null);
  }

  public getMetrics(): MetricsSummary {
    const counts = this.db
      .prepare(
        `SELECT
           COUNT(*) AS total,
           SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) AS completed,
           SUM(CASE WHEN status IN ('UNRESOLVED', 'HUMAN_REVIEW') THEN 1 ELSE 0 END) AS unresolved,
           COALESCE(AVG(correction_attempts), 0) AS avg_retries
         FROM problems`,
      )
      .get() as Record<string, number>;

    const reports = this.db
      .prepare('SELECT final_report_json FROM problems WHERE final_report_json IS NOT NULL')
      .all() as Array<{ final_report_json: string }>;

    let passed = 0;
    let failed = 0;
    for (const r of reports) {
      const parsed = JSON.parse(r.final_report_json) as FinalSolutionReport;
      if (parsed.verification_status === 'PASS') passed += 1;
      else if (parsed.verification_status === 'FAIL') failed += 1;
    }

    const toolStats = this.db
      .prepare('SELECT COUNT(*) AS cnt FROM tool_records')
      .get() as { cnt: number };
    const agentStats = this.db
      .prepare('SELECT COALESCE(AVG(duration_ms), 0) AS avg_dur FROM agent_states')
      .get() as { avg_dur: number };

    return {
      total_problems: Number(counts.total ?? 0),
      completed_problems: Number(counts.completed ?? 0),
      passed_verifications: passed,
      failed_verifications: failed,
      unresolved_problems: Number(counts.unresolved ?? 0),
      average_correction_attempts: Number(Number(counts.avg_retries ?? 0).toFixed(3)),
      average_duration_ms: Number(Number(agentStats.avg_dur ?? 0).toFixed(2)),
      total_tool_executions: Number(toolStats.cnt ?? 0),
    };
  }
}

let _instance: VeriSTEMDatabase | null = null;

export function getDatabase(): VeriSTEMDatabase {
  if (!_instance) {
    _instance = new VeriSTEMDatabase();
  }
  return _instance;
}
