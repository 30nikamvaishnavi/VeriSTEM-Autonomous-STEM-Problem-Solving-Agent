import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import dotenv from 'dotenv';
import express, { type NextFunction, type Request, type Response, Router } from 'express';
import { createServer as createViteServer, type ViteDevServer } from 'vite';
import { z } from 'zod';
import type { BenchmarkCase, HumanReviewDecision, ProblemDomain } from './src/types/veristem.ts';
import { classifyProblemComplexity } from './server/complexity_router.ts';
import { getDatabase } from './server/db.ts';
import { tryFastPathSolve } from './server/fast_path.ts';
import { runProblemWorkflow } from './server/workflow.ts';

dotenv.config();

const PORT = 3000;
const MAX_PROBLEM_CHARS = 8000;
const RATE_LIMIT_WINDOW_MS = 60_000;
const RATE_LIMIT_MAX = Number(process.env.RATE_LIMIT_REQUESTS_PER_MINUTE || 180);
function resolveSessionSecret(): string {
  const configured = (process.env.SESSION_SECRET ?? '').trim();
  const isPlaceholder =
    !configured ||
    configured === 'your_secure_session_secret_here' ||
    configured.startsWith('MY_') ||
    configured.length < 32;

  if (isPlaceholder) {
    if (configured && configured !== 'your_secure_session_secret_here') {
      console.warn(
        'SESSION_SECRET in environment is shorter than 32 characters; using an ephemeral CSPRNG secret in memory.',
      );
    } else if (process.env.NODE_ENV === 'production') {
      console.warn(
        'SESSION_SECRET is missing or placeholder in production; using an ephemeral CSPRNG secret in memory.',
      );
    }
    return crypto.randomBytes(48).toString('hex');
  }

  return configured;
}

const SESSION_SIGNING_SECRET = resolveSessionSecret();

const rateBuckets = new Map<string, number[]>();

function signWorkbenchToken(sessionId: string): string {
  const exp = Date.now() + 12 * 60 * 60 * 1000;
  const payload = `${sessionId}.${exp}`;
  const sig = crypto
    .createHmac('sha256', SESSION_SIGNING_SECRET)
    .update(payload)
    .digest('hex');
  return `${payload}.${sig}`;
}

function verifyWorkbenchToken(token: string): boolean {
  const parts = token.split('.');
  if (parts.length !== 3) return false;
  const [sessionId, expStr, sig] = parts;
  const exp = Number(expStr);
  if (!sessionId || !Number.isFinite(exp) || Date.now() > exp) return false;
  const expectedSig = crypto
    .createHmac('sha256', SESSION_SIGNING_SECRET)
    .update(`${sessionId}.${expStr}`)
    .digest('hex');
  const a = Buffer.from(sig, 'utf-8');
  const b = Buffer.from(expectedSig, 'utf-8');
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}

function rateLimitMiddleware(req: Request, res: Response, next: NextFunction): void {
  const isRead = req.method === 'GET';
  const key = `${req.ip || 'unknown'}:${isRead ? 'read' : 'write'}`;
  const now = Date.now();
  const cutoff = now - RATE_LIMIT_WINDOW_MS;
  const effectiveMax = isRead ? Math.max(RATE_LIMIT_MAX * 10, 300) : RATE_LIMIT_MAX;
  const timestamps = (rateBuckets.get(key) ?? []).filter((t) => t > cutoff);
  if (timestamps.length >= effectiveMax) {
    res.status(429).json({
      error_code: 'RATE_LIMIT_EXCEEDED',
      message: 'Too many requests; please retry shortly.',
      correlation_id: res.getHeader('X-Correlation-ID'),
    });
    return;
  }
  timestamps.push(now);
  rateBuckets.set(key, timestamps);
  next();
}

function apiKeyMiddleware(req: Request, res: Response, next: NextFunction): void {
  const expected = (process.env.VERISTEM_API_KEY ?? '').trim();
  if (!expected) {
    next();
    return;
  }
  const provided = (req.header('X-API-Key') ?? '').trim();
  const a = Buffer.from(provided, 'utf-8');
  const b = Buffer.from(expected, 'utf-8');
  if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) {
    res.status(401).json({
      error_code: 'UNAUTHORIZED',
      message: 'Invalid or missing X-API-Key header.',
      correlation_id: res.getHeader('X-Correlation-ID'),
    });
    return;
  }
  next();
}

function workbenchSessionMiddleware(req: Request, res: Response, next: NextFunction): void {
  const auth = (req.header('Authorization') ?? '').trim();
  const bearer = auth.startsWith('Bearer ') ? auth.slice(7).trim() : '';
  const customHeader = (req.header('X-Requested-With') ?? '').trim();

  if (!bearer || !verifyWorkbenchToken(bearer) || customHeader !== 'VeriSTEM-Workbench') {
    res.status(401).json({
      error_code: 'UNAUTHORIZED',
      message: 'Invalid or expired workbench session. Please reload the page.',
      correlation_id: res.getHeader('X-Correlation-ID'),
    });
    return;
  }
  next();
}

const CreateProblemBody = z.object({
  question: z.string().trim().min(5).max(MAX_PROBLEM_CHARS),
  domain_hint: z
    .enum(['mathematics', 'physics', 'quantitative_finance', 'general_stem'])
    .nullable()
    .optional(),
});

const SolveProblemBody = z.object({
  idempotency_key: z.string().max(128).optional(),
  max_correction_attempts: z.number().int().min(1).max(5).default(3),
  run_async: z.boolean().default(true),
  force_multi_agent: z.boolean().default(false),
});

const ResumeProblemBody = z.object({
  action: z.enum(['approve', 'clarify', 'override', 'reject']),
  clarification_response: z.string().max(4000).nullable().optional(),
  notes: z.string().max(2000).default(''),
  override_answer: z.string().max(1000).nullable().optional(),
});

function buildProtectedRoutes(): Router {
  const router = Router();
  const db = getDatabase();

  router.get('/metrics', rateLimitMiddleware, (_req: Request, res: Response) => {
    res.json(db.getMetrics());
  });

  router.get('/problems', rateLimitMiddleware, (_req: Request, res: Response) => {
    res.json(db.listProblems(30));
  });

  router.post('/problems', rateLimitMiddleware, (req: Request, res: Response) => {
    const parsed = CreateProblemBody.safeParse(req.body);
    if (!parsed.success) {
      res.status(422).json({
        error_code: 'VALIDATION_ERROR',
        message: parsed.error.issues.map((i) => i.message).join('; '),
        correlation_id: res.getHeader('X-Correlation-ID'),
      });
      return;
    }
    const problemId = `prob_${crypto.randomBytes(6).toString('hex')}`;
    const created = db.createProblem(
      problemId,
      parsed.data.question,
      (parsed.data.domain_hint as ProblemDomain | null | undefined) ?? null,
    );
    res.status(201).json({
      problem_id: created.problem_id,
      status: created.status,
      created_at: created.created_at,
    });
  });

  router.post(
    '/problems/:problemId/solve',
    rateLimitMiddleware,
    async (req: Request, res: Response) => {
      const problemId = String(req.params.problemId);
      const problem = db.getProblem(problemId);
      if (!problem) {
        res.status(404).json({
          error_code: 'NOT_FOUND',
          message: `Problem '${problemId}' not found.`,
          correlation_id: res.getHeader('X-Correlation-ID'),
        });
        return;
      }
      const parsed = SolveProblemBody.safeParse(req.body ?? {});
      if (!parsed.success) {
        res.status(422).json({
          error_code: 'VALIDATION_ERROR',
          message: parsed.error.issues.map((i) => i.message).join('; '),
          correlation_id: res.getHeader('X-Correlation-ID'),
        });
        return;
      }

      if (parsed.data.idempotency_key) {
        const existing = db.getIdempotent(parsed.data.idempotency_key);
        if (existing) {
          res.status(202).json({
            problem_id: existing.problem_id,
            execution_id: existing.execution_id,
            status: problem.status,
            message: 'Idempotent solve request already registered.',
          });
          return;
        }
      }

      if (problem.status === 'RUNNING') {
        res.status(409).json({
          error_code: 'INVALID_STATE',
          message: 'Problem workflow is already running.',
          correlation_id: res.getHeader('X-Correlation-ID'),
        });
        return;
      }

      const executionId = `exec_${crypto.randomBytes(6).toString('hex')}`;
      if (parsed.data.idempotency_key) {
        db.saveIdempotent(parsed.data.idempotency_key, problemId, executionId);
      }

      const decision = classifyProblemComplexity(problem.question, problem.domain_hint);
      const effectivePath = parsed.data.force_multi_agent
        ? 'MULTI_AGENT_WORKFLOW'
        : decision.execution_path;

      if (effectivePath === 'FAST_PATH') {
        db.clearExecutionArtifacts(problemId);
        db.updateProblem(problemId, {
          execution_id: executionId,
          status: 'RUNNING',
          current_stage: 'FAST_PATH',
          complexity_level: decision.complexity_level,
          execution_path: 'FAST_PATH',
          correction_attempts: 0,
        });

        const fastResult = await tryFastPathSolve(problemId, executionId, problem.question, decision);
        if (fastResult.handled && fastResult.final_report && fastResult.verification_report) {
          for (const agentStep of fastResult.agent_timeline ?? []) {
            db.addAgentResult(problemId, executionId, agentStep);
          }
          db.updateProblem(problemId, {
            status: 'COMPLETED',
            current_stage: 'COMPLETED',
            complexity_level: decision.complexity_level,
            execution_path: 'FAST_PATH',
            elapsed_ms: fastResult.final_report.elapsed_ms,
            specification: fastResult.specification,
            plan: fastResult.plan,
            verification_report: fastResult.verification_report,
            final_report: fastResult.final_report,
          });
          res.status(200).json({
            problem_id: problemId,
            execution_id: executionId,
            status: 'COMPLETED',
            complexity_level: decision.complexity_level,
            execution_path: 'FAST_PATH',
            elapsed_ms: fastResult.final_report.elapsed_ms,
            verification_status: fastResult.verification_report.status,
            message: 'Solved and independently verified via deterministic Fast Path.',
          });
          return;
        }
      }

      if (parsed.data.run_async) {
        void runProblemWorkflow(problemId, executionId, {
          maxCorrectionAttempts: parsed.data.max_correction_attempts,
          complexityLevel: decision.complexity_level,
        });
        res.status(202).json({
          problem_id: problemId,
          execution_id: executionId,
          status: 'RUNNING',
          complexity_level: decision.complexity_level,
          execution_path: 'MULTI_AGENT_WORKFLOW',
          message: 'Multi-agent verification workflow started in background.',
        });
        return;
      }

      await runProblemWorkflow(problemId, executionId, {
        maxCorrectionAttempts: parsed.data.max_correction_attempts,
        complexityLevel: decision.complexity_level,
      });
      const updated = db.getProblem(problemId)!;
      res.status(200).json({
        problem_id: problemId,
        execution_id: executionId,
        status: updated.status,
        complexity_level: updated.complexity_level,
        execution_path: updated.execution_path,
        elapsed_ms: updated.elapsed_ms,
        verification_status: updated.verification_report?.status ?? null,
        message: 'Multi-agent verification workflow completed.',
      });
    },
  );

  router.get('/problems/:problemId', rateLimitMiddleware, (req: Request, res: Response) => {
    const problem = db.getProblem(String(req.params.problemId));
    if (!problem) {
      res.status(404).json({
        error_code: 'NOT_FOUND',
        message: 'Problem not found.',
        correlation_id: res.getHeader('X-Correlation-ID'),
      });
      return;
    }
    res.json(problem);
  });

  router.get('/problems/:problemId/status', rateLimitMiddleware, (req: Request, res: Response) => {
    const problem = db.getProblem(String(req.params.problemId));
    if (!problem) {
      res.status(404).json({
        error_code: 'NOT_FOUND',
        message: 'Problem not found.',
        correlation_id: res.getHeader('X-Correlation-ID'),
      });
      return;
    }
    res.json({
      problem_id: problem.problem_id,
      execution_id: problem.execution_id,
      status: problem.status,
      current_stage: problem.current_stage,
      complexity_level: problem.complexity_level,
      execution_path: problem.execution_path,
      elapsed_ms: problem.elapsed_ms,
      correction_attempts: problem.correction_attempts,
      verification_status: problem.verification_report?.status ?? null,
      needs_clarification: problem.specification?.needs_clarification ?? false,
      clarification_question: problem.specification?.clarification_question ?? null,
      updated_at: problem.updated_at,
    });
  });

  router.get('/problems/:problemId/report', rateLimitMiddleware, (req: Request, res: Response) => {
    const problem = db.getProblem(String(req.params.problemId));
    if (!problem || !problem.final_report) {
      res.status(404).json({
        error_code: 'NOT_FOUND',
        message: 'Final report is not available yet.',
        correlation_id: res.getHeader('X-Correlation-ID'),
      });
      return;
    }
    if (req.query.format === 'markdown') {
      res.type('text/markdown').send(problem.final_report.explanation_markdown);
      return;
    }
    res.json(problem.final_report);
  });

  router.post('/problems/:problemId/resume', rateLimitMiddleware, (req: Request, res: Response) => {
    const problemId = String(req.params.problemId);
    const problem = db.getProblem(problemId);
    if (!problem) {
      res.status(404).json({
        error_code: 'NOT_FOUND',
        message: 'Problem not found.',
        correlation_id: res.getHeader('X-Correlation-ID'),
      });
      return;
    }
    if (problem.status !== 'NEEDS_CLARIFICATION' && problem.status !== 'HUMAN_REVIEW') {
      res.status(409).json({
        error_code: 'INVALID_STATE',
        message: `Cannot resume problem from status '${problem.status}'.`,
        correlation_id: res.getHeader('X-Correlation-ID'),
      });
      return;
    }
    const parsed = ResumeProblemBody.safeParse(req.body ?? {});
    if (!parsed.success) {
      res.status(422).json({
        error_code: 'VALIDATION_ERROR',
        message: parsed.error.issues.map((i) => i.message).join('; '),
        correlation_id: res.getHeader('X-Correlation-ID'),
      });
      return;
    }

    const decision: HumanReviewDecision = {
      reviewer_action: parsed.data.action,
      notes: parsed.data.notes,
      clarification_response: parsed.data.clarification_response ?? null,
      override_answer: parsed.data.override_answer ?? null,
      decided_at: new Date().toISOString(),
    };
    db.updateProblem(problemId, { human_review: decision });

    const execId = problem.execution_id || `exec_${crypto.randomBytes(6).toString('hex')}`;
    void runProblemWorkflow(problemId, execId, {
      clarificationResponse:
        problem.status === 'NEEDS_CLARIFICATION' || parsed.data.action === 'clarify'
          ? parsed.data.clarification_response || parsed.data.notes
          : null,
      humanDecision: problem.status === 'HUMAN_REVIEW' ? decision : null,
    });

    res.status(202).json({
      problem_id: problemId,
      execution_id: execId,
      status: 'RUNNING',
      message: 'Resumed workflow from persisted checkpoint.',
    });
  });

  return router;
}

async function startServer(): Promise<void> {
  const app = express();
  const db = getDatabase();

  app.set('trust proxy', true);
  app.use(express.json({ limit: '1mb' }));

  app.use((req: Request, res: Response, next: NextFunction) => {
    const corrId = req.header('X-Correlation-ID') || `corr_${crypto.randomBytes(6).toString('hex')}`;
    res.setHeader('X-Correlation-ID', corrId);
    next();
  });

  app.get('/api/v1/health', (_req: Request, res: Response) => {
    const geminiKey = (process.env.GEMINI_API_KEY ?? '').trim();
    const dbOk = db.isHealthy();
    res.json({
      status: dbOk ? 'ok' : 'degraded',
      version: '1.0.0',
      database_connected: dbOk,
      gemini_configured: Boolean(geminiKey && geminiKey !== 'MY_GEMINI_API_KEY'),
      timestamp: new Date().toISOString(),
    });
  });

  app.get('/api/v1/benchmarks', (_req: Request, res: Response) => {
    const datasetPath = path.resolve(process.cwd(), 'veristem', 'datasets', 'stem_benchmarks_32.json');
    if (!fs.existsSync(datasetPath)) {
      res.json([]);
      return;
    }
    const items = JSON.parse(fs.readFileSync(datasetPath, 'utf-8')) as BenchmarkCase[];
    res.json(items);
  });

  // First-party browser workbench session bootstrap (HMAC-signed token held in JS memory)
  app.post('/api/web/session', rateLimitMiddleware, (req: Request, res: Response) => {
    const origin = req.header('Origin');
    const host = req.header('X-Forwarded-Host') || req.header('Host');
    if (origin && host) {
      try {
        const originHost = new URL(origin).host;
        if (originHost !== host) {
          res.status(403).json({
            error_code: 'FORBIDDEN',
            message: 'Origin mismatch on workbench session request.',
            correlation_id: res.getHeader('X-Correlation-ID'),
          });
          return;
        }
      } catch {
        res.status(403).json({
          error_code: 'FORBIDDEN',
          message: 'Invalid Origin header.',
          correlation_id: res.getHeader('X-Correlation-ID'),
        });
        return;
      }
    }
    const sessionId = crypto.randomBytes(16).toString('hex');
    const token = signWorkbenchToken(sessionId);
    res.json({ token });
  });

  const sharedRouter = buildProtectedRoutes();
  // External API v1 protected by X-API-Key when VERISTEM_API_KEY is set
  app.use('/api/v1', apiKeyMiddleware, sharedRouter);
  // First-party React Workbench routes protected by signed session token + custom header
  app.use('/api/web', workbenchSessionMiddleware, sharedRouter);

  const httpServer = http.createServer(app);
  let vite: ViteDevServer | null = null;

  if (process.env.NODE_ENV === 'production') {
    const distDir = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req: Request, res: Response) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  } else {
    const disableHmr = process.env.DISABLE_HMR === 'true';
    vite = await createViteServer({
      server: {
        middlewareMode: true,
        ws: { server: httpServer },
        ...(disableHmr ? { hmr: false } : {}),
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  let shuttingDown = false;
  const shutdown = async (signal: string): Promise<void> => {
    if (shuttingDown) return;
    shuttingDown = true;
    console.log(`VeriSTEM server shutting down on ${signal}...`);
    try {
      if (vite) {
        await vite.close();
      }
      await new Promise<void>((resolve) => {
        httpServer.close(() => resolve());
      });
    } finally {
      db.close();
      process.exit(0);
    }
  };

  process.once('SIGINT', () => void shutdown('SIGINT'));
  process.once('SIGTERM', () => void shutdown('SIGTERM'));

  httpServer.listen(PORT, '0.0.0.0', () => {
    console.log(`VeriSTEM server running on http://0.0.0.0:${PORT}`);
  });
}

void startServer();
