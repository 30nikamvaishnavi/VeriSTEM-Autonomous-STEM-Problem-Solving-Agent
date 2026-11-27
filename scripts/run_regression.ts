import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { classifyProblemComplexity } from '../server/complexity_router.ts';
import { VeriSTEMDatabase } from '../server/db.ts';
import { tryFastPathSolve } from '../server/fast_path.ts';
import { executeToolAsync, executeToolsParallelWithCache } from '../server/tools.ts';

const BASE_URL = 'http://127.0.0.1:3000';

async function testWebSocketHandshake(): Promise<void> {
  const clientRes = await fetch(`${BASE_URL}/@vite/client`);
  const clientJs = await clientRes.text();
  assert.ok(!clientJs.includes('const hmrPort = 24678'), 'Vite client must not point to unreachable port 24678');
  const tokenMatch = clientJs.match(/const wsToken = "([^"]+)"/);
  const wsPath = tokenMatch ? `/?token=${encodeURIComponent(tokenMatch[1])}` : '/';

  await new Promise<void>((resolve, reject) => {
    const key = crypto.randomBytes(16).toString('base64');
    const socket = net.createConnection({ host: '127.0.0.1', port: 3000 }, () => {
      const req = [
        `GET ${wsPath} HTTP/1.1`,
        'Host: 127.0.0.1:3000',
        'Upgrade: websocket',
        'Connection: Upgrade',
        `Sec-WebSocket-Key: ${key}`,
        'Sec-WebSocket-Version: 13',
        'Sec-WebSocket-Protocol: vite-hmr',
        '',
        '',
      ].join('\r\n');
      socket.write(req);
    });

    let buf = '';
    const timer = setTimeout(() => {
      socket.destroy();
      reject(new Error('WebSocket handshake timed out'));
    }, 5000);

    socket.on('data', (chunk) => {
      buf += chunk.toString('utf-8');
      if (buf.includes('\r\n\r\n')) {
        clearTimeout(timer);
        const statusLine = buf.split('\r\n')[0];
        socket.end();
        if (statusLine.startsWith('HTTP/1.1 101')) {
          resolve();
        } else {
          reject(new Error(`Expected 101 Switching Protocols, got: ${statusLine}`));
        }
      }
    });

    socket.on('error', (err) => {
      clearTimeout(timer);
      reject(err);
    });
  });
}

async function main(): Promise<void> {
  const results: Array<{ name: string; passed: boolean; detail: string }> = [];

  async function runTest(name: string, fn: () => Promise<void>): Promise<void> {
    const t0 = performance.now();
    try {
      await fn();
      const ms = (performance.now() - t0).toFixed(1);
      results.push({ name, passed: true, detail: `PASS (${ms} ms)` });
      console.log(`[PASS] ${name} (${ms} ms)`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      results.push({ name, passed: false, detail: `FAIL: ${msg}` });
      console.error(`[FAIL] ${name}: ${msg}`);
    }
  }

  await runTest('1. Sample definite integral exact & numeric + <5s fast path', async () => {
    const q = 'Evaluate the definite integral of x^2 * exp(-x) from x = 0 to x = 2.';
    const decision = classifyProblemComplexity(q);
    assert.equal(decision.complexity_level, 'LEVEL_1_SIMPLE');
    assert.equal(decision.execution_path, 'FAST_PATH');

    const res = await tryFastPathSolve('prob_reg_1', 'exec_reg_1', q, decision);
    assert.equal(res.handled, true);
    assert.equal(res.verification_report?.status, 'PASS');
    assert.ok(res.final_report);
    assert.ok(res.final_report.final_answer.includes('e^(-2)'));
    assert.ok(Math.abs((res.final_report.numeric_value ?? 0) - 0.646647167633873) < 1e-12);
    assert.ok((res.final_report.elapsed_ms ?? 9999) < 5000);
  });

  await runTest('2. Independent verification rejects wrong candidate value', async () => {
    const bad = await executeToolAsync('independent_verifier', {
      verification_type: 'integral_quadrature',
      expression: 'x**2 * math.exp(-x)',
      variable: 'x',
      lower_limit: 0,
      upper_limit: 2,
      candidate_value: 0.999,
      tolerance: 1e-4,
    });
    assert.equal(bad.success, true);
    assert.equal(bad.output_payload.passed, false);
  });

  await runTest('3. Complexity router distinguishes simple vs intermediate/advanced STEM', async () => {
    const d1 = classifyProblemComplexity('Solve 3*x + 7 = 22 for x.');
    const d2 = classifyProblemComplexity(
      'A projectile is launched from ground level at 45 m/s at 35 degrees above horizontal (g = 9.81 m/s^2). Compute the horizontal range in meters.',
    );
    const d3 = classifyProblemComplexity(
      'Compute the Black-Scholes European call price for S=100, K=105, r=0.05, sigma=0.20, T=1.0 year, and verify against Monte Carlo simulation.',
    );
    assert.equal(d1.execution_path, 'FAST_PATH');
    assert.equal(d2.execution_path, 'MULTI_AGENT_WORKFLOW');
    assert.equal(d3.execution_path, 'MULTI_AGENT_WORKFLOW');
    assert.equal(d3.complexity_level, 'LEVEL_3_ADVANCED');
  });

  await runTest('4. Fast path escalates unsupported expressions safely', async () => {
    const q = 'Evaluate the definite integral of exp(-x^2) * sin(x^3) from x = 0 to x = 2.';
    const decision = classifyProblemComplexity(q);
    const res = await tryFastPathSolve('prob_reg_4', 'exec_reg_4', q, decision);
    assert.equal(res.handled, false);
  });

  await runTest('5. Parallel tool execution + deduplication cache', async () => {
    const cache = new Map();
    const args = JSON.stringify({
      verification_type: 'numeric_tolerance',
      candidate_value: 20.0,
      expected_reference: 20.0,
      tolerance: 1e-6,
    });
    const out = await executeToolsParallelWithCache(
      [
        { task_id: 't1', tool_name: 'independent_verifier', arguments_json: args },
        { task_id: 't2', tool_name: 'independent_verifier', arguments_json: args },
      ],
      cache,
    );
    assert.equal(out.length, 2);
    assert.equal(out[0].success, true);
    assert.equal(out[1].success, true);
    assert.equal(cache.size, 1);
  });

  await runTest('6. Python sandbox timeout enforcement', async () => {
    const rec = await executeToolAsync('python_sandbox', {
      code: 'import math\nRESULT = math.factorial(10)',
    });
    assert.equal(rec.success, true);
    assert.equal(rec.output_payload.result, 3628800);
  });

  await runTest('7. SQLite database persistence across instances', async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'veristem_db_test_'));
    const dbFile = path.join(tmpDir, 'test.db');
    try {
      const db1 = new VeriSTEMDatabase(dbFile);
      db1.createProblem('prob_persist_1', 'Solve 2*x = 10 for x.', 'mathematics');
      db1.updateProblem('prob_persist_1', {
        status: 'COMPLETED',
        current_stage: 'COMPLETED',
        complexity_level: 'LEVEL_1_SIMPLE',
        execution_path: 'FAST_PATH',
        elapsed_ms: 42.5,
      });
      db1.close();

      const db2 = new VeriSTEMDatabase(dbFile);
      const loaded = db2.getProblem('prob_persist_1');
      assert.ok(loaded);
      assert.equal(loaded.status, 'COMPLETED');
      assert.equal(loaded.complexity_level, 'LEVEL_1_SIMPLE');
      assert.equal(loaded.execution_path, 'FAST_PATH');
      assert.equal(loaded.elapsed_ms, 42.5);
      db2.close();
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  await runTest('8. Live HTTP health + Vite HMR WebSocket 101 handshake', async () => {
    const healthRes = await fetch(`${BASE_URL}/api/v1/health`);
    assert.equal(healthRes.status, 200);
    const health = (await healthRes.json()) as { status: string; database_connected: boolean };
    assert.equal(health.status, 'ok');
    assert.equal(health.database_connected, true);
    await testWebSocketHandshake();
  });

  await runTest('9. Live Workbench API end-to-end Fast Path solve', async () => {
    const sessRes = await fetch(`${BASE_URL}/api/web/session`, { method: 'POST' });
    assert.equal(sessRes.status, 200);
    const { token } = (await sessRes.json()) as { token: string };

    const headers = {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
      'X-Requested-With': 'VeriSTEM-Workbench',
    };

    const createRes = await fetch(`${BASE_URL}/api/web/problems`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        question: 'Evaluate the definite integral of x^2 * exp(-x) from x = 0 to x = 2.',
        domain_hint: 'mathematics',
      }),
    });
    assert.equal(createRes.status, 201);
    const created = (await createRes.json()) as { problem_id: string };

    const solveRes = await fetch(`${BASE_URL}/api/web/problems/${created.problem_id}/solve`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ max_correction_attempts: 3, run_async: true }),
    });
    assert.equal(solveRes.status, 200);
    const solved = (await solveRes.json()) as {
      status: string;
      complexity_level: string;
      execution_path: string;
      elapsed_ms: number;
      verification_status: string;
    };
    assert.equal(solved.status, 'COMPLETED');
    assert.equal(solved.execution_path, 'FAST_PATH');
    assert.equal(solved.verification_status, 'PASS');
    assert.ok(solved.elapsed_ms < 5000);

    const detailRes = await fetch(`${BASE_URL}/api/web/problems/${created.problem_id}`, { headers });
    assert.equal(detailRes.status, 200);
    const detail = (await detailRes.json()) as {
      final_report: { numeric_value: number; final_answer: string };
    };
    assert.ok(Math.abs(detail.final_report.numeric_value - 0.646647167633873) < 1e-12);
  });

  const failed = results.filter((r) => !r.passed);
  console.log(`\nRegression summary: ${results.length - failed.length}/${results.length} passed.`);
  if (failed.length > 0) {
    process.exit(1);
  }
}

void main();
