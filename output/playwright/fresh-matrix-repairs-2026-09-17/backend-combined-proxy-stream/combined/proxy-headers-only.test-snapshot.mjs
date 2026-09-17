// UAT234: exercise the installed Next server with the application's real config.
// Run separately from unit tests: node --test scripts/__tests__/quickstart-proxy-timeout.test.mjs
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtemp, mkdir, rm, symlink, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import path from 'node:path';
import { after, before, test } from 'node:test';
import { setTimeout as delay } from 'node:timers/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';

const app = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const records = [];
let fixture;
let next;
let origin;
let serverLog = '';
let abortArrived;
let abortClosed;
const abortedRequestArrived = new Promise(resolve => { abortArrived = resolve; });
const abortedRequestClosed = new Promise(resolve => { abortClosed = resolve; });
const pendingTimers = new Set();
const upstream = createServer(async (req, res) => {
  let body = '';
  for await (const chunk of req) body += chunk;
  records.push({ path: req.url, method: req.method, auth: req.headers.authorization, body });
  res.setHeader('content-type', 'application/json');
  if (req.url === '/api/v1/abort') {
    res.once('close', () => abortClosed());
    abortArrived();
  } else if (req.url === '/api/v1/silent-sse') {
    // UAT246 diagnostic: headers are available before the first body bytes.
    res.setHeader('content-type', 'text/event-stream');
    res.flushHeaders();
    const timer = setTimeout(() => {
      pendingTimers.delete(timer);
      res.end('data: {"choices":[{"delta":{"content":"BEEP BOOP"}}]}\n\ndata: [DONE]\n\n');
    }, 31_500);
    pendingTimers.add(timer);
    res.once('close', () => { clearTimeout(timer); pendingTimers.delete(timer); });
  } else if (req.url === '/api/v1/flashcards/generate') {
    // Real wall time: this must outlast Next's default 30-second rewrite cutoff.
    const timer = setTimeout(() => {
      pendingTimers.delete(timer);
      res.end(JSON.stringify({ cards: [1, 2, 3, 4, 5] }));
    }, 31_500);
    pendingTimers.add(timer);
    res.once('close', () => { clearTimeout(timer); pendingTimers.delete(timer); });
  } else if (req.url === '/api/v1/unavailable') {
    res.statusCode = 503;
    res.end(JSON.stringify({ detail: 'controlled upstream unavailable' }));
  } else {
    res.end(JSON.stringify({ ok: true }));
  }
});

async function listen(server) {
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  return server.address().port;
}

before(async () => {
  const backendPort = await listen(upstream);
  const reservation = createServer();
  const webPort = await listen(reservation);
  await new Promise(resolve => reservation.close(resolve));
  origin = `http://127.0.0.1:${webPort}`;
  // A tiny app avoids compiling unrelated UI; config and external rewrite routing
  // are the real application config and installed Next server, without mocks.
  const tempRoot = path.resolve(app, '../../.tmp');
  await mkdir(tempRoot, { recursive: true });
  fixture = await mkdtemp(path.join(tempRoot, 'quickstart-proxy-'));
  await mkdir(path.join(fixture, 'pages'));
  await writeFile(path.join(fixture, 'pages/index.js'), 'export default function Page() { return null }\n');
  await writeFile(path.join(fixture, 'package.json'), '{"type":"module"}\n');
  await writeFile(path.join(fixture, 'next.config.mjs'),
    `export { default } from ${JSON.stringify(pathToFileURL(path.join(app, 'next.config.mjs')).href)};\n`);
  await symlink(path.join(app, 'node_modules'), path.join(fixture, 'node_modules'), 'dir');
  next = spawn(process.execPath, [path.join(app, 'node_modules/next/dist/bin/next'),
    'dev', fixture, '--webpack', '--hostname', '127.0.0.1', '--port', String(webPort)], {
    cwd: app,
    env: {
      ...process.env,
      NODE_ENV: 'development',
      NEXT_TELEMETRY_DISABLED: '1',
      NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: 'quickstart',
      TLDW_INTERNAL_API_ORIGIN: `http://127.0.0.1:${backendPort}`,
      NEXT_PUBLIC_API_URL: '',
      NEXT_PUBLIC_SENTRY_DSN: '',
      TLDW_NEXT_DIST_DIR: '',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  for (const stream of [next.stdout, next.stderr]) {
    stream.on('data', chunk => { serverLog = (serverLog + chunk).slice(-8000); });
  }
  const deadline = Date.now() + 60_000;
  while (Date.now() < deadline) {
    assert.equal(next.exitCode, null, `Next exited during startup: ${serverLog}`);
    try {
      const response = await fetch(`${origin}/health`, { signal: AbortSignal.timeout(2000) });
      if (response.status === 200 && (await response.json()).ok) return;
    } catch { /* Next has not opened its listener yet. */ }
    await delay(100);
  }
  assert.fail(`Next did not become ready: ${serverLog}`);
}, { timeout: 65_000 });

after(async () => {
  if (next && next.exitCode === null) {
    const exited = once(next, 'exit');
    next.kill('SIGTERM');
    const force = setTimeout(() => next.kill('SIGKILL'), 5000);
    await exited;
    clearTimeout(force);
  }
  for (const timer of pendingTimers) clearTimeout(timer);
  upstream.closeAllConnections();
  await new Promise(resolve => upstream.close(resolve));
  if (fixture) await rm(fixture, { recursive: true, force: true });
});

test('forwards ordinary request body and authorization through the real rewrite', async () => {
  const response = await fetch(`${origin}/api/v1/fast`, {
    method: 'POST', headers: { authorization: 'Bearer proxy-test-only' }, body: '{"value":7}',
  });
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { ok: true });
  assert.deepEqual(records.find(row => row.path === '/api/v1/fast'), {
    path: '/api/v1/fast', method: 'POST', auth: 'Bearer proxy-test-only', body: '{"value":7}',
  });
});

test('preserves upstream error status and body', async () => {
  const response = await fetch(`${origin}/api/v1/unavailable`);
  assert.equal(response.status, 503);
  assert.deepEqual(await response.json(), { detail: 'controlled upstream unavailable' });
});

test('client cancellation closes the upstream connection', { timeout: 5000 }, async () => {
  const controller = new AbortController();
  const request = fetch(`${origin}/api/v1/abort`, { signal: controller.signal });
  const rejection = assert.rejects(request, { name: 'AbortError' });
  await abortedRequestArrived;
  controller.abort();
  await rejection;
  await abortedRequestClosed;
});

test('delivers a generation response after the former 30-second cutoff', { timeout: 45_000 }, async () => {
  const started = Date.now();
  const response = await fetch(`${origin}/api/v1/flashcards/generate`, {
    method: 'POST', body: '{"num_cards":5}', signal: AbortSignal.timeout(40_000),
  });
  assert.equal(response.status, 200, await response.clone().text());
  assert.deepEqual(await response.json(), { cards: [1, 2, 3, 4, 5] });
  assert.ok(Date.now() - started >= 31_000, 'must exercise the former timeout boundary');
});

test('preserves established SSE headers while first body bytes wait past 30 seconds', { timeout: 45_000 }, async () => {
  const started = Date.now();
  const response = await fetch(`${origin}/api/v1/silent-sse`, {
    method: 'POST', signal: AbortSignal.timeout(40_000),
  });
  assert.equal(response.status, 200);
  assert.ok(Date.now() - started < 5000, 'headers must precede delayed body bytes');
  assert.equal(await response.text(), 'data: {"choices":[{"delta":{"content":"BEEP BOOP"}}]}\n\ndata: [DONE]\n\n');
  assert.ok(Date.now() - started >= 31_000, 'must exercise established-response inactivity');
});
