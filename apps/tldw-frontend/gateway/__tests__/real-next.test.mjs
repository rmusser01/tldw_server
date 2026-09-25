import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createServer } from 'node:http';
import { stat } from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import { setTimeout as delay } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';

import { createGateway } from '../server.mjs';

const enabled = process.env.TLDW_GATEWAY_REAL_NEXT === '1';
const frontend = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const standalone = path.join(frontend, '.next/standalone/apps/tldw-frontend/server.js');
const key = 'real-next-integration-key-123456789';
const hop = 'real-next-gateway-hop-secret-123456789';

const listen = async (server) => {
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  return `http://127.0.0.1:${server.address().port}`;
};

const reservePort = async () => {
  const server = createServer();
  const origin = await listen(server);
  await new Promise((resolve) => server.close(resolve));
  return Number(new URL(origin).port);
};

const waitForNext = async (origin, child) => {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (child.exitCode !== null) throw new Error('Next exited before readiness');
    try {
      const response = await fetch(`${origin}/api/hello`, { signal: AbortSignal.timeout(500) });
      if (response.status < 500) return;
    } catch {}
    await delay(100);
  }
  throw new Error('Next did not become ready');
};

test('built managed Next exchanges a cookie session only through the gateway',
  { skip: !enabled, timeout: 30_000 }, async (t) => {
    await stat(standalone);
    const backendRequests = [];
    const backend = createServer(async (req, res) => {
      backendRequests.push({ url: req.url, headers: req.headers });
      if (req.url === '/api/v1/auth/single-user/session') {
        res.setHeader('set-cookie', [
          'tldw_session_a1=opaque; Path=/api; HttpOnly; SameSite=Lax',
          'tldw_csrf_a1=csrf-value; Path=/; SameSite=Lax',
        ]);
        res.end(JSON.stringify({ authenticated: true }));
        return;
      }
      res.setHeader('content-type', 'application/json');
      res.end(JSON.stringify({ ok: true }));
    });
    const backendOrigin = await listen(backend);
    const nextPort = await reservePort();
    const nextOrigin = `http://127.0.0.1:${nextPort}`;
    const next = spawn(process.execPath, [standalone], {
      cwd: frontend,
      env: {
        ...process.env,
        HOSTNAME: '127.0.0.1',
        PORT: String(nextPort),
        NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: 'managed',
        AUTH_MODE: 'single_user',
        TLDW_WEBUI_EXPOSE_RUNTIME_AUTH: '1',
        TLDW_INTERNAL_API_ORIGIN: backendOrigin,
        TLDW_GATEWAY_HOP_SECRET: hop,
        SINGLE_USER_API_KEY: key,
        SINGLE_USER_SESSION_COOKIE_NAME: 'tldw_session_a1',
        CSRF_COOKIE_NAME: 'tldw_csrf_a1',
      },
      stdio: 'ignore',
    });
    t.after(async () => {
      next.kill('SIGTERM');
      await Promise.race([once(next, 'exit'), delay(2_000)]);
      backend.closeAllConnections();
      await new Promise((resolve) => backend.close(resolve));
    });
    await waitForNext(nextOrigin, next);

    const gateway = createGateway({
      backendOrigin,
      nextOrigin,
      publicHost: '127.0.0.1',
      publicPort: 0,
      gatewayHopSecret: hop,
    });
    const publicOrigin = await listen(gateway);
    t.after(async () => {
      gateway.closeAllConnections();
      await new Promise((resolve) => gateway.close(resolve));
    });

    const direct = await fetch(`${nextOrigin}/api/_tldw-webui/runtime-config`);
    const throughGateway = await fetch(`${publicOrigin}/api/_tldw-webui/runtime-config`);
    const session = await fetch(`${publicOrigin}/api/_tldw-webui/session`, {
      method: 'POST',
      headers: { origin: publicOrigin, 'sec-fetch-site': 'same-origin' },
    });
    const backendRoute = await fetch(`${publicOrigin}/api/v1/health`);

    assert.deepEqual((await direct.json()).runtimeAuth, { available: false });
    assert.deepEqual((await throughGateway.json()).runtimeAuth, {
      available: true,
      authMode: 'single-user',
      transport: 'cookie-session',
      csrfCookieName: 'tldw_csrf_a1',
    });
    assert.equal(session.status, 200);
    assert.equal(session.headers.getSetCookie().length, 2);
    assert.equal(backendRoute.status, 200);
    assert.equal(backendRequests[0].headers['x-api-key'], key);
    assert.equal(backendRequests[1].headers['x-api-key'], undefined);
    assert.equal(backendRequests[1].headers['x-tldw-gateway-hop'], undefined);
  });
