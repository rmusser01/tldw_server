import assert from 'node:assert/strict';
import { once } from 'node:events';
import { createServer, request as httpRequest } from 'node:http';
import { connect } from 'node:net';
import test from 'node:test';

import { authorizeRequest, createGateway, gatewayRuntimeFromEnv } from '../server.mjs';

const HOP_SECRET = 'gateway-hop-test-secret-at-least-32-characters';

const listen = async (server) => {
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  return `http://127.0.0.1:${server.address().port}`;
};

const close = async (server) => {
  server.closeAllConnections?.();
  await new Promise((resolve) => server.close(resolve));
};

const echoServer = (role, seen) => createServer(async (req, res) => {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const record = { role, url: req.url, method: req.method, headers: req.headers,
    body: Buffer.concat(chunks).toString() };
  seen.push(record);
  res.setHeader('content-type', 'application/json');
  res.end(JSON.stringify(record));
});

const fixture = async (t, options = {}) => {
  const { backendHandler, nextHandler, ...gatewayOptions } = options;
  const seen = [];
  const backend = backendHandler ? createServer(backendHandler) : echoServer('backend', seen);
  const next = nextHandler ? createServer(nextHandler) : echoServer('next', seen);
  const backendOrigin = await listen(backend);
  const nextOrigin = await listen(next);
  const gateway = createGateway({
    backendOrigin,
    nextOrigin,
    publicHost: '127.0.0.1',
    publicPort: 0,
    gatewayHopSecret: HOP_SECRET,
    phase: 'ready',
    ...gatewayOptions,
  });
  const publicOrigin = await listen(gateway);
  t.after(async () => {
    await close(gateway);
    await close(next);
    await close(backend);
  });
  return { publicOrigin, backendOrigin, nextOrigin, seen, gateway, backend, next };
};

const rawRequest = (url, headers = {}) => new Promise((resolve, reject) => {
  const parsed = new URL(url);
  const req = httpRequest(parsed, { headers }, (res) => {
    const chunks = [];
    res.on('data', (chunk) => chunks.push(chunk));
    res.on('end', () => resolve({ status: res.statusCode, headers: res.headers,
      body: Buffer.concat(chunks).toString() }));
  });
  req.on('error', reject);
  req.end();
});

test('rejects duplicate authority headers before route selection', () => {
  assert.equal(authorizeRequest({
    headers: { host: '127.0.0.1:8080' },
    rawHeaders: ['Host', '127.0.0.1:8080', 'Host', 'attacker.test'],
  }, { publicHost: '127.0.0.1', publicPort: 8080 }), false);
});

test('keeps the Docker listener separate from the stable public origin', () => {
  assert.deepEqual(gatewayRuntimeFromEnv({
    TLDW_PUBLIC_HOST: '127.0.0.1',
    TLDW_PUBLIC_PORT: '18080',
    TLDW_GATEWAY_LISTEN_HOST: '0.0.0.0',
    TLDW_GATEWAY_LISTEN_PORT: '8080',
  }), {
    publicHost: '127.0.0.1',
    publicPort: 18080,
    listenHost: '0.0.0.0',
    listenPort: 8080,
  });
  assert.equal(gatewayRuntimeFromEnv({ TLDW_PUBLIC_HOST: '[::1]' }).listenHost, '::1');
});

test('preserves paths, queries and bodies while stripping browser forwarding credentials', async (t) => {
  const { publicOrigin, seen } = await fixture(t);
  const response = await fetch(`${publicOrigin}/api/v1/media/process?tag=a%2Fb`, {
    method: 'POST',
    headers: {
      origin: publicOrigin,
      'content-type': 'multipart/form-data; boundary=fixture',
      forwarded: 'for=attacker',
      'x-forwarded-for': '203.0.113.5',
      'x-tldw-gateway-hop': 'browser-supplied',
    },
    body: '--fixture\r\npart\r\n--fixture--',
  });
  const record = await response.json();

  assert.equal(response.status, 200);
  assert.equal(record.role, 'backend');
  assert.equal(record.url, '/api/v1/media/process?tag=a%2Fb');
  assert.equal(record.body, '--fixture\r\npart\r\n--fixture--');
  assert.equal(record.headers.forwarded, undefined);
  assert.equal(record.headers['x-tldw-gateway-hop'], undefined);
  assert.notEqual(record.headers['x-forwarded-for'], '203.0.113.5');
  assert.equal(seen.length, 1);
});

test('routes Next APIs and assets through one private authenticated hop', async (t) => {
  const { publicOrigin } = await fixture(t);
  for (const path of ['/api/_tldw-webui/session', '/api/documentation/page', '/_next/static/app.js']) {
    const response = await fetch(`${publicOrigin}${path}`, {
      headers: {
        'x-tldw-gateway-hop': 'browser-supplied',
        'x-middleware-subrequest': 'browser-supplied',
        'x-invoke-path': '/admin',
      },
    });
    const record = await response.json();
    assert.equal(record.role, 'next');
    assert.equal(record.url, path);
    assert.equal(record.headers['x-tldw-gateway-hop'], HOP_SECRET);
    assert.equal(record.headers['x-middleware-subrequest'], undefined);
    assert.equal(record.headers['x-invoke-path'], undefined);
  }
});

test('rejects unknown Host and Origin before either upstream', async (t) => {
  const { publicOrigin, seen } = await fixture(t);
  const wrongHost = await rawRequest(`${publicOrigin}/api/v1/health`, { host: 'attacker.test' });
  const wrongOrigin = await rawRequest(`${publicOrigin}/api/v1/health`, {
    origin: 'http://attacker.test',
  });

  assert.equal(wrongHost.status, 403);
  assert.equal(wrongOrigin.status, 403);
  assert.equal(seen.length, 0);
});

test('serves bounded read-only maintenance status without contacting either upstream', async (t) => {
  const { publicOrigin, seen } = await fixture(t, { phase: 'maintenance' });
  const status = await fetch(`${publicOrigin}/_tldw/status`);
  const page = await fetch(`${publicOrigin}/settings`);
  const mutation = await fetch(`${publicOrigin}/_tldw/update`, { method: 'POST',
    headers: { origin: publicOrigin } });

  assert.deepEqual(await status.json(), { phase: 'maintenance', ready: false });
  assert.equal(page.status, 503);
  assert.equal(mutation.status, 405);
  assert.equal(seen.length, 0);
});

test('keeps multiple Set-Cookie headers and rewrites private redirect origins', async (t) => {
  const { publicOrigin } = await fixture(t, {
    backendHandler: (_req, res) => {
      res.writeHead(307, {
        location: `http://127.0.0.1:${res.socket.localPort}/api/v1/final`,
        'set-cookie': ['session=a; Path=/api; HttpOnly', 'csrf=b; Path=/'],
      });
      res.end();
    },
  });
  const response = await fetch(`${publicOrigin}/api/v1/redirect`, { redirect: 'manual' });

  assert.equal(response.status, 307);
  assert.equal(response.headers.get('location'), `${publicOrigin}/api/v1/final`);
  assert.deepEqual(response.headers.getSetCookie(), [
    'session=a; Path=/api; HttpOnly', 'csrf=b; Path=/',
  ]);
});

test('streams the first SSE frame and closes the upstream after browser cancellation', async (t) => {
  let upstreamClosed;
  const closed = new Promise((resolve) => { upstreamClosed = resolve; });
  const { publicOrigin } = await fixture(t, {
    backendHandler: (_req, res) => {
      res.writeHead(200, { 'content-type': 'text/event-stream' });
      res.flushHeaders();
      res.write('data: first\n\n');
      res.once('close', upstreamClosed);
    },
  });
  const response = await fetch(`${publicOrigin}/api/v1/chat/completions`);
  const reader = response.body.getReader();
  const first = await reader.read();
  assert.equal(new TextDecoder().decode(first.value), 'data: first\n\n');
  await reader.cancel();
  await Promise.race([
    closed,
    new Promise((_resolve, reject) => setTimeout(() => reject(new Error('upstream stayed open')), 2_000)),
  ]);
});

test('proxies WebSocket upgrades through the authenticated backend route', async (t) => {
  const { publicOrigin, backend } = await fixture(t);
  let upstreamHeaders;
  backend.on('upgrade', (req, socket) => {
    upstreamHeaders = req.headers;
    socket.write('HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n');
    socket.end();
  });
  const parsed = new URL(publicOrigin);
  const socket = connect(Number(parsed.port), parsed.hostname);
  t.after(() => socket.destroy());
  await once(socket, 'connect');
  socket.write(`GET /api/v1/audio/stream/transcribe HTTP/1.1\r\nHost: ${parsed.host}\r\nOrigin: ${publicOrigin}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nX-Tldw-Gateway-Hop: browser-supplied\r\n\r\n`);
  const [chunk] = await once(socket, 'data');

  assert.match(chunk.toString(), /^HTTP\/1\.1 101 Switching Protocols/);
  assert.equal(upstreamHeaders['x-tldw-gateway-hop'], undefined);
});
