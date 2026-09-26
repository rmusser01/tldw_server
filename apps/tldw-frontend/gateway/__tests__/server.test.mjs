import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
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

test('keeps WebUI setup on Next and backend documentation assets on FastAPI', async (t) => {
  const { publicOrigin } = await fixture(t);
  for (const [path, role] of [
    ['/setup', 'next'],
    ['/setup?first=1', 'next'],
    ['/api/v1/setup/readiness/status', 'backend'],
    ['/docs-static/AuthNZ/AUTHNZ_USAGE_EXAMPLES.md', 'backend'],
    ['/static/favicon.ico', 'backend'],
  ]) {
    const response = await fetch(`${publicOrigin}${path}`);
    assert.equal((await response.json()).role, role, path);
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

for (const hostileHeader of ['origin', 'host']) {
  test(`returns HTTP 403 for a cookie upgrade with hostile ${hostileHeader} before either upstream`, async (t) => {
    const { publicOrigin, backend, next, seen } = await fixture(t);
    const upgrades = [];
    for (const upstream of [backend, next]) {
      upstream.on('upgrade', (_req, socket) => {
        upgrades.push(upstream);
        socket.destroy();
      });
    }
    const hostileValue = hostileHeader === 'origin' ? 'http://attacker.test' : 'attacker.test';
    const response = await new Promise((resolve, reject) => {
      const req = httpRequest(`${publicOrigin}/api/v1/mcp/ws`, {
        headers: {
          host: new URL(publicOrigin).host,
          origin: publicOrigin,
          cookie: 'session=legitimate-session',
          upgrade: 'websocket',
          connection: 'Upgrade',
          authorization: 'Bearer browser-credential',
          'x-tldw-gateway-hop': 'forged-hop',
          'x-forwarded-for': '203.0.113.5',
          [hostileHeader]: hostileValue,
        },
      }, (res) => {
        const chunks = [];
        res.on('data', (chunk) => chunks.push(chunk));
        res.on('end', () => resolve({ status: res.statusCode, headers: res.headers,
          body: Buffer.concat(chunks).toString() }));
      });
      const timeout = setTimeout(() => req.destroy(new Error('upgrade refusal timed out')), 2_000);
      req.once('close', () => clearTimeout(timeout));
      req.on('upgrade', (_res, socket) => {
        socket.destroy();
        reject(new Error('unauthorized upgrade received HTTP 101'));
      });
      req.on('error', reject);
      req.end();
    });

    assert.equal(response.status, 403);
    assert.equal(response.body, 'Forbidden');
    assert.equal(response.headers['set-cookie'], undefined);
    for (const name of Object.keys(response.headers)) {
      assert.equal(/^(?:authorization|cookie|proxy-authenticate|www-authenticate|forwarded|x-forwarded-.*|x-tldw-gateway-.*)$/i.test(name), false);
    }
    const serialized = JSON.stringify(response);
    for (const value of [hostileValue, 'legitimate-session', 'browser-credential', 'forged-hop', '203.0.113.5', HOP_SECRET]) {
      assert.equal(serialized.includes(value), false);
    }
    assert.equal(upgrades.length, 0);
    assert.equal(seen.length, 0);
  });
}

for (const peerCloses of [false, true]) {
  test(`bounds forbidden upgrade cleanup when the peer ${peerCloses ? 'closes' : 'stays open'}`, { timeout: 4_000 }, async (t) => {
    const { publicOrigin, gateway } = await fixture(t);
    const originalSetTimeout = globalThis.setTimeout;
    let backstop;
    let backstopFired = false;
    t.mock.method(globalThis, 'setTimeout', (callback, delay, ...args) => {
      if (delay !== 1_000) return originalSetTimeout(callback, delay, ...args);
      backstop = originalSetTimeout(() => {
        backstopFired = true;
        callback(...args);
      }, delay);
      return backstop;
    });
    const parsed = new URL(publicOrigin);
    const socket = connect({ port: Number(parsed.port), host: parsed.hostname,
      allowHalfOpen: !peerCloses });
    let traffic;
    t.after(() => { clearInterval(traffic); socket.destroy(); });
    await once(socket, 'connect');
    const upgraded = once(gateway, 'upgrade');
    const ended = once(socket, 'end');
    let response = '';
    socket.on('data', (chunk) => { response += chunk.toString(); });
    const started = performance.now();
    socket.write(`GET /api/v1/mcp/ws HTTP/1.1\r\nHost: ${parsed.host}\r\nOrigin: http://attacker.test\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n`);
    const [, gatewaySocket] = await upgraded;
    t.after(() => gatewaySocket.destroy());
    const closed = once(gatewaySocket, 'close');
    assert.equal(backstop.hasRef(), false);
    await ended;
    assert.equal(response, 'HTTP/1.1 403 Forbidden\r\nConnection: close\r\nContent-Length: 9\r\n\r\nForbidden');
    if (!peerCloses) {
      traffic = setInterval(() => socket.write('peer traffic cannot extend the deadline'), 100);
    }
    await closed;
    const elapsed = performance.now() - started;
    if (peerCloses) {
      await new Promise((resolve) => originalSetTimeout(resolve, 1_100));
      assert.equal(backstopFired, false);
    } else {
      assert.equal(backstopFired, true);
      assert.equal(elapsed >= 900 && elapsed < 2_000, true);
    }
  });
}

test('survives a peer reset after a forbidden upgrade and closes that socket', { timeout: 6_000 }, async (t) => {
  const script = `
    import assert from 'node:assert/strict';
    import { once } from 'node:events';
    import { connect } from 'node:net';
    import { createGateway } from ${JSON.stringify(new URL('../server.mjs', import.meta.url).href)};
    const gateway = createGateway({
      backendOrigin: 'http://127.0.0.1:8000', nextOrigin: 'http://127.0.0.1:3000',
      publicHost: '127.0.0.1', publicPort: 0,
      gatewayHopSecret: 'gateway-hop-test-secret-at-least-32-characters',
    });
    gateway.listen(0, '127.0.0.1');
    await once(gateway, 'listening');
    const port = gateway.address().port;
    const peer = connect({ port, host: '127.0.0.1', allowHalfOpen: true });
    await once(peer, 'connect');
    const upgraded = once(gateway, 'upgrade');
    const ended = once(peer, 'end');
    let refusal = '';
    peer.on('data', (chunk) => { refusal += chunk.toString(); });
    peer.write('GET /api/v1/mcp/ws HTTP/1.1\\r\\nHost: 127.0.0.1:' + port +
      '\\r\\nOrigin: http://attacker.test\\r\\nCookie: session=legitimate-session' +
      '\\r\\nUpgrade: websocket\\r\\nConnection: Upgrade\\r\\n\\r\\n');
    const [, deniedSocket] = await upgraded;
    // Do not add an error listener here: the gateway must handle the real reset.
    const emit = deniedSocket.emit;
    let resetCode;
    deniedSocket.emit = function (event, ...args) {
      if (event === 'error') resetCode = args[0].code;
      return emit.call(this, event, ...args);
    };
    const closed = new Promise((resolve) => deniedSocket.once('close', resolve));
    await ended;
    assert.equal(refusal.startsWith('HTTP/1.1 403 Forbidden'), true);
    peer.resetAndDestroy();
    const hadError = await closed;
    assert.equal(resetCode, 'ECONNRESET');
    assert.equal(hadError, true);
    assert.equal(deniedSocket.destroyed, true);
    const status = await fetch('http://127.0.0.1:' + port + '/_tldw/status');
    assert.equal(status.status, 200);
    assert.deepEqual(await status.json(), { phase: 'ready', ready: true });
    gateway.closeAllConnections();
    await new Promise((resolve) => gateway.close(resolve));
    console.log('gateway survived reset and closed refused socket');
  `;
  const child = spawn(process.execPath, ['--input-type=module', '-e', script]);
  t.after(() => child.kill());
  let stdout = '';
  let stderr = '';
  child.stdout.on('data', (chunk) => { stdout += chunk.toString(); });
  child.stderr.on('data', (chunk) => { stderr += chunk.toString(); });
  const timeout = setTimeout(() => child.kill(), 4_000);
  t.after(() => clearTimeout(timeout));
  const [code] = await once(child, 'close');

  assert.equal(code, 0, stderr);
  assert.equal(stdout.trim(), 'gateway survived reset and closed refused socket');
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

test('injects the private setup capability only on canonical setup API paths', async (t) => {
  const { publicOrigin } = await fixture(t);
  for (const path of ['/api/v1/setup', '/api/v1/setup/first-run/metadata', '/api/v1/setup/config?first=1']) {
    const response = await fetch(`${publicOrigin}${path}`, {
      headers: { origin: publicOrigin, 'x-tldw-gateway-hop': 'forged',
        forwarded: 'for=attacker', 'x-real-ip': '8.8.8.8', 'x-forwarded-for': '8.8.8.8',
        'x-forwarded-host': 'attacker', 'x-forwarded-prefix': '/evil' },
    });
    const record = await response.json();
    assert.equal(record.headers['x-tldw-gateway-hop'], HOP_SECRET);
    assert.equal(record.headers['x-forwarded-for'], '127.0.0.1');
    assert.equal(record.headers['x-forwarded-host'], new URL(publicOrigin).host);
    assert.equal(record.headers['x-forwarded-port'], new URL(publicOrigin).port);
    assert.equal(record.headers['x-forwarded-proto'], 'http');
    assert.equal(record.headers.forwarded, undefined);
    assert.equal(record.headers['x-real-ip'], undefined);
    assert.equal(record.headers['x-forwarded-prefix'], undefined);
  }
  const duplicate = await rawRequest(`${publicOrigin}/api/v1/setup/first-run/state`, {
    'x-tldw-gateway-hop': ['forged', 'again'], 'x-forwarded-for': ['8.8.8.8', '9.9.9.9'],
  });
  const cleaned = JSON.parse(duplicate.body).headers;
  assert.equal(cleaned['x-tldw-gateway-hop'], HOP_SECRET);
  assert.equal(cleaned['x-forwarded-for'], '127.0.0.1');
  for (const path of ['/api/v1/media', '/api/v1/setup-other', '/api/v1/setup/%2fconfig', '/api/v1/setup//config']) {
    const response = await rawRequest(`${publicOrigin}${path}`, { 'x-tldw-gateway-hop': 'forged' });
    assert.equal(JSON.parse(response.body).headers['x-tldw-gateway-hop'], undefined);
  }
});

test('injects the managed hop only for canonical MCP HTTP and WebSocket paths', async (t) => {
  const { publicOrigin, backend } = await fixture(t);
  for (const path of ['/api/v1/mcp/status', '/api/v1/mcp/tools/execute?foo=bar']) {
    const response = await fetch(`${publicOrigin}${path}`, {
      headers: { origin: publicOrigin, 'x-tldw-gateway-hop': 'forged' },
    });
    assert.equal((await response.json()).headers['x-tldw-gateway-hop'], HOP_SECRET);
  }
  for (const path of ['/api/v1/mcp-other', '/api/v1/mcp//ws', '/api/v1/mcp/%77s']) {
    const response = await rawRequest(`${publicOrigin}${path}`, { 'x-tldw-gateway-hop': 'forged' });
    assert.equal(JSON.parse(response.body).headers['x-tldw-gateway-hop'], undefined);
  }
  let upstreamHeaders;
  backend.on('upgrade', (req, socket) => {
    upstreamHeaders = req.headers;
    socket.end('HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n');
  });
  const parsed = new URL(publicOrigin);
  const socket = connect(Number(parsed.port), parsed.hostname);
  t.after(() => socket.destroy());
  await once(socket, 'connect');
  socket.write(`GET /api/v1/mcp/ws HTTP/1.1\r\nHost: ${parsed.host}\r\nOrigin: ${publicOrigin}\r\nCookie: session=legitimate-session\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nX-Tldw-Gateway-Hop: forged\r\nX-Forwarded-For: 8.8.8.8\r\n\r\n`);
  const [chunk] = await once(socket, 'data');
  assert.match(chunk.toString(), /^HTTP\/1\.1 101 Switching Protocols/);
  assert.equal(upstreamHeaders['x-tldw-gateway-hop'], HOP_SECRET);
  assert.equal(upstreamHeaders['x-forwarded-for'], '127.0.0.1');
  assert.equal(upstreamHeaders.cookie, 'session=legitimate-session');
  assert.equal(chunk.toString().includes(HOP_SECRET), false);
});
