// Node-only integration tests: real sockets, the production relay and synthetic auth.
import assert from 'node:assert/strict';
import { createServer, get } from 'node:http';
import { once } from 'node:events';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { test } from 'node:test';
const routePath = fileURLToPath(
  new URL('../../lib/server/notification-stream-proxy.mjs', import.meta.url)
);
const frame = 'event: ready\ndata: {"ready":true}\n\n';

async function bounded(promise, milliseconds) {
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((resolve) => {
        timer = setTimeout(() => resolve(false), milliseconds);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}
async function listen(server) {
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  return server.address().port;
}

async function exercise(action, cacheControl = 'no-cache') {
  const keys = [
    'NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE',
    'TLDW_INTERNAL_API_ORIGIN',
    'NEXT_PUBLIC_API_URL',
    'SINGLE_USER_API_KEY',
  ];
  const oldEnv = Object.fromEntries(keys.map((key) => [key, process.env[key]]));
  let upstreamResponse, upstreamClosed, client, response, proxy;
  const closed = new Promise((resolve) => {
    upstreamClosed = resolve;
  });
  const records = [];
  const upstream = createServer((req, res) => {
    records.push({
      method: req.method,
      path: req.url,
      auth: req.headers.authorization,
      cursor: req.headers['last-event-id'],
      key: req.headers['x-api-key'],
      cookie: req.headers.cookie,
      hop: req.headers['x-request-hop'],
      host: req.headers.host,
      forwarded: req.headers['x-forwarded-host'],
    });
    res.once('close', () => upstreamClosed(true));
    if (req.headers.authorization !== 'Bearer c09-proxy-fixture') {
      res.writeHead(401, { 'Content-Type': 'application/json' });
      res.end('{"detail":"unauthorized"}');
      return;
    }
    upstreamResponse = res;
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': cacheControl,
      'X-Accel-Buffering': 'no',
      ...(action === 'hop-headers'
        ? { Connection: 'x-backend-hop', 'X-Backend-Hop': 'must-not-forward' }
        : {}),
    });
    res.write(frame);
  });
  const pending = new Set();
  try {
    const backendPort = await listen(upstream);
    Object.assign(process.env, {
      NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: 'quickstart',
      TLDW_INTERNAL_API_ORIGIN: `http://127.0.0.1:${backendPort}`,
      NEXT_PUBLIC_API_URL: '',
      SINGLE_USER_API_KEY: 'test-only-must-not-inject',
    });
    if (action === 'advanced') {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = 'advanced';
      process.env.NEXT_PUBLIC_API_URL = 'https://unused.invalid';
    }
    if (action === 'invalid-origin') process.env.TLDW_INTERNAL_API_ORIGIN += '/must-not-proxy';
    if (action === 'unavailable') await new Promise((done) => upstream.close(done));
    const { default: handler } = await import(pathToFileURL(routePath).href);
    proxy = createServer((req, res) => {
      const work = handler(req, res);
      pending.add(work);
      work.catch(() => {}).finally(() => pending.delete(work));
    });
    const webPort = await listen(proxy);
    const received = new Promise((resolveResponse, reject) => {
      const options = {
        method: action === 'method' ? 'POST' : 'GET',
        headers: {
          'Last-Event-ID': '41',
          ...(action === 'deny' ? {} : { Authorization: 'Bearer c09-proxy-fixture' }),
          ...(action === 'hop-headers'
            ? {
                Connection: 'x-request-hop',
                'X-Request-Hop': 'must-not-forward',
                Cookie: 'fixture-session=valid',
              }
            : {}),
        },
      };
      client = get(
        `http://127.0.0.1:${webPort}/api/v1/notifications/stream?after=42`,
        options,
        (res) => {
          response = res;
          resolveResponse(res);
        }
      );
      client.on('error', reject);
    });
    assert(await bounded(received, 2000), 'Real response required');
    const expected =
      { method: 405, advanced: 404, 'invalid-origin': 503, unavailable: 502, deny: 401 }[action] ||
      200;
    assert.equal(response.statusCode, expected);
    if (expected === 200) {
      assert.equal(response.headers['content-type'], 'text/event-stream');
      assert.equal(response.headers['cache-control'], cacheControl);
      assert.equal(response.headers['x-accel-buffering'], 'no');
      assert.equal(response.headers['x-backend-hop'], undefined);
    }
    if (action === 'method') assert.equal(response.headers.allow, 'GET');
    if ([405, 404, 503, 502].includes(expected)) assert.deepEqual(records, []);
    else
      assert.deepEqual(records, [
        {
          method: 'GET',
          path: '/api/v1/notifications/stream?after=42',
          auth: action === 'deny' ? undefined : 'Bearer c09-proxy-fixture',
          cursor: '41',
          key: undefined,
          cookie: action === 'hop-headers' ? 'fixture-session=valid' : undefined,
          hop: undefined,
          host: `127.0.0.1:${backendPort}`,
          forwarded: `127.0.0.1:${webPort}`,
        },
      ]);
    const chunks = [];
    const terminal = new Promise((done) => {
      response.once('end', () => done('end'));
      response.once('error', () => done('error'));
      response.once('aborted', () => done('aborted'));
    });
    response.on('data', (chunk) => chunks.push(chunk));
    response.resume();
    if (action === 'normal' || action === 'hop-headers') upstreamResponse.end();
    if (action === 'upstream-abort') upstreamResponse.destroy();
    if (action === 'client-cancel') {
      response.destroy();
      client.destroy();
    }
    if (action === 'client-cancel')
      assert(await bounded(closed, 2000), 'Browser cancellation must release upstream');
    else {
      const ended = await bounded(terminal, 2000);
      assert(ended, 'Upstream termination must terminate the browser response');
      assert.equal(response.complete, action !== 'upstream-abort');
      if (action !== 'upstream-abort') {
        assert.equal(ended, 'end');
        assert.equal(
          Buffer.concat(chunks).toString('utf8'),
          expected === 200
            ? frame
            : expected === 401
              ? '{"detail":"unauthorized"}'
              : expected === 502
                ? '{"detail":"Notification stream unavailable."}'
                : ''
        );
      }
    }
  } finally {
    try {
      response?.destroy();
      client?.destroy();
      proxy?.closeAllConnections();
      upstream.closeAllConnections();
      await Promise.all([
        proxy ? new Promise((done) => proxy.close(done)) : undefined,
        new Promise((done) => upstream.close(done)),
      ]);
      assert(
        await bounded(
          Promise.allSettled([...pending]).then(() => true),
          2000
        ),
        'Relay task release required'
      );
    } finally {
      for (const [key, value] of Object.entries(oldEnv)) {
        if (value === undefined) delete process.env[key];
        else process.env[key] = value;
      }
    }
  }
}

for (const action of [
  'normal',
  'upstream-abort',
  'client-cancel',
  'deny',
  'method',
  'advanced',
  'invalid-origin',
  'unavailable',
  'hop-headers',
]) {
  test(`notification relay: ${action}`, { timeout: 10000 }, () => exercise(action));
}
test('notification relay: upstream abort with no-transform', { timeout: 10000 }, () =>
  exercise('upstream-abort', 'no-cache, no-transform')
);
