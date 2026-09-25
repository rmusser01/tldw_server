import { createServer } from 'node:http';
import { fileURLToPath } from 'node:url';

import { createProxyMiddleware } from 'http-proxy-middleware';

import { routeForPath } from './routes.mjs';

const PHASES = new Set(['starting', 'ready', 'maintenance', 'error']);

const canonicalOrigin = (value, label) => {
  const parsed = new URL(value);
  if (
    !['http:', 'https:'].includes(parsed.protocol) ||
    parsed.username || parsed.password || parsed.pathname !== '/' ||
    parsed.search || parsed.hash ||
    (value !== parsed.origin && value !== `${parsed.origin}/`)
  ) throw new TypeError(`${label} must be a canonical HTTP(S) origin`);
  return parsed.origin;
};

const requestPath = (req) => {
  const raw = req.url;
  if (typeof raw !== 'string' || !raw.startsWith('/') || raw.startsWith('//') || raw.includes('#')) {
    return null;
  }
  const pathname = raw.split('?', 1)[0];
  try {
    return new URL(raw, 'http://gateway.invalid').pathname === pathname ? pathname : null;
  } catch {
    return null;
  }
};

const singleHeader = (value) => typeof value === 'string' ? value : null;

const untrustedControlHeader = (name) =>
  name.startsWith('x-middleware-') ||
  name.startsWith('x-invoke-') ||
  name.startsWith('x-nextjs-rewritten-') ||
  name.startsWith('x-tldw-gateway-') ||
  name === 'x-original-url' ||
  name === 'x-rewrite-url' ||
  name === 'x-now-route-matches';

export const authorizeRequest = (req, { publicHost, publicPort }) => {
  if (Array.isArray(req.rawHeaders)) {
    const names = req.rawHeaders.filter((_value, index) => index % 2 === 0)
      .map((name) => name.toLowerCase());
    if (names.filter((name) => name === 'host').length !== 1 ||
        names.filter((name) => name === 'origin').length > 1) return false;
  }
  const authority = new URL(`http://${publicHost}:${publicPort}`).host;
  const host = singleHeader(req.headers.host);
  if (!host || host.toLowerCase() !== authority.toLowerCase()) return false;
  const origin = req.headers.origin;
  if (origin === undefined) return true;
  return singleHeader(origin) === `http://${authority}`;
};

const normalizeHeaders = (req, { host, port, hopSecret, next }) => {
  for (const name of Object.keys(req.headers)) {
    if (
      name === 'forwarded' || name === 'x-real-ip' ||
      name.startsWith('x-forwarded-') || untrustedControlHeader(name)
    ) delete req.headers[name];
  }
  req.headers['x-forwarded-for'] = String(req.socket.remoteAddress || '');
  req.headers['x-forwarded-host'] = host;
  req.headers['x-forwarded-port'] = String(port);
  req.headers['x-forwarded-proto'] = 'http';
  if (next) req.headers['x-tldw-gateway-hop'] = hopSecret;
};

const send = (res, status, body, contentType = 'text/plain; charset=utf-8') => {
  res.writeHead(status, {
    'content-type': contentType,
    'cache-control': 'no-store',
    'x-content-type-options': 'nosniff',
  });
  res.end(body);
};

const managedResponse = (req, res, pathname, phase) => {
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    res.setHeader('allow', 'GET, HEAD');
    send(res, 405, 'Method not allowed');
    return;
  }
  if (pathname === '/_tldw/status') {
    send(res, 200, JSON.stringify({ phase, ready: phase === 'ready' }), 'application/json; charset=utf-8');
    return;
  }
  if (pathname === '/_tldw' || pathname === '/_tldw/') {
    send(res, 200,
      `<!doctype html><title>tldw status</title><h1>Application ${phase}</h1><p>For Docker updates, run the authenticated host helper from the downloaded bundle.</p>`,
      'text/html; charset=utf-8');
    return;
  }
  send(res, 404, 'Not found');
};

const proxyError = (_error, _req, res) => {
  if (typeof res.writeHead !== 'function') {
    res.destroy();
    return;
  }
  if (!res.headersSent) send(res, 502, 'Upstream unavailable');
  else res.destroy();
};

export function createGateway({ backendOrigin, nextOrigin, publicHost, publicPort,
  gatewayHopSecret, phase = 'ready' }) {
  const backendTarget = canonicalOrigin(backendOrigin, 'backendOrigin');
  const nextTarget = canonicalOrigin(nextOrigin, 'nextOrigin');
  if (!['127.0.0.1', 'localhost', '[::1]'].includes(publicHost)) {
    throw new TypeError('publicHost must be loopback');
  }
  if (!Number.isInteger(publicPort) || publicPort < 0 || publicPort > 65535) {
    throw new TypeError('publicPort must be a valid port');
  }
  if (typeof gatewayHopSecret !== 'string' || gatewayHopSecret.length < 32 || /\s/.test(gatewayHopSecret)) {
    throw new TypeError('gatewayHopSecret must be a strong opaque token');
  }
  const readPhase = () => typeof phase === 'function' ? phase() : phase;
  const proxyFor = (target) => createProxyMiddleware({
    target,
    changeOrigin: false,
    autoRewrite: true,
    xfwd: false,
    on: { error: proxyError },
  });
  const backendProxy = proxyFor(backendTarget);
  const nextProxy = proxyFor(nextTarget);

  const server = createServer((req, res) => {
    const port = publicPort || server.address()?.port;
    const host = new URL(`http://${publicHost}:${port}`).host;
    if (!authorizeRequest(req, { publicHost, publicPort: port })) {
      send(res, 403, 'Forbidden');
      return;
    }
    const pathname = requestPath(req);
    if (!pathname) {
      send(res, 400, 'Bad request');
      return;
    }
    const currentPhase = readPhase();
    if (!PHASES.has(currentPhase)) {
      send(res, 503, 'Application unavailable');
      return;
    }
    const route = routeForPath(pathname);
    if (route === 'managed') {
      managedResponse(req, res, pathname, currentPhase);
      return;
    }
    if (currentPhase !== 'ready') {
      send(res, 503, 'Application is starting');
      return;
    }
    normalizeHeaders(req, { host, port, hopSecret: gatewayHopSecret, next: route === 'next' });
    (route === 'backend' ? backendProxy : nextProxy)(req, res);
  });

  server.on('upgrade', (req, socket, head) => {
    const port = publicPort || server.address()?.port;
    const host = new URL(`http://${publicHost}:${port}`).host;
    const pathname = requestPath(req);
    if (!authorizeRequest(req, { publicHost, publicPort: port }) || !pathname || readPhase() !== 'ready') {
      socket.destroy();
      return;
    }
    const route = routeForPath(pathname);
    if (route === 'managed') {
      socket.destroy();
      return;
    }
    normalizeHeaders(req, { host, port, hopSecret: gatewayHopSecret, next: route === 'next' });
    (route === 'backend' ? backendProxy : nextProxy).upgrade(req, socket, head);
  });

  return server;
}

export const gatewayRuntimeFromEnv = (env) => {
  const port = (value, label) => {
    if (!/^\d{1,5}$/.test(String(value))) throw new TypeError(`${label} must be a port`);
    const parsed = Number(value);
    if (parsed < 1 || parsed > 65535) throw new TypeError(`${label} must be a port`);
    return parsed;
  };
  const publicHost = env.TLDW_PUBLIC_HOST || '127.0.0.1';
  const publicPort = port(env.TLDW_PUBLIC_PORT || '8080', 'TLDW_PUBLIC_PORT');
  const configuredListenHost = env.TLDW_GATEWAY_LISTEN_HOST || publicHost;
  const listenHost = configuredListenHost === '[::1]' ? '::1' : configuredListenHost;
  const listenPort = port(env.TLDW_GATEWAY_LISTEN_PORT || String(publicPort), 'TLDW_GATEWAY_LISTEN_PORT');
  if (!['127.0.0.1', 'localhost', '::1', '0.0.0.0', '::'].includes(listenHost)) {
    throw new TypeError('TLDW_GATEWAY_LISTEN_HOST must be a local interface');
  }
  return { publicHost, publicPort, listenHost, listenPort };
};

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const runtime = gatewayRuntimeFromEnv(process.env);
  const gateway = createGateway({
    backendOrigin: process.env.TLDW_INTERNAL_API_ORIGIN,
    nextOrigin: process.env.TLDW_INTERNAL_WEBUI_ORIGIN,
    publicHost: runtime.publicHost,
    publicPort: runtime.publicPort,
    gatewayHopSecret: process.env.TLDW_GATEWAY_HOP_SECRET,
    phase: process.env.TLDW_GATEWAY_PHASE || 'ready',
  });
  gateway.listen(runtime.listenPort, runtime.listenHost);
}
