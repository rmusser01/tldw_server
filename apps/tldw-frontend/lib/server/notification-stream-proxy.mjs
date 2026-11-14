import { request as httpRequest } from 'node:http';
import { request as httpsRequest } from 'node:https';
import { pipeline } from 'node:stream';
import { validateNetworkingConfig } from '../../scripts/validate-networking-config.mjs';

const HOP_HEADERS = [
  'connection',
  'keep-alive',
  'proxy-authenticate',
  'proxy-authorization',
  'te',
  'trailer',
  'transfer-encoding',
  'upgrade',
  'host',
];

function relayHeaders(headers) {
  const omitted = new Set([
    ...HOP_HEADERS,
    ...String(headers.connection || '')
      .split(',')
      .map((name) => name.trim().toLowerCase()),
  ]);
  return Object.fromEntries(
    Object.entries(headers).filter(([name, value]) => value !== undefined && !omitted.has(name))
  );
}

/**
 * Relay the existing authenticated stream and terminate both sides on failure.
 * @param {import('node:http').IncomingMessage} req
 * @param {import('node:http').ServerResponse} res
 */
export default async function proxyNotificationStream(req, res) {
  if (req.method !== 'GET') {
    res.setHeader('Allow', 'GET');
    res.statusCode = 405;
    res.end();
    return;
  }
  let networking;
  try {
    networking = validateNetworkingConfig();
  } catch {
    res.statusCode = 503;
    res.end();
    return;
  }
  if (networking.deploymentMode !== 'quickstart') {
    res.statusCode = 404;
    res.end();
    return;
  }
  const target = new URL('/api/v1/notifications/stream', networking.internalApiOrigin);
  target.search = new URL(req.url || '/', networking.internalApiOrigin).search;
  const headers = relayHeaders(req.headers);
  delete headers['content-length'];
  headers.host = target.host;
  headers['x-forwarded-host'] = req.headers.host || '';
  const request = target.protocol === 'https:' ? httpsRequest : httpRequest;

  await new Promise((resolve) => {
    let complete = false;
    let response;
    const finish = () => {
      if (complete) return;
      complete = true;
      req.off('aborted', abort);
      res.off('close', abort);
      resolve();
    };
    const upstream = request(target, { method: 'GET', headers, timeout: 180000 }, (incoming) => {
      response = incoming;
      res.statusCode = incoming.statusCode || 502;
      for (const [name, value] of Object.entries(relayHeaders(incoming.headers)))
        res.setHeader(name, value);
      // Plain pipe leaves the browser response open when its source aborts.
      pipeline(incoming, res, finish);
    });
    const abort = () => {
      response?.destroy();
      upstream.destroy();
      finish();
    };
    upstream.on('timeout', () => upstream.destroy());
    upstream.on('error', () => {
      if (!res.destroyed && !res.headersSent) {
        res.statusCode = 502;
        res.setHeader('Content-Type', 'application/json');
        res.setHeader('Cache-Control', 'no-store');
        res.end('{"detail":"Notification stream unavailable."}', finish);
      } else {
        res.destroy();
        finish();
      }
    });
    req.once('aborted', abort);
    res.once('close', abort);
    if (req.aborted || res.destroyed) {
      abort();
      return;
    }
    req.resume();
    upstream.end();
  });
}
