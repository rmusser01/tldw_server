import type { NextApiRequest, NextApiResponse } from 'next';
import { resolveRuntimeAuthPolicy } from './runtime-auth-policy';

const SESSION_PATH = '/api/v1/auth/single-user/session';
const BACKEND_TIMEOUT_MS = 5_000;

const cookieName = (value: string): string => {
  const separator = value.indexOf('=');
  return separator > 0 ? value.slice(0, separator).trim() : '';
};

const filteredRequestCookies = (
  value: string | undefined,
  allowedCookieNames: Set<string>
): string =>
  String(value || '')
    .split(';')
    .map((cookie) => cookie.trim())
    .filter((cookie) => allowedCookieNames.has(cookieName(cookie)))
    .join('; ');

const isExactSameOrigin = (req: NextApiRequest): boolean => {
  const origin = req.headers.origin;
  const host = req.headers.host;
  if (typeof origin !== 'string' || typeof host !== 'string') return false;

  try {
    const protocol = (req.socket as typeof req.socket & { encrypted?: boolean })?.encrypted
      ? 'https:'
      : 'http:';
    const parsedOrigin = new URL(origin);
    const expectedOrigin = new URL(`${protocol}//${host}`).origin;
    return origin === parsedOrigin.origin && parsedOrigin.origin === expectedOrigin;
  } catch {
    return false;
  }
};

const hasAcceptableFetchMetadata = (req: NextApiRequest): boolean => {
  const value = req.headers['sec-fetch-site'];
  return value === undefined || value === 'same-origin';
};

const safeResponseCookies = (headers: Headers, allowedCookieNames: Set<string>): string[] => {
  if (typeof headers.getSetCookie !== 'function') throw new TypeError('Invalid headers');
  const cookies = headers.getSetCookie();
  if (!Array.isArray(cookies)) throw new TypeError('Invalid Set-Cookie headers');
  if (cookies.some((cookie) => /[\r\n]/.test(cookie))) {
    throw new TypeError('Invalid Set-Cookie header');
  }
  return cookies.filter((cookie) => allowedCookieNames.has(cookieName(cookie)));
};

const copySafeResponseMetadata = (
  backend: Response,
  res: NextApiResponse,
  allowedCookieNames: Set<string>
): void => {
  const cookies = safeResponseCookies(backend.headers, allowedCookieNames);
  const contentType = backend.headers.get('content-type');
  const cacheControl = backend.headers.get('cache-control');

  if (cookies.length) res.setHeader('Set-Cookie', cookies);
  if (contentType) res.setHeader('Content-Type', contentType);
  if (cacheControl && /(?:^|,)\s*no-store\s*(?:,|$)/i.test(cacheControl)) {
    res.setHeader('Cache-Control', cacheControl);
  }
};

const isTimeout = (error: unknown): boolean =>
  typeof error === 'object' &&
  error !== null &&
  'name' in error &&
  (error.name === 'TimeoutError' || error.name === 'AbortError');

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  res.setHeader('Cache-Control', 'no-store, max-age=0');

  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    res.status(405).end();
    return;
  }

  if (!isExactSameOrigin(req) || !hasAcceptableFetchMetadata(req)) {
    res.status(403).end();
    return;
  }

  const policy = resolveRuntimeAuthPolicy(req);
  if (!policy.available) {
    res.status(503).end();
    return;
  }

  const allowedCookieNames = new Set([policy.sessionCookieName, 'csrf_token']);
  const headers: Record<string, string> = { 'X-API-KEY': policy.apiKey };
  const cookies = filteredRequestCookies(req.headers.cookie, allowedCookieNames);
  if (cookies) headers.Cookie = cookies;
  if (typeof req.headers['user-agent'] === 'string') {
    headers['User-Agent'] = req.headers['user-agent'];
  }

  let backend: Response | undefined;
  try {
    backend = await fetch(`${policy.internalApiOrigin}${SESSION_PATH}`, {
      method: 'POST',
      headers,
      redirect: 'manual',
      signal: AbortSignal.timeout(BACKEND_TIMEOUT_MS),
    });
    if (!Number.isInteger(backend.status) || backend.status < 200 || backend.status > 599) {
      throw new TypeError('Invalid backend status');
    }
    copySafeResponseMetadata(backend, res, allowedCookieNames);
    res.status(backend.status).end();
  } catch (error) {
    res.status(isTimeout(error) ? 504 : 502).end();
  } finally {
    await backend?.body?.cancel().catch(() => undefined);
  }
}
