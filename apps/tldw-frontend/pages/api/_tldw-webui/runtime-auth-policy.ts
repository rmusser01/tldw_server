import type { NextApiRequest } from 'next';

export type RuntimeAuthPolicy =
  | {
      available: true;
      apiKey: string;
      internalApiOrigin: string;
      sessionCookieName: string;
    }
  | { available: false; reason: string };

const DEFAULT_SESSION_COOKIE_NAME = 'tldw_single_user_session';
const COOKIE_NAME_PATTERN = /^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$/;
const RESERVED_COOKIE_PREFIX_PATTERN = /^__(?:Host|Http|Secure)-/i;

const PLACEHOLDER_KEYS = new Set([
  'change-me',
  'changeme',
  'change_me',
  'default',
  'test-key',
  'your-api-key',
  'your-api-key-here',
  'your_api_key',
  'your_api_key_here',
  'placeholder',
  'replace-me',
  'replace_me',
]);

const LOOPBACK_PEER_ADDRESSES = new Set(['127.0.0.1', '::1', '::ffff:127.0.0.1']);

const MIN_API_KEY_LENGTH = 16;

const normalizeEnvValue = (value?: string): string => String(value || '').trim();

export const getDeploymentMode = (): string =>
  normalizeEnvValue(process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE) || 'quickstart';

const isUsableApiKey = (value?: string): value is string => {
  if (!value || /\s/.test(value) || value.length < MIN_API_KEY_LENGTH) return false;
  const normalized = value.toLowerCase();
  return !normalized.startsWith('change_me') && !PLACEHOLDER_KEYS.has(normalized);
};

const extractHostname = (hostHeader?: string | string[]): string => {
  const host = Array.isArray(hostHeader) ? hostHeader[0] : hostHeader;
  const normalized = normalizeEnvValue(host).toLowerCase();
  if (!normalized) return '';
  if (normalized.startsWith('[') && normalized.includes(']')) {
    return normalized.slice(1, normalized.indexOf(']'));
  }
  if (normalized === '::1') return normalized;
  const colonCount = (normalized.match(/:/g) || []).length;
  return colonCount > 1 ? normalized : normalized.split(':')[0] || '';
};

const isLoopbackHost = (hostHeader?: string | string[]): boolean => {
  const hostname = extractHostname(hostHeader);
  return hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '::1';
};

const extractIPv4Address = (remoteAddress?: string): number[] | null => {
  const normalized = normalizeEnvValue(remoteAddress).toLowerCase();
  const address = normalized.startsWith('::ffff:')
    ? normalized.slice('::ffff:'.length)
    : normalized;
  const octets = address.split('.');
  if (octets.length !== 4) return null;

  const parts = octets.map((octet) => {
    if (!/^\d+$/.test(octet)) return Number.NaN;
    const value = Number(octet);
    return value >= 0 && value <= 255 ? value : Number.NaN;
  });
  return parts.every(Number.isInteger) ? parts : null;
};

const isTrustedLocalPeer = (remoteAddress?: string): boolean => {
  const normalized = normalizeEnvValue(remoteAddress).toLowerCase();
  if (LOOPBACK_PEER_ADDRESSES.has(normalized)) return true;

  const parts = extractIPv4Address(remoteAddress);
  if (!parts) return false;
  const [first, second, third, fourth] = parts;
  return (
    (first === 172 && second >= 16 && second <= 31 && third === 0 && fourth === 1) ||
    (first === 192 && second === 168 && third === 65 && fourth === 1)
  );
};

const hasForwardingHeaders = (req: NextApiRequest): boolean =>
  Object.keys(req.headers).some(
    (name) => name === 'forwarded' || name === 'x-real-ip' || name.startsWith('x-forwarded-')
  );

const resolvedSessionCookieName = (): string | null => {
  const configured = process.env.SINGLE_USER_SESSION_COOKIE_NAME;
  const value = configured === undefined ? DEFAULT_SESSION_COOKIE_NAME : configured;
  if (
    !COOKIE_NAME_PATTERN.test(value) ||
    value === 'csrf_token' ||
    RESERVED_COOKIE_PREFIX_PATTERN.test(value)
  ) {
    return null;
  }
  return value;
};

const validatedInternalOrigin = (): string | null => {
  const value = String(process.env.TLDW_INTERNAL_API_ORIGIN || '');
  try {
    const url = new URL(value);
    if (url.protocol !== 'http:' && url.protocol !== 'https:') return null;
    if (url.username || url.password || url.pathname !== '/' || url.search || url.hash) {
      return null;
    }
    const origin = url.origin;
    if (origin === 'null' || (value !== origin && value !== `${origin}/`)) return null;
    return origin;
  } catch {
    return null;
  }
};

export const resolveRuntimeAuthPolicy = (req: NextApiRequest): RuntimeAuthPolicy => {
  if (process.env.AUTH_MODE !== 'single_user') {
    return { available: false, reason: 'auth-mode' };
  }
  if (process.env.TLDW_WEBUI_EXPOSE_RUNTIME_AUTH !== '1') {
    return { available: false, reason: 'disabled' };
  }
  if (getDeploymentMode() !== 'quickstart') {
    return { available: false, reason: 'deployment-mode' };
  }
  if (!isLoopbackHost(req.headers.host)) return { available: false, reason: 'host' };
  if (!isTrustedLocalPeer(req.socket?.remoteAddress)) {
    return { available: false, reason: 'peer' };
  }
  if (hasForwardingHeaders(req)) return { available: false, reason: 'forwarded' };

  const apiKey = process.env.SINGLE_USER_API_KEY;
  if (!isUsableApiKey(apiKey)) return { available: false, reason: 'api-key' };

  const sessionCookieName = resolvedSessionCookieName();
  if (!sessionCookieName) return { available: false, reason: 'session-cookie-name' };

  const internalApiOrigin = validatedInternalOrigin();
  if (!internalApiOrigin) return { available: false, reason: 'internal-origin' };

  return { available: true, apiKey, internalApiOrigin, sessionCookieName };
};
