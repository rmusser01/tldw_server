export interface RequestHistoryItem {
  id: string;
  method: string;
  url: string;
  baseURL?: string;
  status?: number;
  ok?: boolean;
  duration_ms?: number;
  timestamp: string;
  requestHeaders?: Record<string, string>;
  requestBody?: unknown;
  responseBody?: unknown;
  errorMessage?: string;
}

const KEY = 'tldw-request-history';
const MAX = 200;

const REDACTED = '[REDACTED]';

// Header names (compared case-insensitively) whose values must never be
// persisted to localStorage. These carry credentials or anti-CSRF secrets.
const SENSITIVE_HEADERS = new Set([
  'authorization',
  'cookie',
  'proxy-authorization',
  'set-cookie',
  'x-api-key',
  'x-auth-token',
  'x-csrf-token',
  'x-tldw-org-id',
]);

// Body keys (compared case-insensitively) whose values must never be persisted.
// Stored lower-cased because lookups normalize the key via `.toLowerCase()`.
// Covers OAuth/JWT tokens plus common credential-shaped keys (api keys,
// passwords, client secrets) so they are stripped on non-auth routes too.
const SENSITIVE_BODY_KEYS = new Set([
  'access_token',
  'refresh_token',
  'id_token',
  'session_token',
  'api_key',
  'apikey',
  'x-api-key',
  'jwt',
  'secret',
  'password',
  'client_secret',
]);

// Auth routes whose response bodies carry credentials; their response bodies
// are dropped entirely rather than merely key-redacted.
const AUTH_ROUTE_PATTERNS = ['/auth/login', '/auth/refresh', '/auth/magic-link'];

function isAuthRoute(url?: string): boolean {
  if (!url) return false;
  const lower = url.toLowerCase();
  return AUTH_ROUTE_PATTERNS.some((route) => lower.includes(route));
}

function redactHeaders(
  headers?: Record<string, string>
): Record<string, string> | undefined {
  if (!headers) return headers;
  const out: Record<string, string> = {};
  for (const [key, value] of Object.entries(headers)) {
    out[key] = SENSITIVE_HEADERS.has(key.toLowerCase()) ? REDACTED : value;
  }
  return out;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== 'object') return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

// Non-mutating deep copy that replaces known credential-bearing keys with a
// placeholder. Returns the original reference for non-plain values so we never
// corrupt Blobs/ArrayBuffers or other non-JSON payloads.
function redactTokens(value: unknown, depth = 0): unknown {
  // Fail CLOSED past the depth limit: a security redactor must never return a
  // raw (unredacted) subtree, since credentials could be nested deeper than we
  // walk. Replace the entire truncated subtree with the placeholder instead.
  if (depth > 6) return REDACTED;
  if (Array.isArray(value)) {
    return value.map((entry) => redactTokens(entry, depth + 1));
  }
  if (!isPlainObject(value)) {
    return value;
  }
  const out: Record<string, unknown> = {};
  for (const [key, entry] of Object.entries(value)) {
    out[key] = SENSITIVE_BODY_KEYS.has(key.toLowerCase())
      ? REDACTED
      : redactTokens(entry, depth + 1);
  }
  return out;
}

function redactHistoryItem(item: RequestHistoryItem): RequestHistoryItem {
  return {
    ...item,
    requestHeaders: redactHeaders(item.requestHeaders),
    requestBody: redactTokens(item.requestBody),
    // Auth-route responses can carry tokens under many shapes, so drop the body
    // entirely. Other responses only need known credential keys stripped.
    responseBody: isAuthRoute(item.url)
      ? REDACTED
      : redactTokens(item.responseBody),
  };
}

function parseHistory(raw: string | null): RequestHistoryItem[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function addRequestHistory(item: RequestHistoryItem) {
  try {
    // Redact first so a serialization failure can never persist raw secrets.
    // Re-redact existing entries too, as defense-in-depth against any legacy
    // unredacted entry written by an older build.
    const arr = parseHistory(localStorage.getItem(KEY));
    const next = [redactHistoryItem(item), ...arr.map(redactHistoryItem)].slice(0, MAX);
    localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    // ignore
  }
}

export function getRequestHistory(): RequestHistoryItem[] {
  try {
    const raw = localStorage.getItem(KEY);
    const arr = parseHistory(raw);
    const sanitized = arr.map(redactHistoryItem).slice(0, MAX);
    const sanitizedRaw = JSON.stringify(sanitized);
    if (raw !== null && raw !== sanitizedRaw) {
      localStorage.setItem(KEY, sanitizedRaw);
    }
    return sanitized;
  } catch {
    return [];
  }
}

export function clearRequestHistory() {
  try {
    localStorage.removeItem(KEY);
  } catch {
    // localStorage may be unavailable
  }
}
