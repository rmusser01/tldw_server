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

// Debounce window for flushing buffered history entries to localStorage.
// Exported so tests drive the flush with fake timers at the real window size.
export const FLUSH_DEBOUNCE_MS = 500;

// Hard cap for the stored responseBody of NEW entries (2 KB), marker included.
// Legacy entries written by older builds are never re-truncated on read.
const MAX_RESPONSE_BODY_CHARS = 2048;

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
    if (!SENSITIVE_HEADERS.has(key.toLowerCase())) {
      out[key] = value;
    }
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

// Internal dispatch table. The insert/hydrate paths route redaction through
// this object (rather than calling redactHistoryItem directly) so tests can
// observe insert-time redaction with vi.spyOn(historyInternals,
// 'redactHistoryItem'); spies on the module namespace cannot intercept
// intra-module calls.
export const historyInternals = {
  redactHistoryItem,
};

// Serialize a body for size measurement; null when it has no JSON form
// (undefined, functions, symbols, circular structures).
function serializeBody(body: unknown): string | null {
  if (typeof body === 'string') return body;
  try {
    const json = JSON.stringify(body);
    return typeof json === 'string' ? json : null;
  } catch {
    return null;
  }
}

// Truncate an already-redacted response body for storage. Small bodies keep
// their original value and type; oversized bodies become a string capped at
// MAX_RESPONSE_BODY chars with an explicit truncation marker. Returns the
// input unchanged for values that cannot be serialized.
function truncateResponseBody(body: unknown): unknown {
  const serialized = serializeBody(body);
  if (serialized === null || serialized.length <= MAX_RESPONSE_BODY_CHARS) {
    return body;
  }
  const marker = `…[truncated ${serialized.length} chars]`;
  return serialized.slice(0, MAX_RESPONSE_BODY_CHARS - marker.length) + marker;
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

// --- Buffered persistence (WP-15) -------------------------------------------
//
// Entries live in a module-level array and are flushed to localStorage in one
// batched write per FLUSH_DEBOUNCE_MS trailing-debounce window, deferred to
// requestIdleCallback when the main thread is busy. Flushing also happens
// immediately on visibilitychange→hidden and beforeunload so a normal page
// close never loses buffered entries. There is no synchronous setItem on the
// request path: redaction and truncation are applied exactly once, at insert.

let cache: RequestHistoryItem[] | null = null;
let dirty = false;
let flushTimer: ReturnType<typeof setTimeout> | null = null;
let idleHandle: number | null = null;
let lifecycleBound = false;

// Hydrate the in-memory cache from localStorage once. Legacy entries written
// by older builds are sanitized once here (migration-on-read), never
// re-truncated, and never re-redacted on subsequent operations.
function hydrate(): RequestHistoryItem[] {
  if (cache === null) {
    let stored: RequestHistoryItem[] = [];
    try {
      stored = parseHistory(localStorage.getItem(KEY));
    } catch {
      stored = [];
    }
    cache = stored.map(historyInternals.redactHistoryItem).slice(0, MAX);
  }
  return cache;
}

function cancelScheduledFlush(): void {
  if (flushTimer !== null) {
    clearTimeout(flushTimer);
    flushTimer = null;
  }
  if (idleHandle !== null && typeof cancelIdleCallback === 'function') {
    cancelIdleCallback(idleHandle);
    idleHandle = null;
  }
}

// Write the buffered entries now. Retries naturally on the next scheduled
// flush if localStorage rejects the write (e.g. quota).
function flushNow(): void {
  cancelScheduledFlush();
  if (!dirty || cache === null) return;
  try {
    localStorage.setItem(KEY, JSON.stringify(cache));
    dirty = false;
  } catch {
    // ignore; keep dirty so a later flush retries
  }
}

function onVisibilityChange(): void {
  if (typeof document !== 'undefined' && document.visibilityState === 'hidden') {
    flushNow();
  }
}

function bindLifecycleFlush(): void {
  if (lifecycleBound || typeof window === 'undefined' || typeof document === 'undefined') {
    return;
  }
  document.addEventListener('visibilitychange', onVisibilityChange);
  window.addEventListener('beforeunload', flushNow);
  lifecycleBound = true;
}

function runWhenIdle(flush: () => void): void {
  if (typeof requestIdleCallback === 'function') {
    idleHandle = requestIdleCallback(() => {
      idleHandle = null;
      flush();
    }, { timeout: FLUSH_DEBOUNCE_MS * 4 });
    return;
  }
  flush();
}

function scheduleFlush(): void {
  dirty = true;
  bindLifecycleFlush();
  // Trailing debounce: every insert re-arms the window.
  if (flushTimer !== null) {
    clearTimeout(flushTimer);
  }
  flushTimer = setTimeout(() => {
    flushTimer = null;
    runWhenIdle(flushNow);
  }, FLUSH_DEBOUNCE_MS);
}

export function addRequestHistory(item: RequestHistoryItem) {
  try {
    // Redact first so a serialization failure can never persist raw secrets,
    // then truncate the redacted body — never the reverse, so a truncation
    // slice can never capture raw credential material. Both are applied
    // exactly once, at insert; existing entries are never re-processed.
    const redacted = historyInternals.redactHistoryItem(item);
    const entry: RequestHistoryItem = {
      ...redacted,
      responseBody: truncateResponseBody(redacted.responseBody),
    };
    const entries = hydrate();
    cache = [entry, ...entries].slice(0, MAX);
    scheduleFlush();
  } catch {
    // ignore
  }
}

// Flush buffered entries to localStorage immediately, bypassing the debounce
// window. Public companion to clearRequestHistory for callers (and tests) that
// need at-rest durability before reading the store.
export function flushRequestHistory(): void {
  flushNow();
}

export function getRequestHistory(): RequestHistoryItem[] {
  try {
    const entries = hydrate();
    // Migration-on-read for legacy stores: if persisted content predates
    // redaction (or was modified), rewrite it sanitized. Never truncates.
    const raw = localStorage.getItem(KEY);
    if (raw !== null) {
      const sanitizedRaw = JSON.stringify(entries);
      if (raw !== sanitizedRaw) {
        localStorage.setItem(KEY, sanitizedRaw);
      }
    }
    return entries.slice();
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
  cancelScheduledFlush();
  cache = null;
  dirty = false;
}
