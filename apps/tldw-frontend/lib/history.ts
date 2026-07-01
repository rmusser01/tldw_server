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
const SENSITIVE_HEADER_NAMES = new Set([
  'authorization',
  'cookie',
  'proxy-authorization',
  'set-cookie',
  'x-api-key',
  'x-auth-token',
]);

function redactRequestHeaders(headers: RequestHistoryItem['requestHeaders']): RequestHistoryItem['requestHeaders'] {
  if (!headers) return headers;
  return Object.fromEntries(
    Object.entries(headers).map(([name, value]) => [
      name,
      SENSITIVE_HEADER_NAMES.has(name.toLowerCase()) ? '[REDACTED]' : value,
    ]),
  );
}

function sanitizeHistoryItem(item: RequestHistoryItem): RequestHistoryItem {
  return {
    ...item,
    requestHeaders: redactRequestHeaders(item.requestHeaders),
  };
}

function parseHistory(raw: string | null): RequestHistoryItem[] {
  if (!raw) return [];
  const parsed = JSON.parse(raw);
  return Array.isArray(parsed) ? parsed : [];
}

export function addRequestHistory(item: RequestHistoryItem) {
  try {
    const arr = parseHistory(localStorage.getItem(KEY));
    const next = [sanitizeHistoryItem(item), ...arr.map(sanitizeHistoryItem)].slice(0, MAX);
    localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    // ignore
  }
}

export function getRequestHistory(): RequestHistoryItem[] {
  try {
    const raw = localStorage.getItem(KEY);
    const arr = parseHistory(raw);
    const sanitized = arr.map(sanitizeHistoryItem).slice(0, MAX);
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
