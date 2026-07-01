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

export function addRequestHistory(item: RequestHistoryItem) {
  try {
    const raw = localStorage.getItem(KEY);
    const arr: RequestHistoryItem[] = raw ? JSON.parse(raw) : [];
    const next = [sanitizeHistoryItem(item), ...arr.map(sanitizeHistoryItem)].slice(0, MAX);
    localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    // ignore
  }
}

export function getRequestHistory(): RequestHistoryItem[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const arr: RequestHistoryItem[] = JSON.parse(raw);
    return arr;
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
