import { addRequestHistory } from '@web/lib/history';
import { getApiBearer, getApiKey, hasEnvApiAuth } from '@web/lib/authStorage';
import { buildApiBaseUrl, resolvePublicApiOrigin } from '@web/lib/api-base';
import { captureSessionIdFromHeaders, getOrCreateSessionId, SESSION_HEADER_NAME } from '@web/lib/session';
import type { ApiErrorResponse, ApiRequestConfig, ApiRequestConfigWithMetadata } from '@web/types/common';

type ApiResponse<T = unknown> = {
  data: T;
  status: number;
  headers: Headers;
  config: ApiRequestConfigWithMetadata;
};

type ApiDefaults = {
  baseURL: string;
  headers: Record<string, string>;
  timeout: number;
  withCredentials: boolean;
};

type RequestMethod = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';

// Custom error type that preserves HTTP status and retry hints while remaining compatible with Error
export class ApiError extends Error {
  status?: number;
  statusCode?: number;
  detail?: string;
  retryAfter?: number;

  constructor(message: string, options?: { status?: number; detail?: string; retryAfter?: number }) {
    super(message);
    this.name = 'ApiError';
    if (options?.status !== undefined) {
      this.status = options.status;
      this.statusCode = options.status;
    }
    if (options?.detail !== undefined) {
      this.detail = options.detail;
    }
    if (options?.retryAfter !== undefined) {
      this.retryAfter = options.retryAfter;
    }
  }
}

const deploymentEnv = {
  NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE,
  NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
};
const apiVersion = process.env.NEXT_PUBLIC_API_VERSION || 'v1';
const DEFAULT_TIMEOUT_MS = 30000;

export function shouldIncludeBrowserCredentials(): boolean {
  if (typeof window === 'undefined') {
    return true;
  }

  const hasJwtToken = !!localStorage.getItem('access_token');
  if (hasJwtToken) {
    return true;
  }

  if (getApiKey()) {
    return false;
  }

  if (getApiBearer()) {
    return false;
  }

  return true;
}

const normalizePathname = (pathname: string): string => {
  const trimmed = pathname.trim();
  if (!trimmed) return "/";
  if (trimmed === "/") return "/";
  return trimmed.endsWith("/") ? trimmed.slice(0, -1) : trimmed;
}

export function shouldRedirectUnauthorizedToLogin(pathname?: string): boolean {
  const resolvedPath =
    typeof pathname === "string"
      ? normalizePathname(pathname)
      : typeof window !== "undefined"
        ? normalizePathname(window.location.pathname || "/")
        : "/";

  if (
    resolvedPath === "/login" ||
    resolvedPath === "/setup" ||
    resolvedPath === "/signup" ||
    resolvedPath === "/settings" ||
    resolvedPath.startsWith("/settings/") ||
    resolvedPath.startsWith("/auth/")
  ) {
    return false;
  }

  return true;
}

export function hasEnvAuthConfigured(): boolean {
  return hasEnvApiAuth();
}

function resolveDefaultApiBaseUrl(): string {
  const pageOrigin = typeof window !== 'undefined' ? window.location?.origin : undefined;
  return buildApiBaseUrl(resolvePublicApiOrigin(deploymentEnv, pageOrigin), apiVersion);
}

// Read cookie value on client
function getCookie(name: string): string | null {
  if (typeof document === 'undefined') return null;
  const match = document.cookie.match(new RegExp('(?:^|; )' + name.replace(/([.$?*|{}()[\]\\/+^])/g, '\\$1') + '=([^;]*)'));
  return match ? decodeURIComponent(match[1]) : null;
}

function isBodyInit(value: unknown): value is BodyInit {
  return (
    typeof value === 'string' ||
    value instanceof FormData ||
    value instanceof URLSearchParams ||
    value instanceof Blob ||
    value instanceof ArrayBuffer ||
    ArrayBuffer.isView(value)
  );
}

function normalizeHeaders(headers?: HeadersInit): Headers {
  return new Headers(headers);
}

function headersToRecord(headers: Headers): Record<string, string> {
  const record: Record<string, string> = {};
  headers.forEach((value, key) => {
    record[key] = value;
  });
  return record;
}

function appendParams(url: string, params?: ApiRequestConfig['params']): string {
  if (!params) {
    return url;
  }

  const isAbsolute = /^[a-z][a-z\d+\-.]*:\/\//i.test(url);
  const isProtocolRelative = url.startsWith('//');
  const parsed = new URL(url, isAbsolute ? undefined : 'http://tldw.local');

  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined) {
      continue;
    }
    parsed.searchParams.set(key, String(value));
  }

  if (isAbsolute) {
    return parsed.toString();
  }

  if (isProtocolRelative) {
    return parsed.toString().replace(/^http:/i, '');
  }

  return `${parsed.pathname}${parsed.search}${parsed.hash}`;
}

function joinUrl(baseURL: string | undefined, url: string, params?: ApiRequestConfig['params']): string {
  if (/^[a-z][a-z\d+\-.]*:\/\//i.test(url) || url.startsWith('//')) {
    return appendParams(url, params);
  }

  const base = baseURL || '';
  const normalizedBase = base.endsWith('/') ? base.slice(0, -1) : base;
  const normalizedPath = url.startsWith('/') ? url : `/${url}`;
  return appendParams(`${normalizedBase}${normalizedPath}`, params);
}

function serializeBody(data: unknown, headers: Headers): BodyInit | undefined {
  if (data === undefined || data === null) {
    return undefined;
  }

  if (typeof FormData !== 'undefined' && data instanceof FormData) {
    headers.delete('Content-Type');
    return data;
  }

  if (isBodyInit(data)) {
    return data;
  }

  if (!headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }
  return JSON.stringify(data);
}

function applyBrowserHeaders(headers: Headers, method: RequestMethod): void {
  if (typeof window === 'undefined') {
    return;
  }

  const sessionId = getOrCreateSessionId();
  if (sessionId && !headers.has(SESSION_HEADER_NAME)) {
    headers.set(SESSION_HEADER_NAME, sessionId);
  }

  // Bearer token (multi-user JWT auth)
  const token = localStorage.getItem('access_token');
  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`);
  }

  // Static API auth options via env or localStorage
  // Prefer explicit API bearer if provided (for chat module API_BEARER)
  const apiBearer = getApiBearer();
  if (apiBearer && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${apiBearer}`);
  }

  // X-API-KEY (single-user mode convenience)
  const xApiKey = getApiKey();
  if (xApiKey && !headers.has('X-API-KEY')) {
    headers.set('X-API-KEY', xApiKey);
  }

  // CSRF token for modifying requests when not using X-API-KEY auth
  const needsCsrf = ['POST', 'PUT', 'PATCH', 'DELETE'].includes(method) && !xApiKey;
  if (needsCsrf && !headers.has('X-CSRF-Token')) {
    const csrf = getCookie('csrf_token');
    if (csrf) {
      headers.set('X-CSRF-Token', csrf);
    }
  }
}

function resolveCredentials(config: ApiRequestConfig): RequestCredentials {
  const withCredentials =
    config.withCredentials !== undefined
      ? config.withCredentials
      : typeof window !== 'undefined'
        ? shouldIncludeBrowserCredentials()
        : api.defaults.withCredentials;

  return withCredentials ? 'include' : 'omit';
}

function createAbortSignal(config: ApiRequestConfig): {
  signal?: AbortSignal;
  cleanup: () => void;
  didTimeout: () => boolean;
} {
  const timeout = config.timeout ?? api.defaults.timeout;
  const callerSignal = config.signal;
  const controller = new AbortController();
  let timeoutId: ReturnType<typeof setTimeout> | undefined;
  let timedOut = false;

  const abortFromCaller = () => controller.abort();

  if (callerSignal) {
    if (callerSignal.aborted) {
      controller.abort();
    } else {
      callerSignal.addEventListener('abort', abortFromCaller, { once: true });
    }
  }

  if (timeout && timeout > 0) {
    timeoutId = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, timeout);
  }

  return {
    signal: controller.signal,
    cleanup: () => {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
      if (callerSignal) {
        callerSignal.removeEventListener('abort', abortFromCaller);
      }
    },
    didTimeout: () => timedOut,
  };
}

async function parseResponseBody<T>(response: Response, responseType?: ApiRequestConfig['responseType']): Promise<T> {
  if (response.status === 204 || response.status === 205) {
    return undefined as T;
  }

  if (responseType === 'arraybuffer') {
    return response.arrayBuffer() as Promise<T>;
  }

  if (responseType === 'blob') {
    return response.blob() as Promise<T>;
  }

  const text = await response.text();
  if (!text) {
    return undefined as T;
  }

  if (responseType === 'text') {
    return text as T;
  }

  const contentType = response.headers.get('Content-Type') || '';
  const isJsonHeader = contentType.toLowerCase().includes('application/json');
  if (responseType === 'json' || isJsonHeader) {
    try {
      return JSON.parse(text) as T;
    } catch {
      throw new ApiError('Invalid JSON response', {
        status: response.status,
        detail: 'Invalid JSON response',
      });
    }
  }

  return text as T;
}

function coerceErrorBody(body: unknown): ApiErrorResponse {
  if (body && typeof body === 'object') {
    return body as ApiErrorResponse;
  }
  if (body === undefined || body === null || body === '') {
    return {};
  }
  return { detail: String(body) };
}

async function parseErrorBody(response: Response): Promise<ApiErrorResponse> {
  if (response.status === 204 || response.status === 205) {
    return {};
  }

  const text = await response.text();
  if (!text) {
    return {};
  }

  try {
    return coerceErrorBody(JSON.parse(text));
  } catch {
    const contentType = response.headers.get('Content-Type') || '';
    if (contentType.toLowerCase().includes('application/json')) {
      return {};
    }
    return { detail: text };
  }
}

function retryAfterFromHeaders(headers: Headers): number | undefined {
  const retryAfterHeader = headers.get('retry-after');
  return retryAfterHeader ? parseInt(retryAfterHeader, 10) || undefined : undefined;
}

function buildRequestHistoryConfig(
  method: RequestMethod,
  url: string,
  baseURL: string,
  headers: Headers,
  data: unknown,
  metadata: { start: number }
): ApiRequestConfigWithMetadata {
  return {
    method,
    url,
    baseURL,
    headers: headersToRecord(headers),
    data,
    metadata,
  };
}

function recordSuccess<T>(response: ApiResponse<T>): void {
  try {
    const start = response.config.metadata?.start || Date.now();
    const duration = Date.now() - start;
    addRequestHistory({
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      method: response.config.method || 'GET',
      url: response.config.url || '',
      baseURL: response.config.baseURL || api.defaults.baseURL,
      status: response.status,
      ok: response.status >= 200 && response.status < 300,
      duration_ms: duration,
      timestamp: new Date().toISOString(),
      requestHeaders: response.config.headers,
      requestBody: response.config.data,
      responseBody: response.data,
    });
  } catch {
    // Silently ignore history logging errors to not disrupt API responses
  }
}

function recordFailure(
  config: ApiRequestConfigWithMetadata,
  options: {
    status?: number;
    responseBody?: unknown;
    errorMessage?: string;
  }
): void {
  try {
    const start = config.metadata?.start || Date.now();
    const duration = Date.now() - start;
    addRequestHistory({
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      method: config.method || 'GET',
      url: config.url || '',
      baseURL: config.baseURL || api.defaults.baseURL,
      status: options.status,
      ok: false,
      duration_ms: duration,
      timestamp: new Date().toISOString(),
      requestHeaders: config.headers,
      requestBody: config.data,
      responseBody: options.responseBody,
      errorMessage: options.errorMessage,
    });
  } catch {
    // Silently ignore history logging errors to not disrupt error handling
  }
}

function handleUnauthorized(): void {
  if (typeof window === 'undefined') {
    return;
  }

  localStorage.removeItem('access_token');
  localStorage.removeItem('user');
  const hasStoredAuth = !!(getApiKey() || getApiBearer());
  if (
    !hasEnvAuthConfigured() &&
    !hasStoredAuth &&
    shouldRedirectUnauthorizedToLogin(window.location.pathname)
  ) {
    window.location.href = '/login';
  }
}

async function request<T = unknown>(
  method: RequestMethod,
  url: string,
  data?: unknown,
  config: ApiRequestConfig = {}
): Promise<ApiResponse<T>> {
  const headers = normalizeHeaders(api.defaults.headers);
  normalizeHeaders(config.headers).forEach((value, key) => {
    headers.set(key, value);
  });
  const body = serializeBody(data, headers);
  const baseURL = config.baseURL ?? api.defaults.baseURL;
  const requestUrl = joinUrl(baseURL, url, config.params);
  const metadata = { start: Date.now() };

  applyBrowserHeaders(headers, method);

  const requestConfig = buildRequestHistoryConfig(method, url, baseURL, headers, data, metadata);
  const abort = createAbortSignal(config);

  let response: Response;
  try {
    response = await fetch(requestUrl, {
      method,
      headers,
      body,
      credentials: resolveCredentials(config),
      signal: abort.signal,
    });
  } catch (error) {
    abort.cleanup();
    const message =
      error instanceof DOMException && error.name === 'AbortError'
        ? abort.didTimeout()
          ? 'Request timed out'
          : 'Request aborted'
        : error instanceof Error
          ? error.message
          : 'Network Error';
    recordFailure(requestConfig, { errorMessage: message });
    throw new ApiError(message, { detail: message });
  }

  abort.cleanup();
  captureSessionIdFromHeaders(headersToRecord(response.headers));

  if (!response.ok) {
    if (response.status === 401) {
      handleUnauthorized();
    }

    const errorBody = await parseErrorBody(response);
    const detail = errorBody.detail || errorBody.message;
    if (
      response.status === 403 &&
      detail &&
      typeof detail === 'string' &&
      detail.toLowerCase().includes('csrf')
    ) {
      const message = 'CSRF validation failed. Refresh the page and try again.';
      recordFailure(requestConfig, {
        status: response.status,
        responseBody: errorBody,
        errorMessage: message,
      });
      throw new Error(message);
    }

    const retryAfter = retryAfterFromHeaders(response.headers);
    const message = detail || response.statusText || 'An unexpected error occurred';
    recordFailure(requestConfig, {
      status: response.status,
      responseBody: errorBody,
      errorMessage: message,
    });
    throw new ApiError(message, {
      status: response.status,
      detail,
      retryAfter,
    });
  }

  let responseBody: T;
  try {
    responseBody = await parseResponseBody<T>(response, config.responseType);
  } catch (error) {
    if (error instanceof ApiError) {
      recordFailure(requestConfig, {
        status: response.status,
        errorMessage: error.message,
      });
    }
    throw error;
  }

  const apiResponse: ApiResponse<T> = {
    data: responseBody,
    status: response.status,
    headers: response.headers,
    config: requestConfig,
  };
  recordSuccess(apiResponse);
  return apiResponse;
}

const api = {
  defaults: {
    baseURL: resolveDefaultApiBaseUrl(),
    headers: {
      'Content-Type': 'application/json',
    },
    withCredentials: true,
    timeout: DEFAULT_TIMEOUT_MS,
  } satisfies ApiDefaults,
  request: <T = unknown>(config: ApiRequestConfig & { url: string; method?: RequestMethod; data?: unknown }) =>
    request<T>(config.method || 'GET', config.url, config.data, config),
  get: <T = unknown>(url: string, config?: ApiRequestConfig) => request<T>('GET', url, undefined, config),
  post: <T = unknown>(url: string, data?: unknown, config?: ApiRequestConfig) => request<T>('POST', url, data, config),
  put: <T = unknown>(url: string, data?: unknown, config?: ApiRequestConfig) => request<T>('PUT', url, data, config),
  delete: <T = unknown>(url: string, config?: ApiRequestConfig) => request<T>('DELETE', url, undefined, config),
  patch: <T = unknown>(url: string, data?: unknown, config?: ApiRequestConfig) => request<T>('PATCH', url, data, config),
};

// Helper functions for common HTTP methods
export const apiClient = {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  get: <T = any>(url: string, config?: ApiRequestConfig) => api.get<T>(url, config).then((res) => res.data),
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  post: <T = any>(url: string, data?: unknown, config?: ApiRequestConfig) => api.post<T>(url, data, config).then((res) => res.data),
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  put: <T = any>(url: string, data?: unknown, config?: ApiRequestConfig) => api.put<T>(url, data, config).then((res) => res.data),
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  delete: <T = any>(url: string, config?: ApiRequestConfig) => api.delete<T>(url, config).then((res) => res.data),
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  patch: <T = any>(url: string, data?: unknown, config?: ApiRequestConfig) => api.patch<T>(url, data, config).then((res) => res.data),
};

export default api;

// Streaming helpers
export const API_BASE_URL = resolveDefaultApiBaseUrl();

export function getApiBaseUrl(): string {
  return api.defaults.baseURL || API_BASE_URL;
}

export function buildAuthHeaders(method: string = 'GET', contentType?: string): Record<string, string> {
  const headers: Record<string, string> = {};
  if (contentType) headers['Content-Type'] = contentType;

  if (typeof window !== 'undefined') {
    const sessionId = getOrCreateSessionId();
    if (sessionId) headers[SESSION_HEADER_NAME] = sessionId;

    const token = localStorage.getItem('access_token');
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const apiBearer = getApiBearer();
    if (apiBearer && !headers['Authorization']) {
      headers['Authorization'] = `Bearer ${apiBearer}`;
    }

    const xApiKey = getApiKey();
    if (xApiKey) headers['X-API-KEY'] = xApiKey;

    // CSRF for modifying requests when not using X-API-KEY
    const methodUp = method.toUpperCase();
    const needsCsrf = ['POST', 'PUT', 'PATCH', 'DELETE'].includes(methodUp) && !xApiKey;
    if (needsCsrf) {
      const cookie = (name: string): string | null => {
        const match = document.cookie.match(new RegExp('(?:^|; )' + name.replace(/([.$?*|{}()[\]\\/+^])/g, '\\$1') + '=([^;]*)'));
        return match ? decodeURIComponent(match[1]) : null;
      };
      const csrf = cookie('csrf_token');
      if (csrf) headers['X-CSRF-Token'] = csrf;
    }
  }

  return headers;
}

export function hasExplicitAuthHeaders(): boolean {
  if (typeof window === 'undefined') {
    return false;
  }

  const token = localStorage.getItem('access_token');
  if (token) {
    return true;
  }

  if (getApiBearer()) {
    return true;
  }

  if (getApiKey()) {
    return true;
  }

  return false;
}
