import { addRequestHistory, clearRequestHistory } from '@web/lib/history';
import { dispatchAuthCredentialsChanged } from '@web/lib/auth-events';
import { getApiBearer, getApiKey, getSessionAccessToken, hasEnvApiAuth } from '@web/lib/authStorage';
import { buildApiBaseUrl, resolvePublicApiOrigin } from '@web/lib/api-base';
import { captureSessionIdFromHeaders, getOrCreateSessionId, SESSION_HEADER_NAME } from '@web/lib/session';
import {
  parseRetryAfter,
  tldwRequestRaw,
  type TldwCredentialContext,
  type TldwCredentialResolution,
} from '@/services/tldw/request-core';
import type { ApiErrorResponse, ApiRequestConfig, ApiRequestConfigWithMetadata } from '@web/types/common';
import { asVNGenerationRejectionCode, VN_GENERATION_REJECTION_MESSAGES, type VNGenerationRejectionCode } from '@web/lib/api/vnGenerationErrors';

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

// Canonical shared error model since TASK-13526 stage 2 (ADR-065): the class
// definition lives in packages/ui's api-error module so both clients share
// one shape; re-exported here so every existing `ApiError` import keeps
// working unchanged.
export { ApiError } from '@tldw/ui/services/tldw/api-error';
import { ApiError } from '@tldw/ui/services/tldw/api-error';

const deploymentEnv = {
  NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE,
  NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
};
const apiVersion = process.env.NEXT_PUBLIC_API_VERSION || 'v1';
const DEFAULT_TIMEOUT_MS = 30000;

const PUBLIC_PROVIDER_ERROR_MESSAGES = {
  provider_request_invalid: 'The selected provider or model is invalid.',
  provider_authentication_failed:
    'The selected provider credentials could not be authenticated.',
  invalid_provider_credentials: 'The selected provider credentials are invalid.',
  missing_provider_credentials:
    'The selected provider credentials are not configured.',
  credential_store_unavailable:
    'Provider credential storage is temporarily unavailable.',
  credential_scope_revoked:
    'The selected provider credential scope is no longer available.',
  provider_disabled: 'The selected provider is disabled by administrator policy.',
  model_not_allowed: 'The selected model is not allowed for this provider.',
  provider_configuration_invalid:
    'The selected provider configuration is invalid.',
  provider_unavailable: 'The selected provider is currently unavailable.',
} as const;

type PublicProviderErrorCode = keyof typeof PUBLIC_PROVIDER_ERROR_MESSAGES;
type PublicErrorCode = PublicProviderErrorCode | VNGenerationRejectionCode;

function asPublicProviderErrorCode(value: unknown): PublicProviderErrorCode | undefined {
  if (
    typeof value === 'string' &&
    Object.prototype.hasOwnProperty.call(PUBLIC_PROVIDER_ERROR_MESSAGES, value)
  ) {
    return value as PublicProviderErrorCode;
  }
  return undefined;
}

function normalizeApiErrorBody(errorBody: ApiErrorResponse, statusCode: number, vnGenerationRequest: boolean): {
  detail?: string;
  errorCode?: PublicErrorCode;
} {
  if (errorBody.detail && typeof errorBody.detail === 'object') {
    const errorCode = asPublicProviderErrorCode(errorBody.detail.error_code);
    if (errorCode) {
      return {
        detail: PUBLIC_PROVIDER_ERROR_MESSAGES[errorCode],
        errorCode,
      };
    }
    const rejection = statusCode === 409 && vnGenerationRequest
      ? asVNGenerationRejectionCode(errorBody.detail.code) : undefined;
    if (rejection) return { detail: VN_GENERATION_REJECTION_MESSAGES[rejection], errorCode: rejection };
  }
  // Untyped 5xx bodies are not a public contract and may contain raw upstream
  // or internal details. Typed, allowlisted provider envelopes above remain
  // available with their canonical client-owned messages.
  if (statusCode >= 500) {
    return {};
  }
  if (typeof errorBody.detail === 'string') {
    return { detail: errorBody.detail };
  }
  if (typeof errorBody.message === 'string') {
    return { detail: errorBody.message };
  }
  return {};
}

function buildSafeErrorHistoryBody(
  detail: string | undefined,
  errorCode: PublicErrorCode | undefined,
): ApiErrorResponse | undefined {
  if (errorCode) {
    return {
      detail: {
        error_code: errorCode,
        message: detail,
      },
    };
  }
  if (detail !== undefined) {
    return { detail };
  }
  return undefined;
}

function isCompleteProviderAuthenticationError(
  errorBody: ApiErrorResponse,
  errorCode: PublicErrorCode | undefined,
): boolean {
  if (errorCode !== 'provider_authentication_failed') {
    return false;
  }
  const nestedDetail = errorBody.detail;
  return (
    nestedDetail !== null &&
    typeof nestedDetail === 'object' &&
    typeof nestedDetail.message === 'string' &&
    nestedDetail.message.trim().length > 0
  );
}

export function shouldIncludeBrowserCredentials(): boolean {
  if (typeof window === 'undefined') {
    return true;
  }

  const hasJwtToken = !!getSessionAccessToken();
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

function resolveCredentials(config: ApiRequestConfig): RequestCredentials {
  const withCredentials =
    config.withCredentials !== undefined
      ? config.withCredentials
      : typeof window !== 'undefined'
        ? shouldIncludeBrowserCredentials()
        : api.defaults.withCredentials;

  return withCredentials ? 'include' : 'omit';
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
  // Shared parser (TASK-13526 transport Stage A) handles both delta-seconds
  // and HTTP-date forms and returns MILLISECONDS; the web ApiError.retryAfter
  // field keeps its axios-convention SECONDS semantics, so convert here.
  const ms = parseRetryAfter(headers.get('retry-after'));
  return ms == null ? undefined : Math.ceil(ms / 1000);
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

function createRequestCancellationError(): Error & { code: "REQUEST_ABORTED" } {
  const error = new Error('Request aborted') as Error & { code: "REQUEST_ABORTED" };
  error.name = 'AbortError';
  error.code = 'REQUEST_ABORTED';
  return error;
}

function handleUnauthorized(
  requestHeaders: Record<string, string> | undefined,
  sessionTokenAtStart: string | null
): void {
  if (typeof window === 'undefined') {
    return;
  }

  const authorization = String(requestHeaders?.['Authorization'] || '').trim();
  const requestToken = authorization.toLowerCase().startsWith('bearer ')
    ? authorization.slice(7).trim()
    : null;
  const currentToken = getSessionAccessToken();
  if (
    !requestToken ||
    !sessionTokenAtStart ||
    !currentToken ||
    requestToken !== sessionTokenAtStart ||
    requestToken !== currentToken
  ) {
    return;
  }

  localStorage.removeItem('access_token');
  localStorage.removeItem('user');
  dispatchAuthCredentialsChanged(false);
  // Forced logout on 401: also purge captured request-history so any tokens
  // recorded during the session do not persist past the invalidated session.
  clearRequestHistory();
  const hasStoredAuth = !!(getApiKey() || getApiBearer());
  if (
    !hasEnvAuthConfigured() &&
    !hasStoredAuth &&
    shouldRedirectUnauthorizedToLogin(window.location.pathname)
  ) {
    window.location.href = '/login';
  }
}


/**
 * TASK-13526 transport Stage B2: the web shell's runtime adapter for the
 * shared raw transport. Config makes plan resolution a passthrough for the
 * web's pre-built URL (advanced deployments send an absolute same-origin-
 * for-the-configured-server URL; quickstart sends a same-origin relative
 * URL which the plan resolves identically). Credentials come from the web's
 * authStorage domain via the Stage-B credential strategy.
 */
const webTransportRuntimeFor = (baseURL: string | undefined) => ({
  getConfig: async () => {
    let serverUrl: string | undefined;
    if (baseURL && /^https?:/i.test(baseURL)) {
      try {
        serverUrl = new URL(baseURL).origin;
      } catch {
        serverUrl = undefined;
      }
    }
    return {
      serverUrl,
      requestTimeoutMs: api.defaults.timeout,
    };
  },
  resolveCredentials: (context: TldwCredentialContext): TldwCredentialResolution => {
    if (typeof window === 'undefined') {
      return { headers: {} };
    }
    // Mirrors the retired applyBrowserHeaders precedence exactly: caller
    // headers arrive in context.existingHeaders; storage credentials only
    // fill absent slots (Authorization) per the previous `!headers.has`
    // checks, X-API-KEY follows the previous unconditional-set behavior.
    const existing = context.existingHeaders;
    const existingHas = (name: string) =>
      Object.keys(existing).some(
        (key) => key.toLowerCase() === name.toLowerCase()
      );
    // The shared transport strips auth-slot headers before applying this
    // strategy, so caller-provided values must be carried back explicitly.
    const existingValue = (name: string) =>
      Object.entries(existing).find(
        ([key]) => key.toLowerCase() === name.toLowerCase()
      )?.[1];
    const headers: Record<string, string> = {};
    const existingAuthorization = existingValue('authorization');
    if (existingAuthorization) {
      headers['Authorization'] = existingAuthorization;
    }
    const existingApiKey = existingValue('x-api-key');
    if (existingApiKey) {
      headers['X-API-KEY'] = existingApiKey;
    }
    const existingCsrf = existingValue('x-csrf-token');
    if (existingCsrf) {
      headers['X-CSRF-Token'] = existingCsrf;
    }
    const existingSession = existingValue(SESSION_HEADER_NAME.toLowerCase());
    if (existingSession) {
      headers[SESSION_HEADER_NAME] = existingSession;
    }
    const sessionId = getOrCreateSessionId();
    if (sessionId && !existingHas(SESSION_HEADER_NAME)) {
      headers[SESSION_HEADER_NAME] = sessionId;
    }
    const token = getSessionAccessToken();
    if (token && !existingHas('authorization')) {
      headers['Authorization'] = `Bearer ${token}`;
    }
    const apiBearer = getApiBearer();
    if (apiBearer && !existingHas('authorization') && !headers['Authorization']) {
      headers['Authorization'] = `Bearer ${apiBearer}`;
    }
    const xApiKey = getApiKey();
    if (xApiKey && !existingHas('x-api-key')) {
      headers['X-API-KEY'] = xApiKey;
    }
    const methodUp = context.method.toUpperCase();
    const needsCsrf = ['POST', 'PUT', 'PATCH', 'DELETE'].includes(methodUp) && !xApiKey;
    if (needsCsrf && !context.shouldSkipAuth) {
      const match = document.cookie.match(
        new RegExp('(?:^|; )csrf_token=([^;]*)')
      );
      const csrf = match ? decodeURIComponent(match[1]) : null;
      if (csrf && !existingHas('x-csrf-token')) {
        headers['X-CSRF-Token'] = csrf;
      }
    }
    return { headers };
  },
});

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
  const sessionTokenAtStart =
    typeof window !== 'undefined' ? getSessionAccessToken() : null;

  const requestConfig = buildRequestHistoryConfig(method, url, baseURL, headers, data, metadata);

  // Shared raw transport (TASK-13526 Stage B2): plan/auth/timeout/fetch
  // mechanics live in packages/ui's request-core; the web policy layer
  // (history, session capture, 401 logout, CSRF messaging, error
  // sanitization) stays below.
  const raw = await tldwRequestRaw(
    {
      path: requestUrl as Parameters<typeof tldwRequestRaw>[0]['path'],
      method,
      headers: headersToRecord(headers),
      body,
      timeoutMs: config.timeout,
      abortSignal: config.signal,
      credentials: resolveCredentials(config),
      preResolvedUrl: true,
    },
    webTransportRuntimeFor(baseURL),
  );

  if (raw.ok !== true) {
    if (raw.kind === 'aborted' || (config.signal?.aborted && raw.kind !== 'timeout')) {
      throw createRequestCancellationError();
    }
    const message =
      raw.kind === 'timeout' ? 'Request timed out' : raw.error || 'Network Error';
    recordFailure(requestConfig, { errorMessage: message });
    throw new ApiError(message, {
      detail: message,
      ...(raw.status !== undefined ? { status: raw.status } : {}),
    });
  }
  const response = raw.response;
  const requestHeadersRecord = raw.context.headers;

  captureSessionIdFromHeaders(headersToRecord(response.headers));

  if (!response.ok) {
    const errorBody = await parseErrorBody(response);
    const { detail, errorCode } = normalizeApiErrorBody(
      errorBody,
      response.status,
      method === 'POST' && /^\/vn\/vn-assets\/packs\/[1-9]\d*\/(?:generate|slots\/[1-9]\d*\/retry)$/.test(url),
    );
    const safeErrorHistoryBody = buildSafeErrorHistoryBody(detail, errorCode);
    if (
      response.status === 401 &&
      !isCompleteProviderAuthenticationError(errorBody, errorCode)
    ) {
      handleUnauthorized(requestHeadersRecord, sessionTokenAtStart);
    }
    if (
      response.status === 403 &&
      detail &&
      typeof detail === 'string' &&
      detail.toLowerCase().includes('csrf')
    ) {
      const message = 'CSRF validation failed. Refresh the page and try again.';
      recordFailure(requestConfig, {
        status: response.status,
        responseBody: { detail: message },
        errorMessage: message,
      });
      throw new ApiError(message, {
        status: response.status,
        detail: message,
      });
    }

    const retryAfter = retryAfterFromHeaders(response.headers);
    const message = detail || response.statusText || 'An unexpected error occurred';
    recordFailure(requestConfig, {
      status: response.status,
      responseBody: safeErrorHistoryBody,
      errorMessage: message,
    });
    throw new ApiError(message, {
      status: response.status,
      detail,
      errorCode,
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

    const token = getSessionAccessToken();
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

  const token = getSessionAccessToken();
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
