import { getApiKey, logout } from './auth';

export class ApiError extends Error {
  status: number;
  detail?: unknown;

  constructor(status: number, message: string, detail?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

export class WebhookApiError extends ApiError {
  code: string;
  requestId: string;

  constructor(status: number, code: string, message: string, requestId: string) {
    super(status, message);
    this.name = 'WebhookApiError';
    this.code = code;
    this.requestId = requestId;
  }
}

export class WebhookContractError extends ApiError {
  requestId: string | null;

  constructor(
    status: number,
    message = 'Webhook API returned an invalid contract response',
    requestId: string | null = null
  ) {
    super(status, message);
    this.name = 'WebhookContractError';
    this.requestId = requestId;
  }
}

export class WebhookTransportError extends ApiError {
  requestId: string;

  constructor(status: 502 | 504, requestId: string) {
    super(
      status,
      status === 504
        ? 'Webhook backend request timed out'
        : 'Webhook backend is unavailable',
    );
    this.name = 'WebhookTransportError';
    this.requestId = requestId;
  }
}

export type JsonResponse<T> = {
  data: T;
  status: number;
  etag: string | null;
  requestId: string | null;
};

type ResponseType = 'json' | 'text' | 'blob';

const WEBHOOK_ERROR_CODE = /^[a-z][a-z0-9_]{0,127}$/;
const REQUEST_ID = /^[A-Za-z0-9._:-]{1,128}$/;

type ParsedWebhookError = {
  code: string;
  message: string;
  requestId: string;
};

export const buildProxyUrl = (endpoint: string): string => {
  if (!endpoint) return '/api/proxy';
  return `/api/proxy${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;
};

export const buildAuthHeaders = (): Record<string, string> => {
  const headers: Record<string, string> = {};

  const apiKey = getApiKey();
  if (apiKey) {
    headers['X-API-KEY'] = apiKey;
  }

  return headers;
};

const toApiErrorMessage = (detail: unknown): string => {
  if (!detail) return 'Request failed';
  if (typeof detail === 'string') return detail;
  if (detail && typeof detail === 'object') {
    const record = detail as { detail?: unknown; message?: unknown };
    if (typeof record.detail === 'string') return record.detail;
    if (typeof record.message === 'string') return record.message;
  }
  return 'Request failed';
};

const isRecord = (value: unknown): value is Record<string, unknown> => (
  value !== null && typeof value === 'object' && !Array.isArray(value)
);

const hasExactKeys = (record: Record<string, unknown>, keys: string[]): boolean => {
  const actual = Object.keys(record).sort();
  return actual.length === keys.length
    && actual.every((key, index) => key === [...keys].sort()[index]);
};

const boundedRequestId = (value: unknown): string | null => (
  typeof value === 'string' && REQUEST_ID.test(value) ? value : null
);

const parseWebhookError = (value: unknown): ParsedWebhookError | null => {
  if (!isRecord(value) || !hasExactKeys(value, ['error']) || !isRecord(value.error)) {
    return null;
  }
  const error = value.error;
  if (!hasExactKeys(error, ['code', 'message', 'request_id'])) return null;
  if (typeof error.code !== 'string' || !WEBHOOK_ERROR_CODE.test(error.code)) return null;
  if (
    typeof error.message !== 'string'
    || error.message.length < 1
    || error.message.length > 200
  ) {
    return null;
  }
  const requestId = boundedRequestId(error.request_id);
  if (!requestId) return null;
  return { code: error.code, message: error.message, requestId };
};

const isProxyTransportError = (status: number, value: unknown): status is 502 | 504 => {
  if (!isRecord(value) || !hasExactKeys(value, ['detail'])) return false;
  return (status === 502 && value.detail === 'Backend unavailable')
    || (status === 504 && value.detail === 'Backend request timed out');
};

const isWebhookEndpoint = (endpoint: string): boolean => {
  const path = endpoint.split('?', 1)[0];
  return path === '/admin/webhooks' || path.startsWith('/admin/webhooks/');
};

const buildRequestHeaders = (overrides?: HeadersInit): Headers => {
  const headers = new Headers(buildAuthHeaders());
  if (overrides) {
    const overrideHeaders = new Headers(overrides);
    overrideHeaders.forEach((value, key) => {
      headers.set(key, value);
    });
  }
  return headers;
};

const requestResponse = async (
  endpoint: string,
  options: RequestInit = {},
  inferJsonContentType = false,
): Promise<Response> => {
  const headers = buildRequestHeaders(options.headers);

  if (inferJsonContentType && typeof options.body === 'string' && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  const response = await fetch(buildProxyUrl(endpoint), {
    ...options,
    headers,
    credentials: 'include',
  });

  if (!response.ok) {
    const error = await response.json().catch(() => null);

    if (response.status === 401 && typeof window !== 'undefined') {
      await logout();
      if (!window.location.pathname.startsWith('/login')) {
        window.location.href = '/login';
      }
    }

    if (isWebhookEndpoint(endpoint)) {
      const headerRequestId = boundedRequestId(response.headers.get('x-request-id'));
      if (headerRequestId && isProxyTransportError(response.status, error)) {
        throw new WebhookTransportError(response.status, headerRequestId);
      }
      const parsed = parseWebhookError(error);
      if (parsed && headerRequestId === parsed.requestId) {
        throw new WebhookApiError(
          response.status,
          parsed.code,
          parsed.message,
          parsed.requestId
        );
      }
      throw new WebhookContractError(
        response.status,
        undefined,
        headerRequestId
      );
    }

    if (response.status === 403) {
      const detail = (error as { detail?: unknown })?.detail || '';
      if (typeof detail === 'string' && detail.toLowerCase().includes('csrf')) {
        throw new ApiError(
          response.status,
          'CSRF validation failed. Please refresh the page and try again.',
          error
        );
      }
    }

    throw new ApiError(response.status, toApiErrorMessage(error), error);
  }

  return response;
};

const parseJsonResponse = async <T>(response: Response, endpoint: string): Promise<T> => {
  const text = await response.text();
  if (!text) return {} as T;
  try {
    return JSON.parse(text) as T;
  } catch (error) {
    if (isWebhookEndpoint(endpoint)) {
      throw new WebhookContractError(
        response.status,
        'Webhook API returned invalid JSON',
        boundedRequestId(response.headers.get('x-request-id'))
      );
    }
    throw error;
  }
};

const requestRaw = async <T>(
  endpoint: string,
  responseType: ResponseType,
  options: RequestInit = {}
): Promise<T> => {
  const response = await requestResponse(endpoint, options, responseType === 'json');

  if (responseType === 'text') {
    return response.text() as Promise<T>;
  }

  if (responseType === 'blob') {
    return response.blob() as Promise<T>;
  }

  return parseJsonResponse<T>(response, endpoint);
};

export const requestJson = async <T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> => requestRaw<T>(endpoint, 'json', options);

export const requestJsonWithMetadata = async <T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<JsonResponse<T>> => {
  const response = await requestResponse(endpoint, options, true);
  const data = await parseJsonResponse<T>(response, endpoint);
  return {
    data,
    status: response.status,
    etag: response.headers.get('etag'),
    requestId: boundedRequestId(response.headers.get('x-request-id')),
  };
};

export const requestText = async (
  endpoint: string,
  options: RequestInit = {}
): Promise<string> => requestRaw<string>(endpoint, 'text', options);

export const requestBlob = async (
  endpoint: string,
  options: RequestInit = {}
): Promise<Blob> => requestRaw<Blob>(endpoint, 'blob', options);
