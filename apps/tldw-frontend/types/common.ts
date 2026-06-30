/**
 * Shared types used across the tldw-frontend codebase.
 */

// ========================================
// API Types
// ========================================

/** Request options supported by the WebUI fetch-backed API client. */
export interface ApiRequestConfig {
  baseURL?: string;
  data?: unknown;
  headers?: HeadersInit;
  method?: string;
  params?: Record<string, string | number | boolean | null | undefined>;
  responseType?: 'json' | 'text' | 'arraybuffer' | 'blob';
  signal?: AbortSignal;
  timeout?: number;
  url?: string;
  withCredentials?: boolean;
}

/** Extended API config with request metadata for timing */
export interface ApiRequestConfigWithMetadata extends ApiRequestConfig {
  metadata?: {
    start: number;
  };
  data?: unknown;
  headers?: Record<string, string>;
  method?: string;
  url?: string;
}

/** Standard API error response structure */
export interface ApiErrorResponse {
  detail?: string;
  message?: string;
}

/** Generic API client config */
export type ApiClientConfig = ApiRequestConfig;

// ========================================
// User/Auth Types
// ========================================

/** User object structure from auth endpoints */
export interface AuthUser {
  id?: string | number;
  username?: string;
  email?: string;
  is_admin?: boolean;
  isAdmin?: boolean;
  role?: string;
  roles?: string[] | string;
  permissions?: string[] | string;
  scopes?: string[] | string;
}

// ========================================
// JSON/Schema Types
// ========================================

/** Generic JSON value type */
export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

/** Generic JSON object */
export type JsonObject = { [key: string]: JsonValue };

// ========================================
// SSE/Streaming Types
// ========================================

/** SSE message handler for text deltas */
export type SSEMessageHandler = (delta: string) => void;

/** SSE JSON handler for parsed JSON objects */
export type SSEJSONHandler = (json: JsonObject) => void;

/** SSE connection options */
export interface SSEOptions {
  method?: string;
  headers?: Record<string, string>;
  body?: string;
  signal?: AbortSignal;
  credentials?: RequestCredentials;
}

// ========================================
// Form/Validation Types
// ========================================

/** Form state type - record of string keys to unknown values */
export type FormState = Record<string, unknown>;

/** Validation result - array of error messages */
export type ValidationErrors = string[];

// ========================================
// Request History Types
// ========================================

/** Request history item for API debugging */
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
