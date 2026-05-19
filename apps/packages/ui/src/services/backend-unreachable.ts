export type BackendUnreachableSource = "background" | "direct"

export type BackendUnreachableRecentRequestError = {
  method: string
  path: string
  status?: number
  error: string
  source: BackendUnreachableSource
  at: string
  ageMs: number
}

/**
 * Subtypes distinguish the _reason_ the backend is unreachable so that the UI
 * can show an actionable message and fix hint instead of a generic error.
 *
 * - `cors`               - browser blocked a cross-origin request
 * - `connection_refused`  - server not running / unreachable
 * - `auth_failed`         - 401 Unauthorized
 * - `forbidden`           - 403 Forbidden
 * - `timeout`             - request timed out
 * - `server_error`        - 5xx response
 * - `generic_transport`   - unrecognised transport-level failure
 */
export type BackendUnreachableSubtype =
  | "cors"
  | "connection_refused"
  | "auth_failed"
  | "forbidden"
  | "timeout"
  | "server_error"
  | "generic_transport"

export type BackendUnreachableClassification = {
  kind: "backend_unreachable"
  subtype: BackendUnreachableSubtype
  title: string
  message: string
  fixHint?: string
  rawMessage: string
  status?: number
  method?: string
  path?: string
  source?: BackendUnreachableSource
  recentRequestError?: BackendUnreachableRecentRequestError
  diagnostics: {
    matchedPattern: string
    status?: number
    recentRequestErrorAgeMs?: number
  }
}

export type BackendUnreachableOther = {
  kind: "other"
  rawMessage: string
  status?: number
  diagnostics: {
    reason: "abort_like" | "not_transport"
    matchedPattern?: string
  }
}

export type BackendUnreachableClassificationResult =
  | BackendUnreachableClassification
  | BackendUnreachableOther

export type BackendUnreachableClassifierOptions = {
  nowMs?: number
  recentRequestError?: unknown
  recentRequestErrorFreshnessMs?: number
  serverUrl?: string
}

type ErrorLike = {
  message?: unknown
  name?: unknown
  code?: unknown
  status?: unknown
  method?: unknown
  path?: unknown
  source?: unknown
}

export type BackendUnreachablePattern = {
  id: string
  pattern: RegExp
}

type RecentRequestErrorCandidate = {
  method?: unknown
  path?: unknown
  status?: unknown
  error?: unknown
  source?: unknown
  at?: unknown
}

export const BACKEND_UNREACHABLE_TRANSPORT_MESSAGE_PATTERNS: readonly BackendUnreachablePattern[] = [
  {
    id: "network_error_when_attempting_to_fetch_resource",
    pattern: /NetworkError when attempting to fetch resource\.?/i
  },
  {
    id: "failed_to_fetch_exact",
    pattern: /^(?:TypeError:\s*)?Failed to fetch\.?$/i
  }
]

/**
 * Patterns that strongly suggest a CORS block rather than a generic network
 * failure. Browsers surface these messages when a cross-origin request is
 * rejected by the server's CORS policy.
 */
export const BACKEND_UNREACHABLE_CORS_MESSAGE_PATTERNS: readonly BackendUnreachablePattern[] = [
  {
    id: "cors_blocked_by_policy",
    pattern: /\bblocked by CORS policy\b/i
  },
  {
    id: "cors_no_access_control_allow_origin",
    pattern: /\bNo 'Access-Control-Allow-Origin' header\b/i
  },
  {
    id: "cors_cross_origin_request_blocked",
    pattern: /\bCross-Origin Request Blocked\b/i
  },
  {
    id: "cors_request_blocked",
    pattern: /\bCORS request did not succeed\b/i
  }
]

/**
 * Patterns indicating a connection-level refusal (server not listening).
 */
export const BACKEND_UNREACHABLE_CONN_REFUSED_MESSAGE_PATTERNS: readonly BackendUnreachablePattern[] = [
  {
    id: "econnrefused",
    pattern: /\bECONNREFUSED\b/i
  },
  {
    id: "connection_refused",
    pattern: /\bconnection refused\b/i
  },
  {
    id: "net_err_connection_refused",
    pattern: /\bnet::ERR_CONNECTION_REFUSED\b/i
  }
]

/**
 * Patterns indicating a request timeout.
 */
export const BACKEND_UNREACHABLE_TIMEOUT_MESSAGE_PATTERNS: readonly BackendUnreachablePattern[] = [
  {
    id: "timeout_exceeded",
    pattern: /\btimeout\b.*\bexceeded\b/i
  },
  {
    id: "request_timeout",
    pattern: /\brequest\s+timed?\s*out\b/i
  },
  {
    id: "timeout_error_name",
    pattern: /^TimeoutError$/i
  },
  {
    id: "net_err_timed_out",
    pattern: /\bnet::ERR_TIMED_OUT\b/i
  },
  {
    id: "etimedout",
    pattern: /\bETIMEDOUT\b/i
  }
]

export const BACKEND_UNREACHABLE_ABORT_MESSAGE_PATTERNS: readonly BackendUnreachablePattern[] = [
  {
    id: "abort_error_name",
    pattern: /^AbortError$/i
  },
  {
    id: "request_aborted_code",
    pattern: /^REQUEST_ABORTED$/i
  },
  {
    id: "operation_was_aborted",
    pattern: /^The operation was aborted\.?$/i
  }
]

export const DEFAULT_RECENT_REQUEST_ERROR_FRESHNESS_MS = 2 * 60_000

const normalizeString = (value: unknown): string =>
  typeof value === "string" ? value.trim() : ""

const getErrorLike = (error: unknown): ErrorLike =>
  typeof error === "object" && error !== null ? (error as ErrorLike) : {}

const getRawMessage = (error: unknown): string => {
  if (typeof error === "string") {
    return error.trim()
  }

  const candidate = normalizeString(getErrorLike(error).message)
  if (candidate) {
    return candidate
  }

  if (error instanceof Error && error.message) {
    return error.message.trim()
  }

  return normalizeString(String(error ?? ""))
}

const getRawName = (error: unknown): string => normalizeString(getErrorLike(error).name)

const getRawCode = (error: unknown): string => normalizeString(getErrorLike(error).code)

const parseStatus = (value: unknown): number | undefined => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value
  }

  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : undefined
  }

  return undefined
}

const getStatus = (error: unknown): number | undefined =>
  parseStatus(getErrorLike(error).status)

const findMatchingPattern = (
  value: string,
  patterns: readonly BackendUnreachablePattern[]
): BackendUnreachablePattern | undefined => patterns.find(({ pattern }) => pattern.test(value))

const isAbortLike = (error: unknown, rawMessage: string): boolean => {
  const rawName = getRawName(error)
  const rawCode = getRawCode(error)

  return (
    Boolean(findMatchingPattern(rawMessage, BACKEND_UNREACHABLE_ABORT_MESSAGE_PATTERNS)) ||
    /^AbortError$/i.test(rawName) ||
    /^REQUEST_ABORTED$/i.test(rawCode)
  )
}

const parseRecentRequestError = (
  value: unknown,
  nowMs: number,
  freshnessLimitMs: number
): BackendUnreachableRecentRequestError | undefined => {
  if (typeof value !== "object" || value === null) {
    return undefined
  }

  const candidate = value as RecentRequestErrorCandidate
  const method = normalizeString(candidate.method)
  const path = normalizeString(candidate.path)
  const error = normalizeString(candidate.error)
  const source = normalizeString(candidate.source)
  const at = normalizeString(candidate.at)
  const status = parseStatus(candidate.status)

  if (!method || !path || !error || !at) {
    return undefined
  }

  if (source !== "background" && source !== "direct") {
    return undefined
  }

  const parsedAt = Date.parse(at)
  if (!Number.isFinite(parsedAt)) {
    return undefined
  }

  const ageMs = nowMs - parsedAt
  if (!Number.isFinite(ageMs) || ageMs < 0 || ageMs > freshnessLimitMs) {
    return undefined
  }

  return {
    method,
    path,
    status,
    error,
    source,
    at,
    ageMs
  }
}

const isTransportFailure = (
  error: unknown,
  rawMessage: string
): BackendUnreachablePattern | undefined => {
  // Check explicit CORS patterns first — they are transport failures too
  const corsMatch = findMatchingPattern(rawMessage, BACKEND_UNREACHABLE_CORS_MESSAGE_PATTERNS)
  if (corsMatch) {
    return corsMatch
  }

  // Connection refused patterns
  const connMatch = findMatchingPattern(rawMessage, BACKEND_UNREACHABLE_CONN_REFUSED_MESSAGE_PATTERNS)
  if (connMatch) {
    return connMatch
  }

  // Timeout patterns
  const timeoutMatch = findMatchingPattern(rawMessage, BACKEND_UNREACHABLE_TIMEOUT_MESSAGE_PATTERNS)
  if (timeoutMatch) {
    return timeoutMatch
  }

  const match = findMatchingPattern(rawMessage, BACKEND_UNREACHABLE_TRANSPORT_MESSAGE_PATTERNS)
  if (match) {
    return match
  }

  const status = getStatus(error)
  if (status === 0 && /\b(fetch|network)\b/i.test(rawMessage)) {
    return {
      id: "status_zero_transport_failure",
      pattern: /\b(fetch|network)\b/i
    }
  }

  // HTTP error statuses that indicate backend-level issues
  if (status === 401 || status === 403 || (status !== undefined && status >= 500 && status < 600)) {
    return {
      id: `http_status_${status}`,
      pattern: new RegExp(String(status))
    }
  }

  // Timeout-like error names
  const rawName = getRawName(error)
  if (/^TimeoutError$/i.test(rawName)) {
    return {
      id: "timeout_error_name",
      pattern: /^TimeoutError$/i
    }
  }

  return undefined
}

type SubtypeCopy = {
  subtype: BackendUnreachableSubtype
  title: string
  message: string
  fixHint?: string
}

/**
 * Determines the error subtype and corresponding user-facing copy from the
 * matched pattern, raw message, and HTTP status.
 */
// TODO(i18n): extract user-facing strings (title, message, fixHint) to i18n resources
const resolveSubtype = (
  matchedPatternId: string,
  rawMessage: string,
  status: number | undefined,
  serverUrl?: string
): SubtypeCopy => {
  const url = serverUrl ?? "the configured server URL"

  // --- CORS ---
  if (
    matchedPatternId.startsWith("cors_") ||
    /\bCORS\b/i.test(rawMessage) ||
    /\bAccess-Control-Allow-Origin\b/i.test(rawMessage)
  ) {
    return {
      subtype: "cors",
      title: "Cross-origin request blocked",
      message: `The API server at ${url} doesn't allow requests from this origin.`,
      fixHint:
        "Ensure the API and WebUI are served from the same origin, or add this origin to ALLOWED_ORIGINS in the server configuration."
    }
  }

  // --- Auth failed (401) ---
  if (status === 401 || matchedPatternId === "http_status_401") {
    return {
      subtype: "auth_failed",
      title: "Authentication failed",
      message: "Your API key or session may be invalid or expired.",
      fixHint: "Check your API key in Settings, or log in again."
    }
  }

  // --- Forbidden (403) ---
  if (status === 403 || matchedPatternId === "http_status_403") {
    return {
      subtype: "forbidden",
      title: "Access denied",
      message: "You don't have permission for this action.",
      fixHint: "Check that your account has the required permissions, or contact the server administrator."
    }
  }

  // --- Timeout ---
  if (
    matchedPatternId.startsWith("timeout_") ||
    matchedPatternId === "net_err_timed_out" ||
    matchedPatternId === "etimedout" ||
    /\btimeout\b/i.test(rawMessage) ||
    /\btimed?\s*out\b/i.test(rawMessage)
  ) {
    return {
      subtype: "timeout",
      title: "Server took too long to respond",
      message: "The request to the API server timed out.",
      fixHint:
        "The server may be overloaded or still starting up. Try again in a moment."
    }
  }

  // --- Server error (5xx) ---
  if (status !== undefined && status >= 500 && status < 600) {
    return {
      subtype: "server_error",
      title: "Server encountered an error",
      message: `The API server returned an error (HTTP ${status}).`,
      fixHint: "Check the server logs for details."
    }
  }

  // --- Connection refused ---
  if (
    matchedPatternId.startsWith("econnrefused") ||
    matchedPatternId.startsWith("connection_refused") ||
    matchedPatternId.startsWith("net_err_connection_refused") ||
    /\b(ECONNREFUSED|connection refused|ERR_CONNECTION_REFUSED)\b/i.test(rawMessage)
  ) {
    return {
      subtype: "connection_refused",
      title: "Cannot connect to the API server",
      message: `The server at ${url} is not accepting connections.`,
      fixHint: `Make sure the tldw server is running. Check with: curl ${url}/api/v1/health`
    }
  }

  // --- Generic transport failure ---
  // For "Failed to fetch" and "NetworkError when attempting to fetch resource"
  // without more specific context, this is most likely the server being down
  // or a CORS issue. We lean toward connection_refused since it's the most
  // common cause and actionable.
  return {
    subtype: "connection_refused",
    title: "Cannot connect to the API server",
    message: `Cannot reach the API server at ${url}.`,
    fixHint: `Make sure the tldw server is running and reachable from this browser. Check with: curl ${url}/api/v1/health`
  }
}

const readRequestContext = (
  error: unknown,
  recentRequestError?: BackendUnreachableRecentRequestError
): Pick<
  BackendUnreachableClassification,
  "method" | "path" | "source" | "status"
> => {
  const errorLike = getErrorLike(error)
  const method = normalizeString(errorLike.method) || recentRequestError?.method || ""
  const path = normalizeString(errorLike.path) || recentRequestError?.path || ""
  const sourceValue = normalizeString(errorLike.source)
  const source =
    sourceValue === "background" || sourceValue === "direct"
      ? (sourceValue as BackendUnreachableSource)
      : recentRequestError?.source
  const status = getStatus(error) ?? recentRequestError?.status

  return {
    ...(method ? { method } : {}),
    ...(path ? { path } : {}),
    ...(source ? { source } : {}),
    ...(typeof status === "number" ? { status } : {})
  }
}

export const classifyBackendUnreachableError = (
  error: unknown,
  options: BackendUnreachableClassifierOptions = {}
): BackendUnreachableClassificationResult => {
  const rawMessage = getRawMessage(error)
  const status = getStatus(error)
  const abortLike = isAbortLike(error, rawMessage)

  if (abortLike) {
    return {
      kind: "other",
      rawMessage,
      status,
      diagnostics: {
        reason: "abort_like"
      }
    }
  }

  if (!rawMessage) {
    return {
      kind: "other",
      rawMessage,
      status,
      diagnostics: {
        reason: "not_transport"
      }
    }
  }

  const transportMatch = isTransportFailure(error, rawMessage)
  if (!transportMatch) {
    return {
      kind: "other",
      rawMessage,
      status,
      diagnostics: {
        reason: "not_transport"
      }
    }
  }

  const nowMs = options.nowMs ?? Date.now()
  const freshnessLimitMs =
    options.recentRequestErrorFreshnessMs ?? DEFAULT_RECENT_REQUEST_ERROR_FRESHNESS_MS
  const recentRequestError = parseRecentRequestError(
    options.recentRequestError,
    nowMs,
    freshnessLimitMs
  )
  const requestContext = readRequestContext(error, recentRequestError)

  // Resolve the server URL for use in messages
  const serverUrl = options.serverUrl

  const subtypeCopy = resolveSubtype(
    transportMatch.id,
    rawMessage,
    status,
    serverUrl
  )

  return {
    kind: "backend_unreachable",
    subtype: subtypeCopy.subtype,
    title: subtypeCopy.title,
    message: subtypeCopy.message,
    ...(subtypeCopy.fixHint ? { fixHint: subtypeCopy.fixHint } : {}),
    rawMessage,
    ...requestContext,
    ...(recentRequestError ? { recentRequestError } : {}),
    diagnostics: {
      matchedPattern: transportMatch.id,
      ...(typeof status === "number" ? { status } : {}),
      ...(recentRequestError ? { recentRequestErrorAgeMs: recentRequestError.ageMs } : {})
    }
  }
}
