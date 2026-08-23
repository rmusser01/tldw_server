import { browser } from "wxt/browser"
import { createSafeStorage } from "@/utils/safe-storage"
import { formatErrorMessage } from "@/utils/format-error-message"
import {
  isUnsafeMethod,
  parseRetryAfter,
  readBrowserCookie,
  resolveBrowserRequestTransport,
  tldwRequest
} from "@/services/tldw/request-core"
import { isHostedTldwDeployment } from "@/services/tldw/deployment-mode"
import {
  isCookieSessionBrowserTransport,
  resolveAdvancedRequestTransportGuard
} from "@/services/tldw/browser-networking"
import { getRuntimeSingleUserApiKeyOverride } from "@/services/tldw/runtime-auth-override"
import {
  resolveDirectBrowserConfig as resolveDirectConfig,
  type DirectRuntimeStorage
} from "@/services/tldw/direct-browser-config"
import {
  hasNewerCurrentAccessToken,
  storeRefreshRotationIfCurrent,
  waitForNewerCurrentAccessToken
} from "@/services/tldw/single-user-credential"
import {
  BACKEND_UNREACHABLE_EVENT,
  type BackendUnreachableDetail
} from "@/services/request-events"
import {
  asValidatedHttpStatus,
  sanitizeRagProviderFailure
} from "@/services/rag/provider-error-contract"
import type {
  AllowedMethodFor,
  AllowedPath,
  ClientPathOrUrlWithQuery,
  ClientPathRuntimeWithQuery,
  PathOrUrl,
  UpperLower
} from "@/services/tldw/openapi-guard"
import {
  isAbsoluteUrlAllowlisted,
  isSameOriginAbsoluteUrlForConfiguredServer
} from "@/utils/absolute-url-guard"
import type {
  ServicePromptTargetConfig,
  TldwConfig
} from "@/services/tldw/TldwApiClient"
import {
  createServicePromptScopeChangedError,
  isRequestConfigScopeChangedError,
  isServicePromptRequestPath,
  servicePromptPrincipalMatches,
  servicePromptRefreshLineageMatches,
  servicePromptSingleUserApiKeyScopeMatches,
  servicePromptTargetsMatch
} from "@/services/tldw/service-prompt-scope-error"
import { deriveScopedUserId } from "@/utils/media-navigation-scope"

const ERROR_LOG_THROTTLE_MS = 15_000
const RATE_LIMIT_LOG_THROTTLE_MS = 60_000
const ERROR_LOG_MAX_ENTRIES = 200
const BACKEND_UNREACHABLE_EVENT_THROTTLE_MS = 5_000
const STREAM_RUNTIME_PING_TIMEOUT_MS = 400
const STREAM_RUNTIME_HEALTH_TTL_MS = 30_000
const STREAM_QUEUE_DRAIN_BATCH_LIMIT = 32
const STREAM_QUEUE_DRAIN_SLICE_MS = 12
const SAFE_RUNTIME_MESSAGE_TIMEOUT_MS = 3_000
const UNSAFE_RUNTIME_MESSAGE_TIMEOUT_FLOOR_MS = 5_000
const RAG_STREAM_ABORT_MESSAGE = "RAG stream request was aborted."
// The MV3 worker only replies to an unsafe (write) request once the whole
// server operation finishes, so this messaging-ack timeout must cover the
// longest normal generation/ingest (non-stream chat, media kickoff, export)
// rather than the ~10s it used to be — a 10s cap killed in-flight writes that
// the worker had actually completed, losing the result. A genuinely dead
// worker still rejects fast (connection errors), so this only affects the
// slow-but-alive case. Keep it above the generation request-timeout default.
const DEFAULT_UNSAFE_RUNTIME_MESSAGE_TIMEOUT_MS = 130_000
const DEFAULT_UPLOAD_RUNTIME_MESSAGE_TIMEOUT_MS = 130_000
const ABSOLUTE_URL_BLOCK_ERROR =
  "Direct stream fallback is allowed only for allowlisted absolute URLs."
const BACKEND_UNREACHABLE_PATTERN =
  /(networkerror|failed to fetch|network error|load failed|err_connection|could not establish connection|receiving end does not exist)/i
const WORKSPACE_MIGRATION_CHUNK_PATH_PATTERN =
  /\/api\/v1\/workspaces\/migrations\/[^/?#]+\/chunks\/[^/?#]+/
const WORKSPACE_CONTEXT_REFRESH_PATH_PATTERN =
  /\/api\/v1\/workspaces\/(?!migrations(?:\/|$))[^/?#]+\/(?:context|sources)(?:[?#]|$)/
const WORKSPACE_UPSERT_RECONCILE_PATH_PATTERN =
  /\/api\/v1\/workspaces\/(?!migrations(?:[/?#]|$))[^/?#]+(?:[?#]|$)/
const RESEARCH_WORKSPACE_CHAT_COMMANDS_BOOTSTRAP_PATH_PATTERN =
  /\/api\/v1\/chat\/commands(?:[?#]|$)/
const OPTIONAL_AUDIO_VOICE_BOOTSTRAP_PATH_PATTERN =
  /\/api\/v1\/audio\/voices(?:\/catalog)?(?:[?#]|$)/
const OPTIONAL_INGESTION_SOURCE_CAPABILITIES_PATH_PATTERN =
  /\/api\/v1\/ingestion-sources\/capabilities(?:[?#]|$)/
const errorLogHistory = new Map<string, number>()
let lastBackendUnreachableEventAt = 0
let lastStreamRuntimeHealthCheckAt = 0
let streamRuntimePortUsable: boolean | null = null
let runtimeRequestSequence = 0

const createRuntimeRequestId = (): string => {
  try {
    const randomId = globalThis.crypto?.randomUUID?.()
    if (randomId) return randomId
  } catch {
    // Fall back to a locally unique id when randomUUID is unavailable.
  }
  runtimeRequestSequence += 1
  return `tldw-${Date.now()}-${runtimeRequestSequence}-${Math.random().toString(36).slice(2, 10)}`
}

const cancelRuntimeWorkerRequest = (requestId?: string): void => {
  if (!requestId) return
  try {
    void Promise.resolve(
      browser.runtime.sendMessage({
        type: "tldw:cancel-request",
        payload: { requestId }
      })
    ).catch(() => undefined)
  } catch {
    // Cancellation is best effort when the extension context is closing.
  }
}

const normalizeKnownPathQuirks = <P extends PathOrUrl>(rawPath: P): P => {
  if (typeof rawPath !== "string") return rawPath
  return rawPath
    .replace("/api/v1/media/?", "/api/v1/media?")
    .replace("/api/v1/files/?", "/api/v1/files?") as P
}

const isAudioStudioArtifactMediaPath = (path: string): boolean =>
  /\/api\/v1\/audio-studio\/projects\/[^/?#]+\/artifacts\/[^/?#]+\/media(?:[?#]|$)/.test(path)

const normalizeExpectedStatuses = (statuses: unknown): Set<number> => {
  if (!Array.isArray(statuses)) return new Set()
  return new Set(
    statuses
      .map((status) => Number(status))
      .filter(
        (status) =>
          Number.isFinite(status) &&
          status >= 100 &&
          status <= 599
      )
      .map((status) => Math.trunc(status))
  )
}

// Origin-allowlist / same-origin helpers (parseHttpOrigin,
// isAbsoluteUrlAllowlisted, isSameOriginAbsoluteUrlForConfiguredServer) are
// imported from the canonical utils/absolute-url-guard module — behaviour is
// unchanged (this file's copies were byte-for-byte identical to the guard's).

const extractHttpStatus = (value: unknown): number | null => {
  const statusCandidate = (value as { status?: unknown } | null)?.status
  if (typeof statusCandidate === "number" && Number.isFinite(statusCandidate)) {
    const status = Math.trunc(statusCandidate)
    if (status >= 100 && status <= 599) return status
  }
  const message = value instanceof Error ? value.message : String(value || "")
  const match = message.match(/\bhttp\s+(\d{3})\b/i)
  if (!match) return null
  const parsed = Number(match[1])
  return Number.isFinite(parsed) ? parsed : null
}

const isRateLimitEntry = (entry: { status?: number; error?: string }): boolean => {
  if (entry.status === 429) return true
  const msg = String(entry.error || "").toLowerCase()
  return msg.includes("rate limit") || msg.includes("429")
}

const shouldRecordRequestError = (entry: {
  method: string
  path: string
  status?: number
  error?: string
  source: "background" | "direct"
}): boolean => {
  const now = Date.now()
  const key = `${entry.source}:${entry.method}:${entry.path}:${entry.status ?? "na"}:${entry.error ?? ""}`
  const lastAt = errorLogHistory.get(key)
  const throttleMs = isRateLimitEntry(entry)
    ? RATE_LIMIT_LOG_THROTTLE_MS
    : ERROR_LOG_THROTTLE_MS
  if (lastAt && now - lastAt < throttleMs) return false
  errorLogHistory.set(key, now)
  if (errorLogHistory.size > ERROR_LOG_MAX_ENTRIES) {
    const sorted = Array.from(errorLogHistory.entries()).sort((a, b) => a[1] - b[1])
    const overflow = sorted.length - ERROR_LOG_MAX_ENTRIES
    for (let i = 0; i < overflow; i++) {
      errorLogHistory.delete(sorted[i][0])
    }
  }
  return true
}

const REDACTED_VALUE = "[REDACTED]"
const SENSITIVE_KEY_FRAGMENTS = [
  "stack",
  "trace",
  "sql",
  "query",
  "password",
  "passwd",
  "token",
  "secret",
  "path",
  "headers",
  "internalid",
  "authorization",
  "cookie",
  "api_key",
  "apikey",
  "access_key",
  "accesskey",
  "private",
  "credential",
  "session",
  "bearer"
]

const isSensitiveKey = (key: string): boolean => {
  const normalized = key.toLowerCase().replace(/[\s-]/g, "_")
  return SENSITIVE_KEY_FRAGMENTS.some((fragment) => normalized.includes(fragment))
}

// Redact known sensitive fields (stack/trace/sql/query/secret/headers/etc.) recursively.
export const sanitizeResponseData = (
  value: unknown,
  seen: WeakSet<object> = new WeakSet()
): unknown => {
  if (value == null || typeof value !== "object") return value
  if (seen.has(value as object)) return REDACTED_VALUE
  seen.add(value as object)

  if (Array.isArray(value)) {
    return value.map((entry) => sanitizeResponseData(entry, seen))
  }

  const result: Record<string, unknown> = {}
  Object.entries(value as Record<string, unknown>).forEach(([key, entry]) => {
    if (isSensitiveKey(key)) {
      result[key] = REDACTED_VALUE
      return
    }
    result[key] = sanitizeResponseData(entry, seen)
  })
  return result
}

type NoFallbackError = Error & {
  __tldwNoDirectFallback?: true
  __tldwExtensionTimeout?: true
}

const markNoFallbackError = (
  error: Error,
  options?: { timeout?: boolean }
): NoFallbackError => {
  const marked = error as NoFallbackError
  marked.__tldwNoDirectFallback = true
  if (options?.timeout) {
    marked.__tldwExtensionTimeout = true
  }
  return marked
}

const isNoFallbackError = (error: unknown): error is NoFallbackError => {
  return Boolean((error as NoFallbackError | null)?.__tldwNoDirectFallback)
}

const isExtensionTimeoutError = (error: unknown): boolean => {
  if ((error as NoFallbackError | null)?.__tldwExtensionTimeout) return true
  const message =
    error instanceof Error ? error.message : String(error || "")
  return message.toLowerCase().includes("extension messaging timeout")
}

const isSafeFallbackMethod = (method: unknown): boolean => {
  const methodUpper = String(method || "GET").toUpperCase()
  return methodUpper === "GET" || methodUpper === "HEAD" || methodUpper === "OPTIONS"
}

const resolveRuntimeMessageTimeoutMs = (
  method: unknown,
  override?: number
): number => {
  if (isSafeFallbackMethod(method)) return SAFE_RUNTIME_MESSAGE_TIMEOUT_MS
  const configured = Number(override)
  if (Number.isFinite(configured) && configured > 0) {
    return Math.max(UNSAFE_RUNTIME_MESSAGE_TIMEOUT_FLOOR_MS, configured)
  }
  return DEFAULT_UNSAFE_RUNTIME_MESSAGE_TIMEOUT_MS
}

const isIdempotentWriteFallbackAllowed = (
  method: unknown,
  path: unknown,
  body: unknown
): boolean => {
  if (isSafeFallbackMethod(method)) return false
  if (String(method || "GET").toUpperCase() !== "POST") return false
  const normalizedPath = String(path || "")
    .split("?")[0]
    .replace(/\/+$/, "")
  if (normalizedPath !== "/api/v1/web-clipper/save") return false
  const clipId = (body as { clip_id?: unknown } | null)?.clip_id
  return typeof clipId === "string" && clipId.trim().length > 0
}

const isExtensionTransportFailure = (error: unknown): boolean => {
  if (isNoFallbackError(error)) return false
  const message =
    error instanceof Error ? error.message : String(error || "")
  const normalized = message.toLowerCase()
  return (
    normalized.includes("extension messaging timeout") ||
    normalized.includes("could not establish connection") ||
    normalized.includes("receiving end does not exist") ||
    normalized.includes("message port closed before a response was received") ||
    normalized.includes("extension context invalidated") ||
    normalized.includes("stream disconnected") ||
    normalized.includes("failed to fetch") ||
    normalized.includes("network error")
  )
}

const isProvenNoReceiverError = (error: unknown): boolean => {
  const message = error instanceof Error ? error.message : String(error || "")
  const normalized = message.toLowerCase()
  return normalized.includes("receiving end does not exist") ||
    normalized.includes("could not establish connection")
}

type RequestAbortError = Error & {
  status?: number
  code?: string
  details?: unknown
}

const isAbortErrorMessage = (value?: string) =>
  typeof value === "string" && value.toLowerCase().includes("abort")

const readErrorMessage = (error: unknown, fallback = "Aborted") =>
  error instanceof Error && error.message ? error.message : fallback
const createAbortError = (
  message?: string,
  status?: number,
  details?: unknown
): RequestAbortError => {
  const abortError = new Error(message || "Aborted") as RequestAbortError
  abortError.name = "AbortError"
  abortError.status = typeof status === "number" ? status : 0
  abortError.code = "REQUEST_ABORTED"
  if (typeof details !== "undefined") {
    abortError.details = sanitizeResponseData(details)
  }
  return abortError
}

type StreamInterruptedError = Error & { code?: string; interrupted?: true }

// Surfaced when a non-idempotent streamed request (chat completions,
// complete-v2) loses its extension port before/around the first token. We must
// NOT replay it (that would double-generate and persist a duplicate message),
// so instead we raise a clear error the caller can show as an interruption.
const createStreamInterruptedError = (message: string): StreamInterruptedError => {
  const error = new Error(message) as StreamInterruptedError
  error.name = "StreamInterruptedError"
  error.code = "STREAM_INTERRUPTED"
  error.interrupted = true
  return error
}

const shouldNotifyBackendUnavailable = (entry: {
  method: string
  path: string
  status?: number
  error?: string
}): boolean => {
  const path = String(entry.path || "")
  // Restrict notifications to API requests only.
  if (!path.includes("/api/")) return false
  if (WORKSPACE_MIGRATION_CHUNK_PATH_PATTERN.test(path)) return false
  const method = String(entry.method || "GET").toUpperCase()
  if (
    (method === "GET" && WORKSPACE_CONTEXT_REFRESH_PATH_PATTERN.test(path)) ||
    (method === "GET" &&
      RESEARCH_WORKSPACE_CHAT_COMMANDS_BOOTSTRAP_PATH_PATTERN.test(path)) ||
    (method === "GET" &&
      OPTIONAL_AUDIO_VOICE_BOOTSTRAP_PATH_PATTERN.test(path)) ||
    (method === "GET" &&
      OPTIONAL_INGESTION_SOURCE_CAPABILITIES_PATH_PATTERN.test(path)) ||
    (method === "PUT" && WORKSPACE_UPSERT_RECONCILE_PATH_PATTERN.test(path))
  ) {
    return false
  }
  if (entry.status === 0) return true
  return BACKEND_UNREACHABLE_PATTERN.test(String(entry.error || ""))
}

const notifyBackendUnavailable = (
  entry: {
    method: string
    path: string
    status?: number
    code?: string
    error?: string
    source: "background" | "direct"
  },
  eligible?: boolean
) => {
  if (typeof window === "undefined" || typeof window.dispatchEvent !== "function") {
    return
  }
  if (!(eligible ?? shouldNotifyBackendUnavailable(entry))) return
  const now = Date.now()
  if (now - lastBackendUnreachableEventAt < BACKEND_UNREACHABLE_EVENT_THROTTLE_MS) {
    return
  }
  lastBackendUnreachableEventAt = now

  const detail: BackendUnreachableDetail = {
    method: entry.method,
    path: entry.path,
    status: entry.status,
    code: entry.code,
    message: String(entry.error || "Network error"),
    source: entry.source,
    timestamp: now
  }
  try {
    window.dispatchEvent(
      new CustomEvent<BackendUnreachableDetail>(BACKEND_UNREACHABLE_EVENT, {
        detail
      })
    )
  } catch {
    // best-effort notification only
  }
}

export interface BgRequestInit<
  P extends PathOrUrl = AllowedPath,
  M extends AllowedMethodFor<P> = AllowedMethodFor<P>
> {
  path: P
  method?: UpperLower<M>
  headers?: Record<string, string>
  body?: any
  noAuth?: boolean
  timeoutMs?: number
  abortSignal?: AbortSignal
  responseType?: "json" | "text" | "arrayBuffer"
  returnResponse?: boolean
  preferDirect?: boolean
  suppressBackendUnavailableEvent?: boolean
  expectedStatuses?: number[]
  sanitizeRagProviderError?: boolean
  servicePromptConfig?: ServicePromptTargetConfig
  configSnapshot?: unknown
}

const resolveCurrentServicePromptConfig = async (
  storage: DirectRuntimeStorage,
  checked: ServicePromptTargetConfig
): Promise<TldwConfig> => {
  const stored = await resolveDirectConfig(storage)
  const runtimeApiKey = !isHostedTldwDeployment() &&
    checked.authMode === "single-user"
    ? String(getRuntimeSingleUserApiKeyOverride() || "").trim()
    : ""
  const current = runtimeApiKey
    ? { ...(stored ?? checked), apiKey: runtimeApiKey }
    : stored
  const singleUserApiKeyScopeMatches = current
    ? servicePromptSingleUserApiKeyScopeMatches(
        current,
        checked.expectedSingleUserApiKeyScope
      )
    : true
  if ((!current && !isHostedTldwDeployment()) ||
    (current && !servicePromptTargetsMatch(current, checked)) ||
    (current &&
      checked.expectedUserId !== null &&
      checked.expectedUserId !== undefined &&
      current.authMode === "multi-user" &&
      !isHostedTldwDeployment() &&
      current.authSource !== "cookie-session" &&
      !servicePromptPrincipalMatches(current, checked.expectedUserId)) ||
    (current &&
      !servicePromptRefreshLineageMatches(
        current,
        checked.expectedRefreshToken
      )) ||
    !singleUserApiKeyScopeMatches ||
    (current?.authMode === "multi-user" &&
      !isHostedTldwDeployment() &&
      current.authSource !== "cookie-session" &&
      !String(current.accessToken || "").trim())
  ) {
    throw createServicePromptScopeChangedError()
  }
  const effective = current ?? checked
  return {
    ...effective,
    serverUrl: checked.serverUrl,
    authMode: checked.authMode,
    authSource: checked.authSource,
    orgId: checked.orgId,
    apiKey: current?.apiKey,
    accessToken: current?.accessToken,
    refreshToken: current?.refreshToken
  }
}

// In-flight coalescing for idempotent GET requests: when several callers issue
// the same GET concurrently (a common pattern when many components mount on a
// page load and each fetches the same resource), share a single network request
// instead of firing N identical ones. Successful requests are not cached; 429s
// get a brief per-key cooldown so remount/retry loops do not hammer the server.
const inFlightGetRequests = new Map<string, Promise<unknown>>()
const rateLimitedGetResults = new Map<
  string,
  { expiresAt: number; rejected: boolean; value: unknown }
>()
const DEFAULT_RATE_LIMIT_GET_COOLDOWN_MS = 2_000
const MAX_RATE_LIMIT_GET_COOLDOWN_MS = 60_000

const normalizeGetScopeServer = (value: string): string | null => {
  try {
    const parsed = new URL(String(value || "").trim())
    if (!/^https?:$/.test(parsed.protocol)) return null
    return `${parsed.protocol.toLowerCase()}//${parsed.host.toLowerCase()}${parsed.pathname.replace(/\/+$/, "")}`
  } catch {
    return null
  }
}

const resolveGetRequestScopeFingerprint = async (): Promise<string | null> => {
  const storage = createSafeStorage({ area: "local" })
  const config = await resolveDirectConfig(storage)
  if (!config || config.authSource === "cookie-session") return null
  const server = normalizeGetScopeServer(config.serverUrl)
  if (!server) return null
  const authMode = String(config?.authMode || "unknown")
    .trim()
    .toLowerCase()
  const org = config?.orgId == null ? "none" : String(config.orgId)
  const principal = deriveScopedUserId({
    accessToken: config.accessToken,
    authMode: config.authMode
  })
  if (authMode === "multi-user" && principal === "user:anonymous") return null
  return `${server}:auth:${authMode}:org:${org}:${principal}`
}

const isRateLimitedResult = (value: unknown): boolean => {
  const status = extractHttpStatus(value)
  if (status === 429) return true
  if (!(value instanceof Error)) return false
  const message = value.message
  return /(?:\b429\b|rate limit|too many requests)/i.test(message)
}

const readRateLimitCooldownMs = (value: unknown): number => {
  const candidate = value as
    | {
        retryAfterMs?: unknown
        headers?: Record<string, string | undefined>
      }
    | null
    | undefined
  const retryAfterMs = Number(candidate?.retryAfterMs)
  if (Number.isFinite(retryAfterMs) && retryAfterMs > 0) {
    return Math.min(MAX_RATE_LIMIT_GET_COOLDOWN_MS, Math.max(500, retryAfterMs))
  }
  const retryAfterHeader =
    candidate?.headers?.["retry-after"] ?? candidate?.headers?.["Retry-After"]
  const parsedRetryAfter = parseRetryAfter(retryAfterHeader)
  if (typeof parsedRetryAfter === "number" && parsedRetryAfter > 0) {
    return Math.min(
      MAX_RATE_LIMIT_GET_COOLDOWN_MS,
      Math.max(500, parsedRetryAfter)
    )
  }
  return DEFAULT_RATE_LIMIT_GET_COOLDOWN_MS
}

const pruneRateLimitedGetResults = () => {
  const now = Date.now()
  for (const [key, entry] of rateLimitedGetResults) {
    if (entry.expiresAt <= now) {
      rateLimitedGetResults.delete(key)
    }
  }
  while (rateLimitedGetResults.size > ERROR_LOG_MAX_ENTRIES) {
    const oldestKey = rateLimitedGetResults.keys().next().value
    if (!oldestKey) break
    rateLimitedGetResults.delete(oldestKey)
  }
}

// Module-level single-flight for the web/direct fallback token refresh. Mirrors
// the extension worker's `refreshInFlight`: concurrent 401s (a common pattern
// when many components refetch on a page load) trigger exactly ONE refresh
// instead of a stampede that would each spend and rotate the refresh token,
// persisting a dead one.
let webRefreshInFlight: Promise<void> | null = null
const scopedWebRefreshes = new Map<string, Promise<void>>()

const scopedRefreshKey = (
  checked: ServicePromptTargetConfig,
  refreshToken: string
): string => JSON.stringify([
  checked.serverUrl ?? null,
  checked.authMode ?? null,
  checked.authSource ?? null,
  checked.orgId ?? null,
  refreshToken
])

const commitDirectRefresh = async (
  storage: DirectRuntimeStorage,
  checked: TldwConfig,
  capturedAccessToken: string,
  expectedRefreshToken: string,
  tokens: Readonly<{ accessToken: string; refreshToken: string }>
): Promise<TldwConfig> => {
  const stored = await storeRefreshRotationIfCurrent(
    storage,
    checked,
    expectedRefreshToken,
    tokens
  )
  const observedNewerToken = await hasNewerCurrentAccessToken(
    storage,
    checked,
    capturedAccessToken
  )
  const latest = await resolveDirectConfig(storage)
  const responseApplied = Boolean(
    latest &&
    String(latest.accessToken || "").trim() === tokens.accessToken &&
    String(latest.refreshToken || "").trim() === tokens.refreshToken
  )
  if (
    !latest ||
    latest.authMode !== "multi-user" ||
    !servicePromptTargetsMatch(latest, checked) ||
    (!stored && !observedNewerToken) ||
    (!responseApplied && !observedNewerToken)
  ) {
    throw createServicePromptScopeChangedError()
  }
  return latest
}

const refreshAuthDirect = async (
  storage: DirectRuntimeStorage,
  checked?: ServicePromptTargetConfig,
  originalConfig?: TldwConfig
): Promise<void> => {
  if (checked) {
    const cfg = originalConfig ??
      await resolveCurrentServicePromptConfig(storage, checked)
    const refreshToken = String(cfg.refreshToken || "").trim()
    const capturedAccessToken = String(cfg.accessToken || "").trim()
    if (!refreshToken) {
      throw new Error("Token refresh failed: no refresh token available")
    }
    if (
      originalConfig &&
      await hasNewerCurrentAccessToken(
        storage,
        checked,
        capturedAccessToken
      )
    ) {
      return
    }
    const current = originalConfig
      ? await resolveCurrentServicePromptConfig(storage, checked)
      : cfg
    if (String(current.refreshToken || "").trim() !== refreshToken) {
      throw createServicePromptScopeChangedError()
    }
    const key = scopedRefreshKey(checked, refreshToken)
    let refresh = scopedWebRefreshes.get(key)
    if (!refresh) {
      refresh = (async () => {
        try {
          const resp = await tldwRequest(
            {
              path: "/api/v1/auth/refresh",
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: { refresh_token: refreshToken },
              noAuth: true
            },
            { getConfig: async () => cfg }
          )
          const tokens = (resp?.ok ? resp.data : null) as
            | { access_token?: string; refresh_token?: string }
            | null
          if (!tokens?.access_token) {
            throw new Error(
              `Token refresh failed: ${resp?.error || `no access token in refresh response (status ${resp?.status ?? "unknown"})`}`
            )
          }
          const stored = await storeRefreshRotationIfCurrent(
            storage,
            { ...checked, accessToken: capturedAccessToken },
            refreshToken,
            {
              accessToken: tokens.access_token,
              refreshToken: tokens.refresh_token || refreshToken
            }
          )
          if (!stored) {
            throw createServicePromptScopeChangedError()
          }
        } catch (error) {
          if (isRequestConfigScopeChangedError(error)) throw error
          if (
            originalConfig &&
            await waitForNewerCurrentAccessToken(
              storage,
              checked,
              capturedAccessToken
            )
          ) {
            return
          }
          throw error
        } finally {
          scopedWebRefreshes.delete(key)
        }
      })()
      scopedWebRefreshes.set(key, refresh)
    }
    await refresh
    return
  }

  if (!webRefreshInFlight) {
    webRefreshInFlight = (async () => {
      const cfg =
        (await resolveDirectConfig(storage)) || null
      const refreshToken = String((cfg?.refreshToken as string) || "").trim()
      const capturedAccessToken = String(cfg?.accessToken || "").trim()
      // Signal failure (throw) rather than resolving silently: request-core
      // treats a resolved refreshAuth as success and would retry with the stale
      // token. Throwing makes it mark the refresh as failed so a still-401 retry
      // surfaces "Session expired" instead of masking the failure.
      if (!refreshToken) {
        throw new Error("Token refresh failed: no refresh token available")
      }
      const resp = await tldwRequest(
        {
          path: "/api/v1/auth/refresh",
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: { refresh_token: refreshToken },
          noAuth: true
        },
        { getConfig: () => resolveDirectConfig(storage) }
      )
      const tokens = (resp?.ok ? resp.data : null) as
        | { access_token?: string; refresh_token?: string }
        | null
      if (!tokens?.access_token) {
        throw new Error(
          `Token refresh failed: ${resp?.error || `no access token in refresh response (status ${resp?.status ?? "unknown"})`}`
        )
      }
      if (!cfg || cfg.authMode !== "multi-user") {
        throw new Error("Token refresh failed: account configuration changed")
      }
      await commitDirectRefresh(
        storage,
        cfg,
        capturedAccessToken,
        refreshToken,
        {
          accessToken: tokens.access_token,
          refreshToken: tokens.refresh_token || refreshToken
        }
      )
    })().finally(() => {
      webRefreshInFlight = null
    })
  }
  await webRefreshInFlight
}

// Runtime for the web/direct fallback. Supplies a working `refreshAuth` so
// request-core's 401 refresh-and-retry runs in the browser (not just inside the
// extension worker), and single-flights it across concurrent callers.
const createDirectRuntime = (
  storage: DirectRuntimeStorage,
  servicePromptConfig?: ServicePromptTargetConfig,
  configSnapshot?: unknown
) => {
  let originalConfig: TldwConfig | undefined
  return {
    ...(servicePromptConfig ? { useRuntimeAuthOverride: false } : {}),
    getConfig: servicePromptConfig
      ? async () => {
          const current = await resolveCurrentServicePromptConfig(
            storage,
            servicePromptConfig
          )
          originalConfig ??= current
          return current
        }
      : configSnapshot !== undefined
        ? () => Promise.resolve(configSnapshot)
        : () => resolveDirectConfig(storage),
    refreshAuth: async () => {
      await refreshAuthDirect(
        storage,
        servicePromptConfig,
        originalConfig
      )
    }
  }
}

export async function bgRequest<
  T = any,
  P extends PathOrUrl = AllowedPath,
  M extends AllowedMethodFor<P> = AllowedMethodFor<P>
>(init: BgRequestInit<P, M>): Promise<T> {
  const method = String(init.method || "GET").toUpperCase()
  const coalescable =
    method === "GET" &&
    !init.body &&
    !init.abortSignal &&
    !init.responseType &&
    !init.preferDirect &&
    !init.suppressBackendUnavailableEvent &&
    !init.sanitizeRagProviderError &&
    !init.servicePromptConfig &&
    init.configSnapshot === undefined
  if (!coalescable) {
    return bgRequestImpl<T, P, M>(init)
  }
  const requestScope = await resolveGetRequestScopeFingerprint()
  if (!requestScope) {
    return bgRequestImpl<T, P, M>(init)
  }
  // Header keys are case-insensitive and object key order is not meaningful, so
  // normalize (lowercase + sort) for a stable key. Keep timeoutMs in the key so
  // GETs with different timeouts are not merged, and preserve the distinction
  // between "noAuth omitted" and "noAuth: false" (bgRequestImpl uses
  // hasOwnProperty(noAuth) to decide cross-origin auth suppression).
  const initHeaders = init.headers as Record<string, string> | undefined
  const normalizedHeaders = initHeaders
    ? Object.keys(initHeaders)
        .sort()
        .reduce<Record<string, string>>((acc, headerKey) => {
          acc[headerKey.toLowerCase()] = initHeaders[headerKey]
          return acc
        }, {})
    : null
  const expectedStatuses = Array.from(
    normalizeExpectedStatuses(init.expectedStatuses)
  ).sort((left, right) => left - right)
  const key = JSON.stringify({
    scope: requestScope,
    p: String(init.path),
    h: normalizedHeaders,
    noAuth: Object.prototype.hasOwnProperty.call(init, "noAuth")
      ? Boolean(init.noAuth)
      : "__unset__",
    returnResponse: Boolean(init.returnResponse),
    expectedStatuses,
    suppressBackendUnavailableEvent: Boolean(
      init.suppressBackendUnavailableEvent
    ),
    timeoutMs: typeof init.timeoutMs === "number" ? init.timeoutMs : null
  })
  const existing = inFlightGetRequests.get(key)
  if (existing) {
    return existing as Promise<T>
  }
  const rateLimited = rateLimitedGetResults.get(key)
  if (rateLimited) {
    if (rateLimited.expiresAt > Date.now()) {
      return rateLimited.rejected
        ? Promise.reject(rateLimited.value)
        : Promise.resolve(rateLimited.value as T)
    }
    rateLimitedGetResults.delete(key)
  }
  const rememberRateLimit = (value: unknown, rejected: boolean) => {
    if (!isRateLimitedResult(value)) return
    pruneRateLimitedGetResults()
    rateLimitedGetResults.set(key, {
      expiresAt: Date.now() + readRateLimitCooldownMs(value),
      rejected,
      value
    })
  }
  const promise = bgRequestImpl<T, P, M>(init)
    .then(
      (value) => {
        rememberRateLimit(value, false)
        return value
      },
      (error) => {
        rememberRateLimit(error, true)
        throw error
      }
    )
    .finally(() => {
      // Only clear the entry if it is still the promise we created — defensive in
      // case the map handling changes to allow overwrites in the future.
      if (inFlightGetRequests.get(key) === promise) {
        inFlightGetRequests.delete(key)
      }
    })
  inFlightGetRequests.set(key, promise)
  return promise as Promise<T>
}

async function bgRequestImpl<
  T = any,
  P extends PathOrUrl = AllowedPath,
  M extends AllowedMethodFor<P> = AllowedMethodFor<P>
>(init: BgRequestInit<P, M>): Promise<T> {
  const {
    path: rawPath,
    method = 'GET' as UpperLower<M>,
    headers = {},
    body,
    noAuth = false,
    timeoutMs,
    abortSignal,
    responseType,
    returnResponse,
    preferDirect = false,
    suppressBackendUnavailableEvent = false,
    expectedStatuses,
    sanitizeRagProviderError = false,
    servicePromptConfig,
    configSnapshot
  } = init
  if (servicePromptConfig) {
    if (!isServicePromptRequestPath(rawPath, method)) {
      throw new Error(
        "A Service Prompt config can only be used with Service Prompt requests."
      )
    }
  }
  const path = normalizeKnownPathQuirks(rawPath)
  const expectedStatusSet = normalizeExpectedStatuses(expectedStatuses)
  const isExpectedStatus = (status: unknown): boolean =>
    typeof status === "number" && expectedStatusSet.has(Math.trunc(status))
  const isAbsoluteUrl = typeof path === "string" && /^https?:/i.test(path)
  const noAuthExplicit = Object.prototype.hasOwnProperty.call(init, "noAuth")
  let resolvedNoAuth = noAuthExplicit ? noAuth : (noAuth || isAbsoluteUrl)
  if (!noAuthExplicit && isAbsoluteUrl) {
    const storage = createSafeStorage({ area: "local" })
    const cfg = servicePromptConfig ?? await resolveDirectConfig(storage)
    const sameOriginAbsolute = isSameOriginAbsoluteUrlForConfiguredServer(
      String(path),
      cfg as unknown as Record<string, unknown>
    )
    resolvedNoAuth = noAuth || !sameOriginAbsolute
  }
  const resolvedHeaders = headers
  const recordRequestError = async (entry: {
    method: string
    path: string
    status?: number
    code?: string
    error?: string
    source: "background" | "direct"
  }) => {
    try {
      if (!shouldRecordRequestError(entry)) return
      const storage = createSafeStorage({ area: "local" })
      const at = new Date().toISOString()
      const payload = { ...entry, at }
      const existing = (await storage.get<any[]>("__tldwRequestErrors").catch(() => [])) || []
      const next = Array.isArray(existing) ? existing : []
      next.unshift(payload)
      if (next.length > 20) next.length = 20
      await storage.set("__tldwRequestErrors", next)
      await storage.set("__tldwLastRequestError", payload)
    } catch {
      // best-effort logging only
    }
  }
  const buildRequestError = (
    msg: string,
    status?: number,
    details?: unknown,
    code?: string
  ): (Error & { status?: number; code?: string; details?: unknown }) => {
    if (isAbortErrorMessage(msg)) {
      return createAbortError(msg, status, details)
    }
    const error = new Error(`${msg} (${method} ${path})`) as Error & {
      status?: number
      code?: string
      details?: unknown
    }
    error.status = status
    if (code) {
      error.code = code
    }
    if (typeof details !== "undefined") {
      error.details = sanitizeResponseData(details)
    }
    return error
  }
  const shouldBypassBackground =
    responseType === "arrayBuffer" &&
    typeof path === "string" &&
    (path.includes("/api/v1/audio/") ||
      isAudioStudioArtifactMediaPath(path))
  const isArrayBufferLike = (value: unknown): boolean => {
    if (!value) return false
    if (value instanceof ArrayBuffer) return true
    if (typeof SharedArrayBuffer !== "undefined" && value instanceof SharedArrayBuffer) {
      return true
    }
    if (ArrayBuffer.isView?.(value)) return true
    if (typeof Blob !== "undefined" && value instanceof Blob) return true
    return false
  }
  type RuntimeResponsePayload = {
    ok: boolean
    error?: string
    status?: number
    data?: unknown
    headers?: Record<string, string>
  }
  type NormalizedRequestFailure = {
    message: string
    status?: number
    code?: string
    details?: unknown
  }
  const handleFailedResponse = async (
    resp: RuntimeResponsePayload,
    source: "background" | "direct"
  ): Promise<{
    error: Error & { status?: number; code?: string; details?: unknown }
    response: RuntimeResponsePayload
  }> => {
    const rawMessage = formatErrorMessage(
      resp?.error,
      `Request failed: ${resp?.status}`
    )
    const eligibleForBackendUnavailableEvent = shouldNotifyBackendUnavailable({
      method: String(method),
      path: String(path),
      status: resp?.status,
      error: rawMessage
    })
    const scopedError =
      servicePromptConfig &&
      isRequestConfigScopeChangedError({
        status: resp?.status,
        details: resp?.data
      })
        ? createServicePromptScopeChangedError()
        : null
    const sanitized: NormalizedRequestFailure = scopedError
      ? {
          message: scopedError.message,
          status: scopedError.status,
          details: scopedError.details
        }
      : sanitizeRagProviderError
      ? isAbortErrorMessage(rawMessage)
        ? {
            message: "Aborted",
            status: asValidatedHttpStatus(resp?.status)
          }
        : sanitizeRagProviderFailure({
            status: resp?.status,
            error: rawMessage,
            data: resp?.data
          })
      : {
          message: rawMessage,
          status: resp?.status,
          details: resp?.data
        }
    const diagnosticEntry = {
      method: String(method),
      path: String(path),
      status: sanitized.status,
      code: sanitized.code,
      error: sanitized.message,
      source
    }

    if (
      !isAbortErrorMessage(sanitized.message) &&
      !isExpectedStatus(resp?.status)
    ) {
      if (sanitized.code) {
        console.warn(
          "[tldw:request]",
          method,
          path,
          sanitized.status,
          sanitized.message,
          sanitized.code
        )
      } else {
        console.warn(
          "[tldw:request]",
          method,
          path,
          sanitized.status,
          sanitized.message
        )
      }
      await recordRequestError(diagnosticEntry)
      if (!suppressBackendUnavailableEvent) {
        notifyBackendUnavailable(
          diagnosticEntry,
          eligibleForBackendUnavailableEvent
        )
      }
    }

    return {
      error: buildRequestError(
        sanitized.message,
        sanitized.status,
        sanitized.details,
        sanitized.code
      ),
      response: sanitizeRagProviderError || scopedError
        ? {
            ...resp,
            error: sanitized.message,
            status: sanitized.status,
            data: sanitized.details
          }
        : resp
    }
  }
  const requestDirectArrayBufferFallback = async () => {
    const storage = createSafeStorage({ area: "local" })
    return await tldwRequest(
      {
        path,
        method,
        headers: resolvedHeaders,
        body,
        noAuth: resolvedNoAuth,
        timeoutMs,
        abortSignal,
        responseType
      },
      createDirectRuntime(storage, servicePromptConfig, configSnapshot)
    )
  }
  const resolveArrayBufferResponse = async (
    resp: RuntimeResponsePayload
  ): Promise<T> => {
    if (!resp?.ok || responseType !== "arrayBuffer" || isArrayBufferLike(resp.data)) {
      return (returnResponse ? resp : resp.data) as T
    }

    const fallback = await requestDirectArrayBufferFallback()
    if (!fallback) {
      const error = buildRequestError("Request failed: missing fallback response")
      throw error
    }
    if (!fallback.ok && !returnResponse) {
      const msg = formatErrorMessage(
        fallback.error,
        `Request failed: ${fallback.status}`
      )
      const error = buildRequestError(msg, fallback.status, fallback.data)
      throw error
    }
    return (returnResponse ? fallback : fallback.data) as T
  }
  const hasRuntimeMessage =
    !preferDirect &&
    Boolean(browser?.runtime?.sendMessage && browser?.runtime?.id)
  const methodIsSafeFallback = isSafeFallbackMethod(method)
  const allowIdempotentWriteFallback = isIdempotentWriteFallbackAllowed(
    method,
    path,
    body
  )
  const runtimeMessageTimeoutMs = resolveRuntimeMessageTimeoutMs(
    method,
    Number(timeoutMs)
  )

  // Some binary responses do not survive extension message serialization.
  if (shouldBypassBackground) {
    const storage = createSafeStorage({ area: "local" })
    const resp = await tldwRequest(
      {
        path,
        method,
        headers: resolvedHeaders,
        body,
        noAuth: resolvedNoAuth,
        timeoutMs,
        abortSignal,
        responseType
      },
      createDirectRuntime(storage, servicePromptConfig, configSnapshot)
    )
    if (!resp?.ok) {
      const failure = await handleFailedResponse(resp, "direct")
      if (!returnResponse) {
        throw failure.error
      }
      return failure.response as T
    }
    return (returnResponse ? resp : resp.data) as T
  }

  // If extension messaging is available, use it (extension context)
  try {
    if (hasRuntimeMessage) {
      const requestId =
        servicePromptConfig && abortSignal
          ? createRuntimeRequestId()
          : undefined
      const payload = {
        type: 'tldw:request',
        payload: {
          path,
          method,
          headers: resolvedHeaders,
          body,
          noAuth: resolvedNoAuth,
          timeoutMs,
          responseType,
          servicePromptConfig,
          ...(requestId ? { requestId } : {})
        }
      }

      if (!abortSignal) {
        // Add timeout to extension messaging - if service worker doesn't respond, fall back to direct request
        const messagePromiseNoSignal = browser.runtime.sendMessage(payload)
        const timeoutPromiseNoSignal = new Promise<null>((resolve) =>
          setTimeout(() => resolve(null), runtimeMessageTimeoutMs)
        )
        const resp = await Promise.race([messagePromiseNoSignal, timeoutPromiseNoSignal]) as { ok: boolean; error?: string; status?: number; data: T } | undefined | null
        if (resp === null) {
          throw markNoFallbackError(
            new Error("Extension messaging timeout"),
            { timeout: true }
          )
        }
        if (!resp) {
          throw new Error(`Background request failed (${method} ${path})`)
        }
        if (!resp.ok) {
          const failure = await handleFailedResponse(resp, "background")
          if (!returnResponse) {
            throw markNoFallbackError(failure.error)
          }
          return await resolveArrayBufferResponse(failure.response)
        }
        return await resolveArrayBufferResponse(resp as RuntimeResponsePayload)
      }

      if (abortSignal.aborted) {
        throw markNoFallbackError(createAbortError())
      }

      const messagePromise = browser.runtime.sendMessage(payload) as Promise<
        { ok: boolean; error?: string; status?: number; data: T } | undefined
      >

      // Add timeout to extension messaging with abort signal support
      const resp = await new Promise<
        { ok: boolean; error?: string; status?: number; data: T } | undefined | null
      >((resolve, reject) => {
        let timeoutId: ReturnType<typeof setTimeout>
        const cleanup = () => {
          clearTimeout(timeoutId)
          abortSignal.removeEventListener('abort', onAbort)
        }
        const onAbort = () => {
          cleanup()
          cancelRuntimeWorkerRequest(requestId)
          reject(markNoFallbackError(createAbortError()))
        }
        timeoutId = setTimeout(() => {
          abortSignal.removeEventListener('abort', onAbort)
          cancelRuntimeWorkerRequest(requestId)
          resolve(null)
        }, runtimeMessageTimeoutMs)
        abortSignal.addEventListener('abort', onAbort, { once: true })
        if (abortSignal.aborted) onAbort()
        messagePromise
          .then((r) => {
            cleanup()
            resolve(r)
          })
          .catch((e) => {
            cleanup()
            reject(e)
          })
      })

      if (resp === null) {
        throw markNoFallbackError(
          new Error("Extension messaging timeout"),
          { timeout: true }
        )
      }
      if (!resp) {
        throw new Error(`Background request failed (${method} ${path})`)
      }
      if (!resp.ok) {
        const failure = await handleFailedResponse(resp, "background")
        if (!returnResponse) {
          throw markNoFallbackError(failure.error)
        }
        return await resolveArrayBufferResponse(failure.response)
      }
      return await resolveArrayBufferResponse(resp as RuntimeResponsePayload)
    }
  } catch (e) {
    if (isNoFallbackError(e)) {
      if (
        isExtensionTimeoutError(e) &&
        (methodIsSafeFallback || allowIdempotentWriteFallback)
      ) {
        // Safe methods and explicitly idempotent write endpoints can fall
        // through when extension messaging itself times out.
      } else {
        throw e
      }
    } else if (!methodIsSafeFallback) {
      const canReplayIdempotentWrite =
        allowIdempotentWriteFallback && isExtensionTransportFailure(e)
      if (!isProvenNoReceiverError(e) && !canReplayIdempotentWrite) {
        throw e
      }
    }
  }

  // Fallback: direct fetch (web/dev context)
  const storage = createSafeStorage({ area: "local" })
  if (servicePromptConfig) {
    await resolveCurrentServicePromptConfig(storage, servicePromptConfig)
  }
  const resp = await tldwRequest(
    {
      path,
      method,
      headers: resolvedHeaders,
      body,
      noAuth: resolvedNoAuth,
      timeoutMs,
      abortSignal,
      responseType
    },
    createDirectRuntime(storage, servicePromptConfig, configSnapshot)
  )
  if (!resp?.ok) {
    const failure = await handleFailedResponse(resp, "direct")
    if (!returnResponse) {
      throw failure.error
    }
    return failure.response as T
  }
  return (returnResponse ? resp : resp.data) as T
}

export interface BgStreamInit<
  P extends AllowedPath = AllowedPath,
  M extends AllowedMethodFor<P> = AllowedMethodFor<P>
> {
  path: P
  method?: UpperLower<M>
  headers?: Record<string, string>
  body?: any
  streamIdleTimeoutMs?: number
  abortSignal?: AbortSignal
  onOpen?: () => void
  sanitizeRagProviderStreamError?: boolean
  servicePromptConfig?: ServicePromptTargetConfig
}

const deriveStreamIdleTimeout = (cfg: any, path: string, override?: number) => {
  if (override && override > 0) return override
  const p = String(path || "")
  const defaultIdle = 45000
  if (p.includes("/api/v1/chat/completions")) {
    return Number(cfg?.chatStreamIdleTimeoutMs) > 0
      ? Number(cfg.chatStreamIdleTimeoutMs)
      : Number(cfg?.streamIdleTimeoutMs) > 0
        ? Number(cfg.streamIdleTimeoutMs)
        : defaultIdle
  }
  return Number(cfg?.streamIdleTimeoutMs) > 0
    ? Number(cfg.streamIdleTimeoutMs)
    : defaultIdle
}

type StreamErrorInfo = { message: string; details?: unknown }

const parseStreamError = async (resp: Response): Promise<StreamErrorInfo> => {
  const ct = resp.headers.get("content-type") || ""
  if (ct.includes("application/json")) {
    const json = await resp.json().catch(() => null)
    if (json && (json.detail || json.error || json.message)) {
      const candidate = json.detail ?? json.error ?? json.message
      return {
        message: formatErrorMessage(candidate, resp.statusText || `HTTP ${resp.status}`),
        details: json,
      }
    }
  }
  const text = await resp.text().catch(() => null)
  if (text) return { message: text }
  return { message: resp.statusText }
}

type SanitizedRagStreamError = Error & {
  status?: number
  code?: string
  details?: unknown
}

const buildSanitizedRagStreamError = (
  error: unknown
): SanitizedRagStreamError => {
  const sanitized = sanitizeRagProviderFailure(error)
  const streamError = new Error(sanitized.message) as SanitizedRagStreamError
  if (typeof sanitized.status === "number") {
    streamError.status = sanitized.status
  }
  if (sanitized.code) {
    streamError.code = sanitized.code
  }
  if (sanitized.details) {
    streamError.details = sanitized.details
  }
  return streamError
}

const createSanitizedRagStreamAbortError = (): RequestAbortError =>
  createAbortError(RAG_STREAM_ABORT_MESSAGE)

const isRequestAbort = (error: unknown, signal?: AbortSignal): boolean =>
  Boolean(signal?.aborted) ||
  (error as { name?: unknown } | null)?.name === "AbortError" ||
  (error as { code?: unknown } | null)?.code === "REQUEST_ABORTED"

const yieldToBrowser = async (): Promise<void> => {
  if (typeof requestAnimationFrame === "function") {
    await new Promise<void>((resolve) => {
      requestAnimationFrame(() => resolve())
    })
    return
  }
  await new Promise<void>((resolve) => setTimeout(resolve, 0))
}

/**
 * Direct streaming fallback used before handoff, or for safe/idempotent replay.
 */
async function* bgStreamDirectUnsafe<
  P extends AllowedPath = AllowedPath,
  M extends AllowedMethodFor<P> = AllowedMethodFor<P>
>(
  { path, method = 'POST' as UpperLower<M>, headers = {}, body, streamIdleTimeoutMs, abortSignal, onOpen, servicePromptConfig }: BgStreamInit<P, M>
): AsyncGenerator<string> {
  if (servicePromptConfig && !isServicePromptRequestPath(path, method)) {
    throw new Error(
      "A Service Prompt config can only be used with Service Prompt requests."
    )
  }
  const storage = createSafeStorage({ area: "local" })
  const cfg = servicePromptConfig
    ? await resolveCurrentServicePromptConfig(storage, servicePromptConfig)
    : (await resolveDirectConfig(storage)) || null
  const normalizedPath = normalizeKnownPathQuirks(path)
  const isAbsolute = typeof normalizedPath === "string" && /^https?:/i.test(normalizedPath)
  const absolutePath = isAbsolute ? String(normalizedPath) : ""
  const transport =
    !isAbsolute && typeof normalizedPath === "string"
      ? resolveBrowserRequestTransport({
          config: cfg,
          path: String(normalizedPath)
        })
      : null
  const hostedMode = transport?.mode === "hosted"
  const advancedTransportGuard = resolveAdvancedRequestTransportGuard({
    transport,
    hasConfiguredServerUrl: Boolean(cfg?.serverUrl),
    isAbsolute
  })
  if (
    isAbsolute &&
    !isAbsoluteUrlAllowlisted(
      absolutePath,
      cfg as unknown as Record<string, unknown>
    )
  ) {
    throw new Error(ABSOLUTE_URL_BLOCK_ERROR)
  }
  if (advancedTransportGuard.isUnconfigured) {
    throw new Error("tldw server not configured")
  }
  const baseUrl =
    advancedTransportGuard.origin ||
    (cfg?.serverUrl ? String(cfg.serverUrl).replace(/\/$/, "") : "")
  const url = isAbsolute
    ? absolutePath
    : transport?.url ||
      `${baseUrl}${String(normalizedPath).startsWith("/") ? "" : "/"}${String(normalizedPath)}`
  const sameOriginAbsolute = isAbsolute
    ? isSameOriginAbsoluteUrlForConfiguredServer(
        absolutePath,
        cfg as unknown as Record<string, unknown>
      )
    : false
  const shouldSkipAuth = isAbsolute && !sameOriginAbsolute
  const pageOrigin =
    typeof window === "undefined" ? null : String(window.location?.origin || "")
  const samePageOriginAbsolute =
    isAbsolute && pageOrigin
      ? isSameOriginAbsoluteUrlForConfiguredServer(absolutePath, {
          serverUrl: pageOrigin
        })
      : false
  const absoluteCookieTransport =
    isAbsolute && sameOriginAbsolute && samePageOriginAbsolute
      ? resolveBrowserRequestTransport({
          config: cfg,
          path: absolutePath,
          pageOrigin
        })
      : null
  const cookieSession = isCookieSessionBrowserTransport({
    authMode: cfg?.authMode,
    authSource: cfg?.authSource,
    transportMode: absoluteCookieTransport?.mode || transport?.mode,
    transportKind: absoluteCookieTransport?.kind || transport?.kind,
    pageOrigin
  })
  const resolvedHeaders: Record<string, string> = { ...(headers || {}) }
  for (const k of Object.keys(resolvedHeaders)) {
    const kl = k.toLowerCase()
    if (
      kl === "x-api-key" ||
      kl === "authorization" ||
      (cookieSession && kl === "x-csrf-token")
    ) {
      delete resolvedHeaders[k]
    }
  }

  if (cookieSession) {
    if (!shouldSkipAuth && isUnsafeMethod(String(method))) {
      const csrfToken = readBrowserCookie("csrf_token")
      if (csrfToken) resolvedHeaders["X-CSRF-Token"] = csrfToken
    }
  } else if (!shouldSkipAuth && !hostedMode && cfg?.authMode === "single-user") {
    const runtimeApiKey = servicePromptConfig
      ? ""
      : String(getRuntimeSingleUserApiKeyOverride() || "").trim()
    const key = runtimeApiKey || String(cfg?.apiKey || "").trim()
    if (!key) {
      throw new Error(
        "Add or update your API key in Settings -> tldw server, then try again."
      )
    }
    resolvedHeaders["X-API-KEY"] = key
  } else if (!shouldSkipAuth && !hostedMode && cfg?.authMode === "multi-user") {
    const token = String(cfg?.accessToken || "").trim()
    if (token) {
      resolvedHeaders["Authorization"] = `Bearer ${token}`
    } else {
      throw new Error("Not authenticated. Please login under Settings > tldw.")
    }
  }
  if (cfg?.orgId) {
    resolvedHeaders["X-TLDW-Org-Id"] = String(cfg.orgId)
  }

  resolvedHeaders["Accept"] = resolvedHeaders["Accept"] || "text/event-stream"
  resolvedHeaders["Cache-Control"] =
    resolvedHeaders["Cache-Control"] || "no-cache"
  resolvedHeaders["Connection"] =
    resolvedHeaders["Connection"] || "keep-alive"

  const controller = new AbortController()
  const idleMs = deriveStreamIdleTimeout(
    cfg,
    normalizedPath as string,
    Number(streamIdleTimeoutMs)
  )
  let idleTimer: ReturnType<typeof setTimeout> | null = null
  let idleError: Error | null = null
  const resetIdle = () => {
    if (idleTimer) clearTimeout(idleTimer)
    idleTimer = setTimeout(() => {
      idleError = new Error("Stream timeout: no updates received")
      try {
        controller.abort()
      } catch {}
    }, idleMs)
  }

  const onAbort = () => {
    try {
      controller.abort()
    } catch {}
  }
  if (abortSignal) {
    if (abortSignal.aborted) onAbort()
    else abortSignal.addEventListener("abort", onAbort, { once: true })
  }

  const fetchStream = async (): Promise<Response> => {
    return await fetch(url, {
      method,
      headers: resolvedHeaders,
      body:
        body != null
          ? typeof body === "string"
            ? body
            : JSON.stringify(body)
          : undefined,
      signal: controller.signal
    })
  }

  let resp = await fetchStream()
  if (
    !shouldSkipAuth &&
    !hostedMode &&
    resp.status === 401 &&
    cfg?.authMode === "multi-user" &&
    cfg?.refreshToken
  ) {
    try {
      if (servicePromptConfig) {
        await refreshAuthDirect(storage, servicePromptConfig, cfg)
        const latestCfg = await resolveCurrentServicePromptConfig(
          storage,
          servicePromptConfig
        )
        if (latestCfg.accessToken) {
          resolvedHeaders["Authorization"] = `Bearer ${latestCfg.accessToken}`
          resp = await fetchStream()
        }
      } else {
        const refreshResp = await fetch(`${baseUrl}/api/v1/auth/refresh`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ refresh_token: cfg.refreshToken }),
          signal: controller.signal
        })
        if (refreshResp.ok) {
          const tokens = await refreshResp.json().catch(() => null)
          if (tokens?.access_token && !controller.signal.aborted) {
            const latestCfg = await commitDirectRefresh(
              storage,
              cfg,
              String(cfg.accessToken || "").trim(),
              String(cfg.refreshToken || "").trim(),
              {
                accessToken: tokens.access_token,
                refreshToken: tokens.refresh_token || cfg.refreshToken
              }
            )
            resolvedHeaders["Authorization"] =
              `Bearer ${latestCfg.accessToken}`
            resp = await fetchStream()
          }
        }
      }
    } catch (error) {
      if (isRequestConfigScopeChangedError(error)) {
        throw error
      }
      // ignore refresh failures and continue with original response
    }
  }

  if (!resp.ok) {
    const errorInfo = await parseStreamError(resp)
    const error = new Error(
      formatErrorMessage(errorInfo.message, `HTTP ${resp.status}`)
    ) as Error & { status?: number; details?: unknown; retryAfter?: number }
    error.status = resp.status
    if (errorInfo.details) error.details = errorInfo.details
    const retryAfterMs = parseRetryAfter(resp.headers.get("retry-after"))
    if (typeof retryAfterMs === "number" && retryAfterMs > 0) {
      error.retryAfter = retryAfterMs / 1_000
    }
    throw error
  }
  if (!resp.body) {
    throw new Error("No response body")
  }

  onOpen?.()
  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  resetIdle()
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      resetIdle()
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split("\n")
      buffer = lines.pop() || ""
      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed) continue
        resetIdle()
        if (trimmed.startsWith("data:")) {
          const data = trimmed.slice(5).trim()
          if (data === "[DONE]") {
            if (idleTimer) clearTimeout(idleTimer)
            return
          }
          yield data
        } else if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
          yield trimmed
        }
      }
    }
    const tail = buffer.trim()
    if (tail) {
      if (tail.startsWith("data:")) {
        const data = tail.slice(5).trim()
        if (data !== "[DONE]") {
          yield data
        }
      } else if (tail.startsWith("{") || tail.startsWith("[")) {
        yield tail
      }
    }
  } catch (e: any) {
    if (idleError) {
      throw idleError
    }
    if (abortSignal?.aborted) {
      throw createAbortError(readErrorMessage(e))
    }
    throw e
  } finally {
    if (idleTimer) clearTimeout(idleTimer)
    try {
      reader.cancel()
    } catch {}
    if (abortSignal) {
      try {
        abortSignal.removeEventListener("abort", onAbort)
      } catch {}
    }
  }
}

async function* bgStreamDirect<
  P extends AllowedPath = AllowedPath,
  M extends AllowedMethodFor<P> = AllowedMethodFor<P>
>(init: BgStreamInit<P, M>): AsyncGenerator<string> {
  try {
    yield* bgStreamDirectUnsafe(init)
  } catch (error) {
    if (!init.sanitizeRagProviderStreamError) {
      throw error
    }
    if (isRequestAbort(error, init.abortSignal)) {
      throw createSanitizedRagStreamAbortError()
    }
    throw buildSanitizedRagStreamError(error)
  }
}

export async function* bgStream<
  P extends AllowedPath = AllowedPath,
  M extends AllowedMethodFor<P> = AllowedMethodFor<P>
>(
  {
    path,
    method = 'POST' as UpperLower<M>,
    headers = {},
    body,
    streamIdleTimeoutMs,
    abortSignal,
    onOpen,
    sanitizeRagProviderStreamError = false,
    servicePromptConfig
  }: BgStreamInit<P, M>
): AsyncGenerator<string> {
  const hasHttpStatus = (value: unknown): boolean =>
    extractHttpStatus(value) !== null
  let openNotified = false
  const notifyOpen = () => {
    if (openNotified) return
    openNotified = true
    onOpen?.()
  }

  const canUseRuntimePortTransport = async (): Promise<boolean> => {
    const hasRuntimePort = Boolean(browser?.runtime?.connect && browser?.runtime?.id)
    if (!hasRuntimePort) return false

    const now = Date.now()
    if (
      streamRuntimePortUsable !== null &&
      now - lastStreamRuntimeHealthCheckAt < STREAM_RUNTIME_HEALTH_TTL_MS
    ) {
      return streamRuntimePortUsable
    }

    if (!browser?.runtime?.sendMessage) {
      streamRuntimePortUsable = true
      lastStreamRuntimeHealthCheckAt = now
      return true
    }

    try {
      const pingResult = await Promise.race([
        browser.runtime.sendMessage({ type: "tldw:ping" }),
        new Promise<null>((resolve) =>
          setTimeout(() => resolve(null), STREAM_RUNTIME_PING_TIMEOUT_MS)
        )
      ])
      streamRuntimePortUsable =
        Boolean(pingResult) && Boolean((pingResult as any)?.ok)
    } catch {
      streamRuntimePortUsable = false
    }
    lastStreamRuntimeHealthCheckAt = now
    return Boolean(streamRuntimePortUsable)
  }

  const hasRuntimePort = await canUseRuntimePortTransport()
  if (!hasRuntimePort) {
    yield* bgStreamDirect({
      path,
      method,
      headers,
      body,
      streamIdleTimeoutMs,
      abortSignal,
      onOpen: notifyOpen,
      sanitizeRagProviderStreamError,
      servicePromptConfig
    })
    return
  }
  const mayReplayAfterHandoff =
    isSafeFallbackMethod(method) ||
    isIdempotentWriteFallbackAllowed(method, path, body)

  // Derive the response-acquisition timeout from config instead of a hard-coded
  // 5s. Time-to-response over 5s is normal for large prompts, RAG,
  // or a cold local model, and a premature disconnect used to replay the whole
  // request. Reuse the stream idle-timeout budget (chat default 45s).
  const streamStorage = createSafeStorage({ area: "local" })
  const streamCfg =
    (await streamStorage
      .get<Record<string, unknown>>("tldwConfig")
      .catch(() => null)) || null
  const connectionTimeoutMs = deriveStreamIdleTimeout(
    streamCfg,
    String(path),
    Number(streamIdleTimeoutMs)
  )
  // Extension streaming permits direct fallback before postMessage handoff.
  let port: ReturnType<typeof browser.runtime.connect>
  try {
    port = browser.runtime.connect({ name: 'tldw:stream' })
  } catch (connectError) {
    if (!abortSignal?.aborted) {
      yield* bgStreamDirect({
        path,
        method,
        headers,
        body,
        streamIdleTimeoutMs,
        abortSignal,
        onOpen: notifyOpen,
        sanitizeRagProviderStreamError,
        servicePromptConfig
      })
      return
    }
    if (sanitizeRagProviderStreamError) {
      throw createSanitizedRagStreamAbortError()
    }
    throw connectError
  }
  const queue: string[] = []
  let done = false
  let error: any = null
  let firstDataReceived = false
  let streamOpened = false
  let connectionTimedOut = false
  let handoffAttempted = false

  // Connection timeout - if no response body is acquired within the derived window,
  // give up on the port. After handoff, replay requires explicit idempotency.
  const connectionTimer = setTimeout(() => {
    if (!streamOpened && !done) {
      connectionTimedOut = true
      done = true
      try { port.disconnect() } catch {}
    }
  }, connectionTimeoutMs)

  const onMessage = (msg: any) => {
    if (msg?.event === 'open') {
      streamOpened = true
      clearTimeout(connectionTimer)
      notifyOpen()
    } else if (msg?.event === 'data') {
      if (!streamOpened) {
        streamOpened = true
        clearTimeout(connectionTimer)
        notifyOpen()
      }
      if (!firstDataReceived) {
        firstDataReceived = true
        clearTimeout(connectionTimer)
      }
      queue.push(msg.data as string)
    } else if (msg?.event === 'done') {
      done = true
    } else if (msg?.event === 'error') {
      const streamError = sanitizeRagProviderStreamError
        ? buildSanitizedRagStreamError({
            status: msg.status,
            error: msg.message,
            data:
              msg.details ??
              (msg.code
                ? {
                    detail: {
                      error_code: msg.code,
                      message: msg.message
                    }
                  }
                : undefined)
          })
        : (new Error(msg.message || 'Stream error') as Error & {
            status?: number
            details?: unknown
            retryAfter?: number
          })
      if (!sanitizeRagProviderStreamError) {
        if (typeof msg.status === "number" && Number.isFinite(msg.status)) {
          streamError.status = Math.trunc(msg.status)
        }
        if (typeof msg.details !== "undefined" && msg.details !== null) {
          streamError.details = sanitizeResponseData(msg.details)
        }
        const retryAfterMs = parseRetryAfter(
          typeof msg.retryAfter === "string" ? msg.retryAfter : null
        )
        if (typeof retryAfterMs === "number" && retryAfterMs > 0) {
          const retryableError = streamError as Error & {
            retryAfter?: number
          }
          retryableError.retryAfter = retryAfterMs / 1_000
        }
      }
      error = streamError
      done = true
    }
  }
  port.onMessage.addListener(onMessage)
  const onDisconnect = () => {
    clearTimeout(connectionTimer)
    if (!done) {
      if (!error) error = new Error('Stream disconnected')
      done = true
    }
  }
  port.onDisconnect.addListener(onDisconnect)
  const onAbort = () => {
    clearTimeout(connectionTimer)
    if (!error) error = new Error('Aborted')
    done = true
    try { port.disconnect() } catch {}
  }
  if (abortSignal) {
    if (abortSignal.aborted) onAbort()
    else abortSignal.addEventListener('abort', onAbort, { once: true })
  }
  if (!done) {
    try {
      // postMessage may throw after delivery. Once attempted, dispatch is
      // unknown and therefore conservatively treated as having occurred.
      handoffAttempted = true
      const portPayload: Record<string, unknown> = {
        path,
        method,
        headers,
        body,
        streamIdleTimeoutMs
      }
      if (sanitizeRagProviderStreamError) {
        portPayload.sanitizeRagProviderStreamError = true
      }
      if (servicePromptConfig) {
        portPayload.servicePromptConfig = servicePromptConfig
      }
      port.postMessage(portPayload)
    } catch (e) {
      clearTimeout(connectionTimer)
      if (!error) error = e
      done = true
    }
  }

  try {
    let drainedSinceYield = 0
    let sliceStartedAt = Date.now()
    while (!done || queue.length > 0) {
      if (queue.length > 0) {
        yield queue.shift() as string
        drainedSinceYield += 1
        const sliceElapsedMs = Date.now() - sliceStartedAt
        if (
          drainedSinceYield >= STREAM_QUEUE_DRAIN_BATCH_LIMIT ||
          sliceElapsedMs >= STREAM_QUEUE_DRAIN_SLICE_MS
        ) {
          await yieldToBrowser()
          drainedSinceYield = 0
          sliceStartedAt = Date.now()
        }
      } else {
        await new Promise((r) => setTimeout(r, 10))
        drainedSinceYield = 0
        sliceStartedAt = Date.now()
      }
    }
    if (sanitizeRagProviderStreamError && abortSignal?.aborted) {
      throw createSanitizedRagStreamAbortError()
    }
    // Resolve a response-acquisition timeout without replaying ambiguous dispatch.
    if (connectionTimedOut) {
      if (!handoffAttempted || mayReplayAfterHandoff) {
        yield* bgStreamDirect({
          path,
          method,
          headers,
          body,
          streamIdleTimeoutMs,
          abortSignal,
          onOpen: notifyOpen,
          sanitizeRagProviderStreamError,
          servicePromptConfig
        })
        return
      }
      const timeoutMessage = `Stream connection timed out after ${connectionTimeoutMs}ms before response acquisition`
      throw createStreamInterruptedError(
        sanitizeRagProviderStreamError
          ? sanitizeRagProviderFailure(new Error(timeoutMessage)).message
          : timeoutMessage
      )
    }
    const shouldFallbackAfterEarlyError =
      !firstDataReceived &&
      !abortSignal?.aborted &&
      Boolean(error) &&
      (isExtensionTransportFailure(error) || !hasHttpStatus(error))
    if (shouldFallbackAfterEarlyError) {
      if (!handoffAttempted || mayReplayAfterHandoff) {
        yield* bgStreamDirect({
          path,
          method,
          headers,
          body,
          streamIdleTimeoutMs,
          abortSignal,
          onOpen: notifyOpen,
          sanitizeRagProviderStreamError,
          servicePromptConfig
        })
        return
      }
      const interruptionMessage =
        error instanceof Error
          ? error.message
          : String(error || "Stream transport interrupted")
      throw createStreamInterruptedError(
        sanitizeRagProviderStreamError
          ? sanitizeRagProviderFailure(error).message
          : interruptionMessage
      )
    }
    const shouldGracefullyEndAfterPartialStreamError =
      firstDataReceived &&
      !abortSignal?.aborted &&
      Boolean(error) &&
      (isExtensionTransportFailure(error) || !hasHttpStatus(error))
    if (shouldGracefullyEndAfterPartialStreamError) {
      // We already delivered data to the caller; avoid replaying non-idempotent
      // streamed requests after transport loss and let caller finalize partial output.
      const rawInterruptionDetail =
        error instanceof Error
          ? error.message
          : String(error || "Stream transport interrupted")
      const sanitizedInterruption = sanitizeRagProviderStreamError
        ? sanitizeRagProviderFailure(error)
        : null
      const interruption: Record<string, unknown> = {
        event: "stream_transport_interrupted",
        detail: sanitizedInterruption?.message ?? rawInterruptionDetail,
        partial_response_saved: true
      }
      if (sanitizedInterruption?.status) {
        interruption.status = sanitizedInterruption.status
      }
      if (sanitizedInterruption?.code) {
        interruption.code = sanitizedInterruption.code
      }
      if (sanitizedInterruption?.details) {
        interruption.details = sanitizedInterruption.details
      }
      yield JSON.stringify(interruption)
      return
    }
    if (error) throw error
  } finally {
    clearTimeout(connectionTimer)
    try { port.onMessage.removeListener(onMessage); } catch {}
    try { port.onDisconnect.removeListener(onDisconnect); } catch {}
    try { port.disconnect(); } catch {}
    if (abortSignal) {
      try { abortSignal.removeEventListener('abort', onAbort) } catch {}
    }
  }
}

export type BgUploadFile = {
  fieldName?: string
  name?: string
  type?: string
  data: ArrayBuffer | Uint8Array | number[]
}

export interface BgUploadInit<P extends AllowedPath = AllowedPath, M extends AllowedMethodFor<P> = AllowedMethodFor<P>> {
  path: P
  method?: UpperLower<M>
  headers?: Record<string, string>
  // key/value fields to include alongside file in FormData
  fields?: Record<string, any>
  // File payload as raw bytes with metadata (structured-cloneable)
  file?: Omit<BgUploadFile, "fieldName">
  files?: BgUploadFile[]
  // Optional override for the multipart file field name
  fileFieldName?: string
  // Optional timeout override for upload requests
  timeoutMs?: number
  abortSignal?: AbortSignal
  responseType?: "json" | "text" | "arrayBuffer"
  preferDirect?: boolean
  servicePromptConfig?: ServicePromptTargetConfig
}

export async function bgUpload<T = any, P extends AllowedPath = AllowedPath, M extends AllowedMethodFor<P> = AllowedMethodFor<P>>(
  {
    path,
    method = 'POST' as UpperLower<M>,
    headers = {},
    fields = {},
    file,
    files,
    fileFieldName,
    timeoutMs,
    abortSignal,
    responseType,
    preferDirect = false,
    servicePromptConfig
  }: BgUploadInit<P, M>
): Promise<T> {
  if (servicePromptConfig && !isServicePromptRequestPath(path, method)) {
    throw new Error(
      "A Service Prompt config can only be used with Service Prompt requests."
    )
  }
  const hasRuntimeMessage =
    !preferDirect &&
    Boolean(browser?.runtime?.sendMessage && browser?.runtime?.id)
  const methodIsSafeFallback = isSafeFallbackMethod(method)
  if (hasRuntimeMessage) {
    try {
      // Add timeout to extension messaging for uploads. The worker only acks
      // after the upload (and any synchronous processing kickoff) completes, so
      // a short cap would abort large-but-progressing uploads the worker is
      // still finishing.
      const resolvedTimeout =
        typeof timeoutMs === "number" && timeoutMs > 0
          ? timeoutMs
          : DEFAULT_UPLOAD_RUNTIME_MESSAGE_TIMEOUT_MS
      const uploadTimeout = Math.max(5000, resolvedTimeout)
      if (abortSignal?.aborted) {
        throw markNoFallbackError(createAbortError())
      }
      const requestId =
        servicePromptConfig && abortSignal
          ? createRuntimeRequestId()
          : undefined
      const uploadPromise = browser.runtime.sendMessage({
        type: 'tldw:upload',
        payload: {
          path,
          method,
          headers,
          fields,
          file,
          files,
          fileFieldName,
          timeoutMs: resolvedTimeout,
          responseType,
          servicePromptConfig,
          ...(requestId ? { requestId } : {})
        }
      })
      const resp = await new Promise<{
        ok: boolean
        error?: string
        status?: number
        data: T
      } | undefined | null>((resolve, reject) => {
        let timeoutId: ReturnType<typeof setTimeout>
        const onAbort = () => {
          clearTimeout(timeoutId)
          abortSignal?.removeEventListener('abort', onAbort)
          cancelRuntimeWorkerRequest(requestId)
          reject(markNoFallbackError(createAbortError()))
        }
        timeoutId = setTimeout(() => {
          abortSignal?.removeEventListener('abort', onAbort)
          cancelRuntimeWorkerRequest(requestId)
          resolve(null)
        }, uploadTimeout)
        abortSignal?.addEventListener('abort', onAbort, { once: true })
        if (abortSignal?.aborted) onAbort()
        uploadPromise.then(
          (response) => {
            clearTimeout(timeoutId)
            abortSignal?.removeEventListener('abort', onAbort)
            resolve(response as {
              ok: boolean
              error?: string
              status?: number
              data: T
            } | undefined)
          },
          (error) => {
            clearTimeout(timeoutId)
            abortSignal?.removeEventListener('abort', onAbort)
            reject(error)
          }
        )
      })
      if (resp === null) {
        throw markNoFallbackError(
          new Error("Extension messaging timeout"),
          { timeout: true }
        )
      }
      if (!resp?.ok) {
        const msg = formatErrorMessage(
          resp?.error,
          `Upload failed: ${resp?.status}`
        )
        const error = new Error(msg) as Error & { status?: number; details?: unknown }
        error.status = resp?.status
        if (typeof resp?.data !== "undefined") {
          error.details = sanitizeResponseData(resp.data)
        }
        throw markNoFallbackError(error)
      }
      return resp.data as T
    } catch (e) {
      if (isNoFallbackError(e)) {
        if (isExtensionTimeoutError(e) && methodIsSafeFallback) {
          // Safe methods can fall through on timeout because duplicate side-effects are not expected.
        } else {
          throw e
        }
      } else if (!methodIsSafeFallback && !isProvenNoReceiverError(e)) {
        throw e
      }
    }
  }

  if (typeof FormData === "undefined") {
    throw new Error("File upload is not supported in this environment.")
  }
  const formData = new FormData()
  Object.entries(fields || {}).forEach(([key, value]) => {
    if (value == null) return
    if (Array.isArray(value)) {
      value.forEach((entry) => formData.append(key, String(entry)))
    } else {
      formData.append(key, String(value))
    }
  })
  const appendFile = (
    item: BgUploadFile,
    fieldName: string,
    { appendLegacyFileAlias = false }: { appendLegacyFileAlias?: boolean } = {}
  ) => {
    const name = item.name || "file"
    const type = item.type || "application/octet-stream"
    const toBytes = (data: ArrayBuffer | Uint8Array | number[]) => {
      if (data instanceof Uint8Array) return data
      if (data instanceof ArrayBuffer) return new Uint8Array(data)
      return Uint8Array.from(data)
    }
    const bytes = toBytes(item.data)
    if (typeof Blob === "undefined") {
      throw new Error("File upload is not supported in this environment.")
    }
    const buffer = bytes.buffer
    const slice =
      buffer instanceof ArrayBuffer
        ? buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
        : new Uint8Array(bytes).buffer
    const blob = new Blob([slice], { type })
    formData.append(fieldName, blob, name)
    if (appendLegacyFileAlias && fieldName !== "file") {
      formData.append("file", blob, name)
    }
  }
  if (Array.isArray(files) && files.length > 0) {
    files.forEach((item) => {
      appendFile(item, item.fieldName || "files")
    })
  } else if (file) {
    const legacyFieldName = fileFieldName || "files"
    appendFile(
      {
        ...file,
        fieldName: legacyFieldName
      },
      legacyFieldName,
      { appendLegacyFileAlias: !fileFieldName }
    )
  }

  const storage = createSafeStorage({ area: "local" })
  const resp = await tldwRequest(
    {
      path,
      method,
      headers,
      body: formData,
      timeoutMs,
      abortSignal,
      responseType
    },
    createDirectRuntime(storage, servicePromptConfig)
  )
  if (!resp?.ok) {
    const msg = formatErrorMessage(
      resp?.error,
      `Upload failed: ${resp?.status}`
    )
    const error = new Error(msg) as Error & { status?: number; details?: unknown }
    error.status = resp?.status
    if (typeof resp?.data !== "undefined") {
      error.details = sanitizeResponseData(resp.data)
    }
    throw error
  }
  return resp.data as T
}

export async function bgRequestValidated<
  T = any,
  P extends PathOrUrl = AllowedPath,
  M extends AllowedMethodFor<P> = AllowedMethodFor<P>
>(
  init: BgRequestInit<P, M>,
  validate?: (data: unknown) => T
): Promise<T> {
  const data = await bgRequest<any, P, M>(init)
  return validate ? validate(data) : (data as T)
}

// Strict variants: enforce that call sites use ClientPath-derived strings by default.
export async function bgRequestClient<
  T = any,
  P extends ClientPathOrUrlWithQuery = ClientPathOrUrlWithQuery,
  M extends AllowedMethodFor<P> = AllowedMethodFor<P>
>(init: BgRequestInit<P, M>): Promise<T> {
  return bgRequest<T, P, M>(init)
}

export async function* bgStreamClient<
  P extends ClientPathRuntimeWithQuery = ClientPathRuntimeWithQuery,
  M extends AllowedMethodFor<P> = AllowedMethodFor<P>
>(init: BgStreamInit<P, M>): AsyncGenerator<string> {
  for await (const chunk of bgStream<P, M>(init)) {
    yield chunk
  }
}
