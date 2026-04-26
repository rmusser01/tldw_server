import { browser } from "wxt/browser"
import { Storage } from "@plasmohq/storage"
import { createSafeStorage } from "@/utils/safe-storage"
import { formatErrorMessage } from "@/utils/format-error-message"
import {
  resolveBrowserRequestTransport,
  tldwRequest
} from "@/services/tldw/request-core"
import {
  BACKEND_UNREACHABLE_EVENT,
  type BackendUnreachableDetail
} from "@/services/request-events"
import type {
  AllowedMethodFor,
  AllowedPath,
  ClientPathOrUrlWithQuery,
  ClientPathRuntimeWithQuery,
  PathOrUrl,
  UpperLower
} from "@/services/tldw/openapi-guard"

const ERROR_LOG_THROTTLE_MS = 15_000
const RATE_LIMIT_LOG_THROTTLE_MS = 60_000
const ERROR_LOG_MAX_ENTRIES = 200
const BACKEND_UNREACHABLE_EVENT_THROTTLE_MS = 5_000
const STREAM_RUNTIME_PING_TIMEOUT_MS = 400
const STREAM_RUNTIME_HEALTH_TTL_MS = 30_000
const STREAM_QUEUE_DRAIN_BATCH_LIMIT = 32
const STREAM_QUEUE_DRAIN_SLICE_MS = 12
const ABSOLUTE_URL_BLOCK_ERROR =
  "Direct stream fallback is allowed only for allowlisted absolute URLs."
const BACKEND_UNREACHABLE_PATTERN =
  /(networkerror|failed to fetch|network error|load failed|err_connection|could not establish connection|receiving end does not exist)/i
const errorLogHistory = new Map<string, number>()
let lastBackendUnreachableEventAt = 0
let lastStreamRuntimeHealthCheckAt = 0
let streamRuntimePortUsable: boolean | null = null

const normalizeKnownPathQuirks = <P extends PathOrUrl>(rawPath: P): P => {
  if (typeof rawPath !== "string") return rawPath
  return rawPath
    .replace("/api/v1/media/?", "/api/v1/media?")
    .replace("/api/v1/files/?", "/api/v1/files?") as P
}

const parseHttpOrigin = (value: unknown): string | null => {
  const raw = String(value || "").trim()
  if (!raw) return null
  try {
    const parsed = new URL(raw)
    if (!/^https?:$/i.test(parsed.protocol)) return null
    return parsed.origin.toLowerCase()
  } catch {
    return null
  }
}

const toAllowlistEntries = (value: unknown): string[] => {
  if (Array.isArray(value)) {
    return value
      .map((entry) => String(entry || "").trim())
      .filter((entry) => entry.length > 0)
  }
  if (typeof value === "string") {
    const trimmed = value.trim()
    if (!trimmed) return []
    if (!trimmed.includes(",")) return [trimmed]
    return trimmed
      .split(",")
      .map((entry) => entry.trim())
      .filter((entry) => entry.length > 0)
  }
  return []
}

const configuredServerOrigin = (cfg: Record<string, unknown> | null): string | null => {
  return parseHttpOrigin(cfg?.serverUrl)
}

const absoluteOriginAllowlistFromConfig = (
  cfg: Record<string, unknown> | null
): Set<string> => {
  const out = new Set<string>()
  const serverOrigin = configuredServerOrigin(cfg)
  if (serverOrigin) out.add(serverOrigin)
  for (const entry of toAllowlistEntries(cfg?.absoluteUrlAllowlist)) {
    const parsedOrigin = parseHttpOrigin(entry)
    if (parsedOrigin) out.add(parsedOrigin)
  }
  return out
}

const isAbsoluteUrlAllowlisted = (
  absoluteUrl: string,
  cfg: Record<string, unknown> | null
): boolean => {
  try {
    const target = new URL(absoluteUrl)
    if (!/^https?:$/i.test(target.protocol)) return false
    const allowlistedOrigins = absoluteOriginAllowlistFromConfig(cfg)
    return allowlistedOrigins.has(target.origin.toLowerCase())
  } catch {
    return false
  }
}

const isSameOriginAbsoluteUrlForConfiguredServer = (
  absoluteUrl: string,
  cfg: Record<string, unknown> | null
): boolean => {
  const serverOrigin = configuredServerOrigin(cfg)
  if (!serverOrigin) return false
  try {
    const target = new URL(absoluteUrl)
    if (!/^https?:$/i.test(target.protocol)) return false
    return target.origin.toLowerCase() === serverOrigin
  } catch {
    return false
  }
}

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
const sanitizeResponseData = (
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

const shouldNotifyBackendUnavailable = (entry: {
  method: string
  path: string
  status?: number
  error?: string
}): boolean => {
  const path = String(entry.path || "")
  // Restrict notifications to API requests only.
  if (!path.includes("/api/")) return false
  if (entry.status === 0) return true
  return BACKEND_UNREACHABLE_PATTERN.test(String(entry.error || ""))
}

const notifyBackendUnavailable = (entry: {
  method: string
  path: string
  status?: number
  error?: string
  source: "background" | "direct"
}) => {
  if (typeof window === "undefined" || typeof window.dispatchEvent !== "function") {
    return
  }
  if (!shouldNotifyBackendUnavailable(entry)) return
  const now = Date.now()
  if (now - lastBackendUnreachableEventAt < BACKEND_UNREACHABLE_EVENT_THROTTLE_MS) {
    return
  }
  lastBackendUnreachableEventAt = now

  const detail: BackendUnreachableDetail = {
    method: entry.method,
    path: entry.path,
    status: entry.status,
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
}

export async function bgRequest<
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
    preferDirect = false
  } = init
  const path = normalizeKnownPathQuirks(rawPath)
  const isAbsoluteUrl = typeof path === "string" && /^https?:/i.test(path)
  const noAuthExplicit = Object.prototype.hasOwnProperty.call(init, "noAuth")
  let resolvedNoAuth = noAuthExplicit ? noAuth : (noAuth || isAbsoluteUrl)
  if (!noAuthExplicit && isAbsoluteUrl) {
    const storage = createSafeStorage()
    const cfg = (await storage.get<Record<string, unknown>>("tldwConfig").catch(() => null)) || null
    const sameOriginAbsolute = isSameOriginAbsoluteUrlForConfiguredServer(
      String(path),
      cfg
    )
    resolvedNoAuth = noAuth || !sameOriginAbsolute
  }
  const resolvedHeaders = headers
  const recordRequestError = async (entry: {
    method: string
    path: string
    status?: number
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
    details?: unknown
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
    if (typeof details !== "undefined") {
      error.details = sanitizeResponseData(details)
    }
    return error
  }
  const shouldBypassBackground =
    responseType === "arrayBuffer" &&
    typeof path === "string" &&
    path.includes("/api/v1/audio/")
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
  const hasRuntimeMessage =
    !preferDirect &&
    Boolean(browser?.runtime?.sendMessage && browser?.runtime?.id)
  const methodIsSafeFallback = isSafeFallbackMethod(method)

  // Some binary responses do not survive extension message serialization.
  if (shouldBypassBackground) {
    const storage = createSafeStorage()
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
      { getConfig: () => storage.get("tldwConfig").catch(() => null) }
    )
    if (!resp?.ok) {
      const msg = formatErrorMessage(
        resp?.error,
        `Request failed: ${resp?.status}`
      )
      if (!isAbortErrorMessage(msg)) {
        console.warn("[tldw:request]", method, path, resp?.status, msg)
        await recordRequestError({
          method: String(method),
          path: String(path),
          status: resp?.status,
          error: msg,
          source: "direct"
        })
        notifyBackendUnavailable({
          method: String(method),
          path: String(path),
          status: resp?.status,
          error: msg,
          source: "direct"
        })
      }
      const error = buildRequestError(msg, resp?.status, resp?.data)
      if (!returnResponse) {
        throw error
      }
    }
    return (returnResponse ? resp : resp.data) as T
  }

  // If extension messaging is available, use it (extension context)
  try {
    if (hasRuntimeMessage) {
      const payload = {
        type: 'tldw:request',
        payload: {
          path,
          method,
          headers: resolvedHeaders,
          body,
          noAuth: resolvedNoAuth,
          timeoutMs,
          responseType
        }
      }

      if (!abortSignal) {
        // Add timeout to extension messaging - if service worker doesn't respond, fall back to direct request
        const extensionTimeout = 3000 // 3 second timeout for extension messaging
        const messagePromiseNoSignal = browser.runtime.sendMessage(payload)
        const timeoutPromiseNoSignal = new Promise<null>((resolve) =>
          setTimeout(() => resolve(null), extensionTimeout)
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
          const msg = formatErrorMessage(
            resp?.error,
            `Request failed: ${resp?.status}`
          )
          if (!isAbortErrorMessage(msg)) {
            console.warn("[tldw:request]", method, path, resp?.status, msg)
            await recordRequestError({
              method: String(method),
              path: String(path),
              status: resp?.status,
              error: msg,
              source: "background"
            })
            notifyBackendUnavailable({
              method: String(method),
              path: String(path),
              status: resp?.status,
              error: msg,
              source: "background"
            })
          }
          const error = buildRequestError(msg, resp?.status, resp?.data)
          if (!returnResponse) {
            throw markNoFallbackError(error)
          }
        }
        if (!returnResponse && responseType === "arrayBuffer") {
          const raw = (resp as any)?.data
          if (!isArrayBufferLike(raw)) {
            const storage = createSafeStorage()
            const fallback = await tldwRequest(
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
              { getConfig: () => storage.get("tldwConfig").catch(() => null) }
            )
            if (!fallback?.ok) {
              const msg = formatErrorMessage(
                fallback?.error,
                `Request failed: ${fallback?.status}`
              )
              const error = buildRequestError(msg, fallback?.status, fallback?.data)
              throw error
            }
            return fallback.data as T
          }
        }
        return (returnResponse ? resp : resp.data) as T
      }

      if (abortSignal.aborted) {
        throw markNoFallbackError(new Error("Aborted"))
      }

      const messagePromise = browser.runtime.sendMessage(payload) as Promise<
        { ok: boolean; error?: string; status?: number; data: T } | undefined
      >

      // Add timeout to extension messaging with abort signal support
      const extensionTimeoutWithSignal = 3000 // 3 second timeout
      const resp = await new Promise<
        { ok: boolean; error?: string; status?: number; data: T } | undefined | null
      >((resolve, reject) => {
        const onAbort = () => {
          reject(new Error('Aborted'))
        }
        const timeoutId = setTimeout(() => {
          abortSignal.removeEventListener('abort', onAbort)
          resolve(null) // timeout - fall through to direct request
        }, extensionTimeoutWithSignal)
        abortSignal.addEventListener('abort', onAbort, { once: true })
        messagePromise
          .then((r) => {
            clearTimeout(timeoutId)
            abortSignal.removeEventListener('abort', onAbort)
            resolve(r)
          })
          .catch((e) => {
            clearTimeout(timeoutId)
            abortSignal.removeEventListener('abort', onAbort)
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
        const msg = formatErrorMessage(
          resp?.error,
          `Request failed: ${resp?.status}`
        )
        if (!isAbortErrorMessage(msg)) {
          console.warn("[tldw:request]", method, path, resp?.status, msg)
          await recordRequestError({
            method: String(method),
            path: String(path),
            status: resp?.status,
            error: msg,
            source: "background"
          })
          notifyBackendUnavailable({
            method: String(method),
            path: String(path),
            status: resp?.status,
            error: msg,
            source: "background"
          })
        }
        const error = buildRequestError(msg, resp?.status, resp?.data)
        if (!returnResponse) {
          throw markNoFallbackError(error)
        }
      }
      if (!returnResponse && responseType === "arrayBuffer") {
        const raw = (resp as any)?.data
        if (!isArrayBufferLike(raw)) {
          const storage = createSafeStorage()
          const fallback = await tldwRequest(
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
            { getConfig: () => storage.get("tldwConfig").catch(() => null) }
          )
          if (!fallback?.ok) {
            const msg = formatErrorMessage(
              fallback?.error,
              `Request failed: ${fallback?.status}`
            )
            const error = buildRequestError(msg, fallback?.status, fallback?.data)
            throw error
          }
          return fallback.data as T
        }
      }
      return (returnResponse ? resp : resp.data) as T
    }
  } catch (e) {
    if (isNoFallbackError(e)) {
      if (isExtensionTimeoutError(e) && methodIsSafeFallback) {
        // Safe methods can fall through on timeout because duplicate side-effects are not expected.
      } else {
        throw e
      }
    } else if (!methodIsSafeFallback && !isExtensionTransportFailure(e)) {
      throw e
    }
  }

  // Fallback: direct fetch (web/dev context)
  const storage = createSafeStorage()
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
    { getConfig: () => storage.get("tldwConfig").catch(() => null) }
  )
  if (!resp?.ok) {
    const msg = formatErrorMessage(
      resp?.error,
      `Request failed: ${resp?.status}`
    )
    if (!isAbortErrorMessage(msg)) {
      console.warn("[tldw:request]", method, path, resp?.status, msg)
      await recordRequestError({
        method: String(method),
        path: String(path),
        status: resp?.status,
        error: msg,
        source: "direct"
      })
      notifyBackendUnavailable({
        method: String(method),
        path: String(path),
        status: resp?.status,
        error: msg,
        source: "direct"
      })
    }
    const error = buildRequestError(msg, resp?.status, resp?.data)
    if (!returnResponse) {
      throw error
    }
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
 * Direct fetch streaming implementation - used as fallback when extension messaging is unavailable or times out
 */
async function* bgStreamDirect<
  P extends AllowedPath = AllowedPath,
  M extends AllowedMethodFor<P> = AllowedMethodFor<P>
>(
  { path, method = 'POST' as UpperLower<M>, headers = {}, body, streamIdleTimeoutMs, abortSignal }: BgStreamInit<P, M>
): AsyncGenerator<string> {
  const storage = createSafeStorage()
  const cfg = (await storage.get<Record<string, unknown>>("tldwConfig").catch(() => null)) || null
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
  const advancedTransportOrigin =
    transport?.mode === "advanced" ? parseHttpOrigin(transport?.url) : null
  if (isAbsolute && !isAbsoluteUrlAllowlisted(absolutePath, cfg)) {
    throw new Error(ABSOLUTE_URL_BLOCK_ERROR)
  }
  if (
    !cfg?.serverUrl &&
    !isAbsolute &&
    transport?.mode === "advanced" &&
    !advancedTransportOrigin
  ) {
    throw new Error("tldw server not configured")
  }
  const baseUrl =
    advancedTransportOrigin ||
    (cfg?.serverUrl ? String(cfg.serverUrl).replace(/\/$/, "") : "")
  const url = isAbsolute
    ? absolutePath
    : transport?.url ||
      `${baseUrl}${String(normalizedPath).startsWith("/") ? "" : "/"}${String(normalizedPath)}`
  const sameOriginAbsolute = isAbsolute
    ? isSameOriginAbsoluteUrlForConfiguredServer(absolutePath, cfg)
    : false
  const shouldSkipAuth = isAbsolute && !sameOriginAbsolute
  const resolvedHeaders: Record<string, string> = { ...(headers || {}) }
  for (const k of Object.keys(resolvedHeaders)) {
    const kl = k.toLowerCase()
    if (kl === "x-api-key" || kl === "authorization") delete resolvedHeaders[k]
  }

  if (!shouldSkipAuth && !hostedMode && cfg?.authMode === "single-user") {
    const key = String(cfg?.apiKey || "").trim()
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
      const refreshResp = await fetch(`${baseUrl}/api/v1/auth/refresh`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: cfg.refreshToken })
      })
      if (refreshResp.ok) {
        const tokens = await refreshResp.json().catch(() => null)
        if (tokens?.access_token) {
          const latestCfg =
            (await storage
              .get<Record<string, unknown>>("tldwConfig")
              .catch(() => null)) || null
          const updated = {
            ...(latestCfg || cfg || {}),
            accessToken: tokens.access_token,
            refreshToken:
              tokens?.refresh_token ||
              latestCfg?.refreshToken ||
              cfg?.refreshToken
          }
          await storage.set("tldwConfig", updated)
          resolvedHeaders["Authorization"] = `Bearer ${tokens.access_token}`
          resp = await fetchStream()
        }
      }
    } catch {
      // ignore refresh failures and continue with original response
    }
  }

  if (!resp.ok) {
    const errorInfo = await parseStreamError(resp)
    const error = new Error(
      formatErrorMessage(errorInfo.message, `HTTP ${resp.status}`)
    ) as Error & { status?: number; details?: unknown }
    error.status = resp.status
    if (errorInfo.details) error.details = errorInfo.details
    throw error
  }
  if (!resp.body) {
    throw new Error("No response body")
  }

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

export async function* bgStream<
  P extends AllowedPath = AllowedPath,
  M extends AllowedMethodFor<P> = AllowedMethodFor<P>
>(
  { path, method = 'POST' as UpperLower<M>, headers = {}, body, streamIdleTimeoutMs, abortSignal }: BgStreamInit<P, M>
): AsyncGenerator<string> {
  const hasHttpStatus = (value: unknown): boolean =>
    extractHttpStatus(value) !== null

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
    yield* bgStreamDirect({ path, method, headers, body, streamIdleTimeoutMs, abortSignal })
    return
  }

  // Extension port-based streaming with connection-time and connection-establish fallback.
  let port: ReturnType<typeof browser.runtime.connect>
  try {
    port = browser.runtime.connect({ name: 'tldw:stream' })
  } catch (connectError) {
    if (!abortSignal?.aborted) {
      yield* bgStreamDirect({ path, method, headers, body, streamIdleTimeoutMs, abortSignal })
      return
    }
    throw connectError
  }
  const queue: string[] = []
  let done = false
  let error: any = null
  let firstDataReceived = false
  let connectionTimedOut = false

  // Connection timeout - if no data arrives within 5s, fall back to direct fetch
  const CONNECTION_TIMEOUT_MS = 5000
  const connectionTimer = setTimeout(() => {
    if (!firstDataReceived && !done) {
      connectionTimedOut = true
      done = true
      try { port.disconnect() } catch {}
    }
  }, CONNECTION_TIMEOUT_MS)

  const onMessage = (msg: any) => {
    if (msg?.event === 'data') {
      if (!firstDataReceived) {
        firstDataReceived = true
        clearTimeout(connectionTimer)
      }
      queue.push(msg.data as string)
    } else if (msg?.event === 'done') {
      done = true
    } else if (msg?.event === 'error') {
      const streamError = new Error(msg.message || 'Stream error') as Error & {
        status?: number
        details?: unknown
      }
      if (typeof msg.status === "number" && Number.isFinite(msg.status)) {
        streamError.status = Math.trunc(msg.status)
      }
      if (typeof msg.details !== "undefined" && msg.details !== null) {
        streamError.details = sanitizeResponseData(msg.details)
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
      port.postMessage({ path, method, headers, body, streamIdleTimeoutMs })
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
    // If connection timed out before receiving any data, fall back to direct fetch
    if (connectionTimedOut) {
      yield* bgStreamDirect({ path, method, headers, body, streamIdleTimeoutMs, abortSignal })
      return
    }
    const shouldFallbackAfterEarlyError =
      !firstDataReceived &&
      !abortSignal?.aborted &&
      Boolean(error) &&
      (isExtensionTransportFailure(error) || !hasHttpStatus(error))
    if (shouldFallbackAfterEarlyError) {
      yield* bgStreamDirect({ path, method, headers, body, streamIdleTimeoutMs, abortSignal })
      return
    }
    const shouldGracefullyEndAfterPartialStreamError =
      firstDataReceived &&
      !abortSignal?.aborted &&
      Boolean(error) &&
      (isExtensionTransportFailure(error) || !hasHttpStatus(error))
    if (shouldGracefullyEndAfterPartialStreamError) {
      // We already delivered data to the caller; avoid replaying non-idempotent
      // streamed requests after transport loss and let caller finalize partial output.
      const interruptionDetail =
        error instanceof Error ? error.message : String(error || "Stream transport interrupted")
      yield JSON.stringify({
        event: "stream_transport_interrupted",
        detail: interruptionDetail,
        partial_response_saved: true
      })
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

export interface BgUploadInit<P extends AllowedPath = AllowedPath, M extends AllowedMethodFor<P> = AllowedMethodFor<P>> {
  path: P
  method?: UpperLower<M>
  // key/value fields to include alongside file in FormData
  fields?: Record<string, any>
  // File payload as raw bytes with metadata (structured-cloneable)
  file?: { name?: string; type?: string; data: ArrayBuffer | Uint8Array | number[] }
  // Optional override for the multipart file field name
  fileFieldName?: string
  // Optional timeout override for upload requests
  timeoutMs?: number
  preferDirect?: boolean
}

export async function bgUpload<T = any, P extends AllowedPath = AllowedPath, M extends AllowedMethodFor<P> = AllowedMethodFor<P>>(
  {
    path,
    method = 'POST' as UpperLower<M>,
    fields = {},
    file,
    fileFieldName,
    timeoutMs,
    preferDirect = false
  }: BgUploadInit<P, M>
): Promise<T> {
  const hasRuntimeMessage =
    !preferDirect &&
    Boolean(browser?.runtime?.sendMessage && browser?.runtime?.id)
  const methodIsSafeFallback = isSafeFallbackMethod(method)
  if (hasRuntimeMessage) {
    try {
      // Add timeout to extension messaging for uploads
      const resolvedTimeout = typeof timeoutMs === "number" && timeoutMs > 0 ? timeoutMs : 60000
      const uploadTimeout = Math.max(5000, resolvedTimeout)
      const uploadPromise = browser.runtime.sendMessage({
        type: 'tldw:upload',
        payload: { path, method, fields, file, fileFieldName, timeoutMs: resolvedTimeout }
      })
      const uploadTimeoutPromise = new Promise<null>((resolve) =>
        setTimeout(() => resolve(null), uploadTimeout)
      )
      const resp = await Promise.race([uploadPromise, uploadTimeoutPromise]) as { ok: boolean; error?: string; status?: number; data: T } | undefined | null
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
      } else if (!methodIsSafeFallback && !isExtensionTransportFailure(e)) {
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
  if (file) {
    const name = file.name || "file"
    const type = file.type || "application/octet-stream"
    const toBytes = (data: ArrayBuffer | Uint8Array | number[]) => {
      if (data instanceof Uint8Array) return data
      if (data instanceof ArrayBuffer) return new Uint8Array(data)
      return Uint8Array.from(data)
    }
    const bytes = toBytes(file.data)
    if (typeof Blob === "undefined") {
      throw new Error("File upload is not supported in this environment.")
    }
    const buffer = bytes.buffer
    const slice =
      buffer instanceof ArrayBuffer
        ? buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
        : new Uint8Array(bytes).buffer
    const blob = new Blob([slice], { type })
    formData.append(fileFieldName || "file", blob, name)
  }

  const storage = createSafeStorage()
  const resp = await tldwRequest(
    { path, method, body: formData, timeoutMs },
    { getConfig: () => storage.get("tldwConfig").catch(() => null) }
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
