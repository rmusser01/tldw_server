import { formatErrorMessage } from "@/utils/format-error-message"
import { isPlaceholderApiKey } from "@/utils/api-key"
import type { PathOrUrl } from "@/services/tldw/openapi-guard"
import type { ApiSendResponse } from "@/services/api-send"
import { isHostedTldwDeployment } from "@/services/tldw/deployment-mode"
import {
  buildBrowserHttpBase,
  resolveBrowserTransport,
  type BrowserSurface
} from "@/services/tldw/browser-networking"

export type TldwRequestPayload = {
  path: PathOrUrl
  method?: string
  headers?: Record<string, string>
  body?: any
  noAuth?: boolean
  timeoutMs?: number
  abortSignal?: AbortSignal
  responseType?: "json" | "text" | "arrayBuffer"
}

type TldwConfigLike = Record<string, any> | null | undefined

type TldwRequestRuntime = {
  getConfig: () => Promise<TldwConfigLike>
  refreshAuth?: () => Promise<void>
  fetchFn?: typeof fetch
}

export type BrowserRequestTransport = {
  mode: "hosted" | "quickstart" | "advanced"
  kind: "same-origin" | "absolute"
  url: string
}

const ABSOLUTE_URL_BLOCK_ERROR =
  "Absolute URL requests are blocked unless the request origin is explicitly allowlisted."
const REQUEST_LOG_PREFIX = "[tldw:request]"
const malformedConfigServerUrlWarnings = new Set<string>()
const malformedAllowlistEntryWarnings = new Set<string>()

const toHostedProxyPath = (path: string): string => {
  const [pathname, search = ""] = path.split("?")
  if (pathname.startsWith("/api/v1/")) {
    const proxiedPath = pathname.replace(/^\/api\/v1\//, "/api/proxy/")
    return search ? `${proxiedPath}?${search}` : proxiedPath
  }
  return path
}

const normalizeKnownPathQuirks = (path: PathOrUrl): PathOrUrl => {
  if (typeof path !== "string") return path
  // Some callers still build media listing URLs as `/api/v1/media/?...`.
  // Certain proxies treat that as a distinct route and return 404.
  return path.replace("/api/v1/media/?", "/api/v1/media?") as PathOrUrl
}

const isMediaApiPath = (path: string): boolean => /\/api\/v1\/media(?:\/|\?|$)/.test(path)
const isFilesApiPath = (path: string): boolean => /\/api\/v1\/files(?:\/|\?|$)/.test(path)
const isSlidesApiPath = (path: string): boolean => /\/api\/v1\/slides(?:\/|\?|$)/.test(path)
const SLIDES_REQUEST_TIMEOUT_FLOOR_MS = 120000

const getCurrentBrowserSurface = (): BrowserSurface => {
  if (typeof window === "undefined") {
    return "extension"
  }

  try {
    const protocol = String(window.location?.protocol || "").trim().toLowerCase()
    if (protocol === "chrome-extension:" || protocol === "moz-extension:") {
      return "extension"
    }
    if (protocol === "http:" || protocol === "https:") {
      return "webui-page"
    }
  } catch {
    // Fall through to the browser-app default.
  }

  return "browser-app"
}

const joinOriginAndPath = (origin: string, path: string): string =>
  `${origin.replace(/\/$/, "")}${path.startsWith("/") ? "" : "/"}${path}`

export const deriveRequestTimeout = (
  cfg: TldwConfigLike,
  path: PathOrUrl,
  override?: number
): number => {
  if (override && override > 0) return override
  const p = String(normalizeKnownPathQuirks(path) || "")
  if (p.includes("/api/v1/chat/completions")) {
    return Number(cfg?.chatRequestTimeoutMs) > 0
      ? Number(cfg.chatRequestTimeoutMs)
      : Number(cfg?.requestTimeoutMs) > 0
        ? Number(cfg.requestTimeoutMs)
        : 10000
  }
  if (p.includes("/api/v1/rag/")) {
    return Number(cfg?.ragRequestTimeoutMs) > 0
      ? Number(cfg.ragRequestTimeoutMs)
      : Number(cfg?.requestTimeoutMs) > 0
        ? Number(cfg.requestTimeoutMs)
        : 10000
  }
  if (isMediaApiPath(p)) {
    return Number(cfg?.mediaRequestTimeoutMs) > 0
      ? Number(cfg.mediaRequestTimeoutMs)
      : Number(cfg?.requestTimeoutMs) > 0
        ? Number(cfg.requestTimeoutMs)
        : 10000
  }
  if (isFilesApiPath(p)) {
    return Number(cfg?.mediaRequestTimeoutMs) > 0
      ? Number(cfg.mediaRequestTimeoutMs)
      : Number(cfg?.requestTimeoutMs) > 0
        ? Number(cfg.requestTimeoutMs)
        : 10000
  }
  if (isSlidesApiPath(p)) {
    const configuredTimeout =
      Number(cfg?.requestTimeoutMs) > 0 ? Number(cfg.requestTimeoutMs) : 0
    return Math.max(configuredTimeout, SLIDES_REQUEST_TIMEOUT_FLOOR_MS)
  }
  return Number(cfg?.requestTimeoutMs) > 0
    ? Number(cfg.requestTimeoutMs)
    : 10000
}

export const parseRetryAfter = (headerValue?: string | null): number | null => {
  if (!headerValue) return null
  const asNumber = Number(headerValue)
  if (!Number.isNaN(asNumber)) {
    return Math.max(0, asNumber * 1000)
  }
  const asDate = Date.parse(headerValue)
  if (!Number.isNaN(asDate)) {
    return Math.max(0, asDate - Date.now())
  }
  return null
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

const warnMalformedServerUrl = (raw: string, error: unknown) => {
  const key = raw.trim()
  if (!key || malformedConfigServerUrlWarnings.has(key)) return
  malformedConfigServerUrlWarnings.add(key)
  console.warn(
    `${REQUEST_LOG_PREFIX} Invalid configured serverUrl: ${key}`,
    error
  )
}

const warnMalformedAllowlistEntry = (raw: string, error: unknown) => {
  const key = raw.trim()
  if (!key || malformedAllowlistEntryWarnings.has(key)) return
  malformedAllowlistEntryWarnings.add(key)
  console.warn(
    `${REQUEST_LOG_PREFIX} Invalid absoluteUrlAllowlist entry: ${key}`,
    error
  )
}

const parseConfiguredServerOrigin = (cfg: TldwConfigLike): string | null => {
  const configuredServerUrl = String(
    (cfg as Record<string, unknown> | null)?.serverUrl || ""
  ).trim()
  if (!configuredServerUrl) return null
  try {
    const serverParsed = new URL(configuredServerUrl)
    if (!/^https?:$/i.test(serverParsed.protocol)) return null
    return serverParsed.origin.toLowerCase()
  } catch (error) {
    warnMalformedServerUrl(configuredServerUrl, error)
    return null
  }
}

const absoluteOriginAllowlistFromConfig = (cfg: TldwConfigLike): Set<string> => {
  const out = new Set<string>()
  const configuredServerUrl = String((cfg as Record<string, unknown> | null)?.serverUrl || "").trim()
  if (configuredServerUrl) {
    try {
      const serverParsed = new URL(configuredServerUrl)
      if (/^https?:$/i.test(serverParsed.protocol)) {
        out.add(serverParsed.origin.toLowerCase())
      }
    } catch {
      // Ignore malformed configured server URL.
    }
  }
  const entries = toAllowlistEntries((cfg as Record<string, unknown> | null)?.absoluteUrlAllowlist)
  for (const entry of entries) {
    try {
      const parsed = new URL(entry)
      if (/^https?:$/i.test(parsed.protocol)) {
        out.add(parsed.origin.toLowerCase())
      }
    } catch (error) {
      warnMalformedAllowlistEntry(entry, error)
    }
  }
  return out
}

const isSameOriginAbsoluteUrlForConfiguredServer = (
  absoluteUrl: string,
  cfg: TldwConfigLike
): boolean => {
  const configuredServerOrigin = parseConfiguredServerOrigin(cfg)
  if (!configuredServerOrigin) return false
  try {
    const target = new URL(absoluteUrl)
    if (!/^https?:$/i.test(target.protocol)) return false
    return target.origin.toLowerCase() === configuredServerOrigin
  } catch {
    return false
  }
}

const isAbsoluteUrlAllowlisted = (
  absoluteUrl: string,
  cfg: TldwConfigLike
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

export const resolveBrowserRequestTransport = ({
  config,
  path,
  pageOrigin
}: {
  config: TldwConfigLike
  path: string
  pageOrigin?: string | null
}): BrowserRequestTransport => {
  if (isHostedTldwDeployment()) {
    return {
      mode: "hosted",
      kind: "same-origin",
      url: toHostedProxyPath(path)
    }
  }

  const configuredServerUrl = String(
    (config as Record<string, unknown> | null)?.serverUrl || ""
  ).trim()
  const surface = getCurrentBrowserSurface()
  if (surface === "webui-page") {
    try {
      const resolved = resolveBrowserTransport({
        surface,
        deploymentMode: process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE,
        pageOrigin:
          pageOrigin ??
          (typeof window === "undefined"
            ? null
            : String(window.location?.origin || "").trim()),
        apiOrigin: configuredServerUrl || process.env.NEXT_PUBLIC_API_URL
      })
      const browserHttpBase = buildBrowserHttpBase(resolved)
      if (!browserHttpBase) {
        return {
          mode: "quickstart",
          kind: "same-origin",
          url: path
        }
      }

      return {
        mode: "advanced",
        kind: "absolute",
        url: joinOriginAndPath(browserHttpBase, path)
      }
    } catch {
      // Fall through to explicit configured server handling below.
    }
  }

  return {
    mode: "advanced",
    kind: "absolute",
    url: joinOriginAndPath(configuredServerUrl, path)
  }
}

export const tldwRequest = async (
  payload: TldwRequestPayload,
  runtime: TldwRequestRuntime
): Promise<ApiSendResponse> => {
  const {
    path,
    method = "GET",
    headers = {},
    body,
    noAuth = false,
    timeoutMs: overrideTimeoutMs,
    abortSignal,
    responseType
  } = payload || {}
  const normalizedPath = normalizeKnownPathQuirks(path)
  const fetchFn = runtime.fetchFn || fetch
  const cfg = await runtime.getConfig()
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
  const sameOriginAbsoluteUrl =
    isAbsolute && isSameOriginAbsoluteUrlForConfiguredServer(absolutePath, cfg)
  if (
    isAbsolute &&
    !sameOriginAbsoluteUrl &&
    !isAbsoluteUrlAllowlisted(absolutePath, cfg)
  ) {
    return {
      ok: false,
      status: 400,
      error: ABSOLUTE_URL_BLOCK_ERROR
    }
  }
  if (!cfg?.serverUrl && !isAbsolute && transport?.mode === "advanced") {
    return { ok: false, status: 400, error: "tldw server not configured" }
  }
  if (!normalizedPath) {
    return { ok: false, status: 400, error: "Request path is required" }
  }
  const url = isAbsolute
    ? normalizedPath
    : transport?.url || String(normalizedPath)
  const shouldSkipAuth = noAuth || (isAbsolute && !sameOriginAbsoluteUrl)
  const h: Record<string, string> = { ...(headers || {}) }
  const hasContentType = Object.keys(h).some(
    (key) => key.toLowerCase() === "content-type"
  )
  const isBinaryBody = (value: any) => {
    if (!value || typeof value !== "object") return false
    if (typeof FormData !== "undefined" && value instanceof FormData) return true
    if (typeof Blob !== "undefined" && value instanceof Blob) return true
    if (
      typeof URLSearchParams !== "undefined" &&
      value instanceof URLSearchParams
    ) {
      return true
    }
    if (typeof ArrayBuffer !== "undefined") {
      if (value instanceof ArrayBuffer) return true
      if (ArrayBuffer.isView?.(value)) return true
    }
    return false
  }
  if (body != null && !hasContentType && typeof body !== "string" && !isBinaryBody(body)) {
    h["Content-Type"] = "application/json"
  }
  if (!shouldSkipAuth) {
    for (const k of Object.keys(h)) {
      const kl = k.toLowerCase()
      if (kl === "x-api-key" || kl === "authorization") delete h[k]
    }
    if (!hostedMode) {
      if (cfg?.authMode === "single-user") {
        const key = (cfg?.apiKey || "").trim()
        if (!key) {
          return {
            ok: false,
            status: 401,
            error:
              "Add or update your API key in Settings -> tldw server, then try again."
          }
        }
        if (isPlaceholderApiKey(key)) {
          return {
            ok: false,
            status: 401,
            error:
              "tldw server API key is still set to a placeholder value. Replace it with your real API key in Settings -> tldw server before continuing."
          }
        }
        h["X-API-KEY"] = key
      } else if (cfg?.authMode === "multi-user") {
        const token = (cfg?.accessToken || "").trim()
        if (token) h["Authorization"] = `Bearer ${token}`
        else {
          return {
            ok: false,
            status: 401,
            error: "Not authenticated. Please login under Settings > tldw."
          }
        }
      }
    }
    if (cfg?.orgId) {
      h["X-TLDW-Org-Id"] = String(cfg.orgId)
    }
  }

  const controller = new AbortController()
  const timeoutMs = deriveRequestTimeout(cfg, normalizedPath, Number(overrideTimeoutMs))
  const onAbort = () => {
    try {
      controller.abort()
    } catch {}
  }
  let timeoutId: ReturnType<typeof setTimeout> | null = null
  let retryTimeoutId: ReturnType<typeof setTimeout> | null = null

  try {
    timeoutId = setTimeout(() => controller.abort(), timeoutMs)
    if (abortSignal) {
      if (abortSignal.aborted) {
        controller.abort()
      } else {
        abortSignal.addEventListener("abort", onAbort, { once: true })
      }
    }

    const resolvedBody =
      body == null
        ? undefined
        : typeof body === "string" || isBinaryBody(body)
          ? body
          : JSON.stringify(body)

    let resp = await fetchFn(url, {
      method,
      headers: h,
      body: resolvedBody,
      signal: controller.signal
    })
    if (timeoutId) {
      clearTimeout(timeoutId)
      timeoutId = null
    }

    if (
      !shouldSkipAuth &&
      !hostedMode &&
      resp.status === 401 &&
      cfg?.authMode === "multi-user" &&
      cfg?.refreshToken &&
      runtime.refreshAuth
    ) {
      let refreshSucceeded = false
      try {
        await runtime.refreshAuth()
        refreshSucceeded = true
      } catch (refreshError) {
        console.warn(
          `${REQUEST_LOG_PREFIX} Token refresh failed — retrying with stale token`,
          refreshError
        )
      }
      const updated = await runtime.getConfig()
      const retryHeaders = { ...h }
      for (const k of Object.keys(retryHeaders)) {
        const kl = k.toLowerCase()
        if (kl === "authorization" || kl === "x-api-key") delete retryHeaders[k]
      }
      if (updated?.accessToken) retryHeaders["Authorization"] = `Bearer ${updated.accessToken}`
      const retryController = new AbortController()
      retryTimeoutId = setTimeout(() => retryController.abort(), timeoutMs)
      resp = await fetchFn(url, {
        method,
        headers: retryHeaders,
        body: body ? (typeof body === "string" ? body : JSON.stringify(body)) : undefined,
        signal: retryController.signal
      })
      if (retryTimeoutId) {
        clearTimeout(retryTimeoutId)
        retryTimeoutId = null
      }
      if (!refreshSucceeded && resp.status === 401) {
        return {
          ok: false,
          status: 401,
          error: "Session expired. Please log in again."
        }
      }
    }

    const headersOut: Record<string, string> = {}
    try {
      resp.headers.forEach((value, key) => {
        headersOut[key] = value
      })
    } catch {}

    const retryAfterMs = parseRetryAfter(resp.headers?.get?.("retry-after"))
    const contentType = resp.headers.get("content-type") || ""
    let data: any = null
    const readDefaultBody = async () => {
      if (contentType.includes("application/json")) {
        return await resp.json().catch(() => null)
      }
      return await resp.text().catch(() => null)
    }
    if (responseType === "arrayBuffer") {
      data = resp.ok ? await resp.arrayBuffer().catch(() => null) : await readDefaultBody()
    } else if (responseType === "json") {
      data = await resp.json().catch(() => null)
    } else if (responseType === "text") {
      data = await resp.text().catch(() => null)
    } else {
      data = await readDefaultBody()
    }

    if (!resp.ok) {
      let detail: unknown = undefined
      if (typeof data === "object" && data) {
        const raw = data.detail ?? data.error ?? data.message
        // FastAPI validation errors return detail as an array
        if (Array.isArray(raw)) {
          detail = raw
            .map((item: any) =>
              typeof item === "string"
                ? item
                : typeof item?.msg === "string"
                  ? item.msg
                  : JSON.stringify(item)
            )
            .join("; ")
        } else if (raw !== undefined && raw !== null) {
          detail = raw
        }
      }
      const errorMessage = formatErrorMessage(
        detail !== undefined && detail !== null
          ? detail
          : resp.statusText || `HTTP ${resp.status}`,
        `HTTP ${resp.status}`
      )
      return {
        ok: false,
        status: resp.status,
        error: errorMessage,
        data,
        headers: headersOut,
        retryAfterMs
      }
    }

    return { ok: true, status: resp.status, data, headers: headersOut, retryAfterMs }
  } catch (e: any) {
    return {
      ok: false,
      status: 0,
      error: formatErrorMessage(e, "Network error")
    }
  } finally {
    if (timeoutId) {
      clearTimeout(timeoutId)
    }
    if (retryTimeoutId) {
      clearTimeout(retryTimeoutId)
    }
    if (abortSignal) {
      try {
        abortSignal.removeEventListener("abort", onAbort)
      } catch {}
    }
  }
}
