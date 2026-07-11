export type BrowserSurface = "webui-page" | "extension" | "browser-app"

export type BrowserTransportMode = "quickstart" | "advanced"

export type BrowserTransport = {
  surface: BrowserSurface
  mode: BrowserTransportMode
  pageOrigin: string
  apiOrigin: string
}

export type BrowserNetworkingIssue = {
  kind: "loopback_api_not_browser_reachable"
  apiOrigin: string
  pageOrigin: string
}

export const COOKIE_SESSION_CONFIG_KEY = "tldwCookieSessionConfig"

export type CookieSessionBrowserTransportInput = {
  authMode?: unknown
  authSource?: unknown
  transportMode?: unknown
  transportKind?: unknown
  pageOrigin?: string | null
}

type ResolveBrowserTransportInput = {
  surface: BrowserSurface
  deploymentMode?: string | null
  pageOrigin?: string | null
  apiOrigin?: string | null
}

type AdvancedRequestTransportLike = {
  mode: string
  url: string
}

type ResolveAdvancedRequestTransportGuardInput = {
  transport?: AdvancedRequestTransportLike | null
  hasConfiguredServerUrl: boolean
  isAbsolute: boolean
}

export type AdvancedRequestTransportGuard = {
  origin: string | null
  isUnconfigured: boolean
}

const trimTrailingSlash = (value: string): string => value.replace(/\/+$/, "")

const normalizeString = (value?: string | null): string =>
  typeof value === "string" ? value.trim() : ""

const parseHttpOrigin = (value?: string | null): URL | null => {
  const normalized = normalizeString(value)
  if (!normalized) {
    return null
  }

  try {
    const parsed = new URL(normalized)
    if (!/^https?:$/i.test(parsed.protocol)) {
      return null
    }
    return parsed
  } catch {
    return null
  }
}

export const isCookieSessionBrowserTransport = (
  input: CookieSessionBrowserTransportInput
): boolean =>
  input.authMode === "single-user" &&
  input.authSource === "cookie-session" &&
  input.transportMode === "quickstart" &&
  input.transportKind === "same-origin" &&
  parseHttpOrigin(input.pageOrigin) !== null

const normalizeOrigin = (value?: string | null): string => {
  const parsed = parseHttpOrigin(value)
  if (parsed) {
    return trimTrailingSlash(parsed.origin)
  }

  return trimTrailingSlash(normalizeString(value))
}

export const resolveAdvancedRequestTransportGuard = ({
  transport,
  hasConfiguredServerUrl,
  isAbsolute
}: ResolveAdvancedRequestTransportGuardInput): AdvancedRequestTransportGuard => {
  const parsedOrigin =
    transport?.mode === "advanced" ? parseHttpOrigin(transport.url) : null
  const origin = parsedOrigin?.origin.toLowerCase() ?? null

  return {
    origin,
    isUnconfigured:
      !hasConfiguredServerUrl &&
      !isAbsolute &&
      transport?.mode === "advanced" &&
      !origin
  }
}

export const resolveBrowserTransportMode = (
  deploymentMode?: string | null
): BrowserTransportMode =>
  normalizeString(deploymentMode) === "quickstart" ? "quickstart" : "advanced"

export const isLoopbackHost = (hostname: string): boolean => {
  const normalized = normalizeString(hostname).toLowerCase()
  return (
    normalized === "localhost" ||
    normalized === "127.0.0.1" ||
    normalized === "::1" ||
    normalized === "[::1]"
  )
}

export const resolveBrowserTransport = (
  input: ResolveBrowserTransportInput
): BrowserTransport => {
  const mode = resolveBrowserTransportMode(input.deploymentMode)
  const pageOrigin = normalizeOrigin(input.pageOrigin)

  if (mode === "quickstart") {
    return {
      surface: input.surface,
      mode,
      pageOrigin,
      apiOrigin: ""
    }
  }

  const parsedApiOrigin = parseHttpOrigin(input.apiOrigin)
  if (!parsedApiOrigin) {
    throw new Error(
      "Invalid WebUI networking config: advanced mode requires NEXT_PUBLIC_API_URL to be an absolute URL."
    )
  }

  return {
    surface: input.surface,
    mode,
    pageOrigin,
    apiOrigin: trimTrailingSlash(parsedApiOrigin.origin)
  }
}

export const buildBrowserHttpBase = (resolved: BrowserTransport): string =>
  resolved.mode === "quickstart" ? "" : resolved.apiOrigin

export const buildBrowserWebSocketBase = (resolved: BrowserTransport): string => {
  const origin =
    resolved.mode === "quickstart" ? resolved.pageOrigin : resolved.apiOrigin
  const parsedOrigin = parseHttpOrigin(origin)

  if (!parsedOrigin) {
    return ""
  }

  return `${parsedOrigin.protocol === "https:" ? "wss:" : "ws:"}//${parsedOrigin.host}`
}

export const resolveWebUiQuickstartServerUrl = (
  input: ResolveBrowserTransportInput
): string | null => {
  const resolved = resolveBrowserTransport(input)
  if (resolved.surface !== "webui-page" || resolved.mode !== "quickstart") {
    return null
  }

  return resolved.pageOrigin || null
}

export const detectBrowserNetworkingIssue = (
  input: ResolveBrowserTransportInput
): BrowserNetworkingIssue | undefined => {
  if (input.surface !== "webui-page") {
    return undefined
  }

  const pageOrigin = normalizeOrigin(input.pageOrigin)
  if (!pageOrigin) {
    return undefined
  }

  let resolved: BrowserTransport
  try {
    resolved = resolveBrowserTransport(input)
  } catch {
    return undefined
  }

  if (resolved.mode !== "advanced" || !resolved.apiOrigin) {
    return undefined
  }

  const parsedApiOrigin = parseHttpOrigin(resolved.apiOrigin)
  const parsedPageOrigin = parseHttpOrigin(pageOrigin)

  if (!parsedApiOrigin || !parsedPageOrigin) {
    return undefined
  }

  if (
    isLoopbackHost(parsedApiOrigin.hostname) &&
    !isLoopbackHost(parsedPageOrigin.hostname)
  ) {
    return {
      kind: "loopback_api_not_browser_reachable",
      apiOrigin: resolved.apiOrigin,
      pageOrigin
    }
  }

  return undefined
}
