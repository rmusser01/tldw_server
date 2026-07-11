import {
  buildBrowserWebSocketBase,
  isCookieSessionBrowserTransport,
  resolveBrowserTransport,
  type BrowserSurface
} from "@/services/tldw/browser-networking"

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

const getCurrentPageOrigin = (): string | null => {
  if (typeof window === "undefined") {
    return null
  }

  try {
    const origin = String(window.location?.origin || "").trim()
    return origin || null
  } catch {
    return null
  }
}

const fallbackWebSocketBase = (serverUrl: string): string =>
  serverUrl.replace(/^http/i, "ws").replace(/\/$/, "")

export const resolveBrowserWebSocketBase = (serverUrl: string): string => {
  const normalizedServerUrl = String(serverUrl || "").trim()
  if (!normalizedServerUrl) {
    return ""
  }

  try {
    const resolved = resolveBrowserTransport({
      surface: getCurrentBrowserSurface(),
      deploymentMode: process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE,
      pageOrigin: getCurrentPageOrigin(),
      apiOrigin: normalizedServerUrl || process.env.NEXT_PUBLIC_API_URL
    })

    return buildBrowserWebSocketBase(resolved) || fallbackWebSocketBase(normalizedServerUrl)
  } catch {
    return fallbackWebSocketBase(normalizedServerUrl)
  }
}

export const resolveCookieSessionWebSocketBase = (config: {
  serverUrl?: string | null
  authMode?: unknown
  authSource?: unknown
}): string | null => {
  const pageOrigin = getCurrentPageOrigin()
  const surface = getCurrentBrowserSurface()
  if (!pageOrigin) return null

  try {
    const transport = resolveBrowserTransport({
      surface,
      deploymentMode: process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE,
      pageOrigin,
      apiOrigin: String(config.serverUrl || "").trim()
    })
    if (
      !isCookieSessionBrowserTransport({
        authMode: config.authMode,
        authSource: config.authSource,
        transportMode: transport.mode,
        transportKind: transport.mode === "quickstart" ? "same-origin" : "absolute",
        pageOrigin
      })
    ) {
      return null
    }
    return buildBrowserWebSocketBase(transport) || null
  } catch {
    return null
  }
}
