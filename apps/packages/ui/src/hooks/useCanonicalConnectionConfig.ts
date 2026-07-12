import React from "react"
import { useStorage } from "@plasmohq/storage/hook"

import { tldwClient, type TldwConfig } from "@/services/tldw/TldwApiClient"
import {
  resolveWebUiQuickstartServerUrl,
  type BrowserSurface
} from "@/services/tldw/browser-networking"

const DEFAULT_SERVER_URL = "http://127.0.0.1:8000"

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

const getQuickstartWebUiServerUrl = (
): string | null => {
  try {
    return resolveWebUiQuickstartServerUrl({
      surface: getCurrentBrowserSurface(),
      deploymentMode: process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE,
      pageOrigin:
        typeof window === "undefined" ? null : String(window.location?.origin || "").trim(),
      apiOrigin: process.env.NEXT_PUBLIC_API_URL
    })
  } catch {
    return null
  }
}

const resolveCanonicalServerUrl = (
  configuredServerUrl: string | null | undefined,
  fallbackServerUrl: string
): string => {
  const quickstartWebUiServerUrl = getQuickstartWebUiServerUrl()
  if (quickstartWebUiServerUrl) {
    return quickstartWebUiServerUrl
  }

  return typeof configuredServerUrl === "string" && configuredServerUrl.trim().length > 0
    ? configuredServerUrl
    : fallbackServerUrl
}

const normalizeConnectionConfig = (
  config: TldwConfig | null | undefined,
  fallback: TldwConfig
): TldwConfig => ({
  serverUrl: resolveCanonicalServerUrl(config?.serverUrl, fallback.serverUrl),
  authMode:
    config?.authMode === "multi-user" || config?.authMode === "single-user"
      ? config.authMode
      : fallback.authMode,
  apiKey: config?.apiKey ?? fallback.apiKey,
  accessToken: config?.accessToken ?? fallback.accessToken,
  refreshToken: config?.refreshToken ?? fallback.refreshToken,
  orgId: typeof config?.orgId === "number" ? config.orgId : fallback.orgId,
  authSource: config?.authSource ?? fallback.authSource
})

export const useCanonicalConnectionConfig = (): {
  config: TldwConfig | null
  loading: boolean
} => {
  const [legacyServerUrl] = useStorage(
    "serverUrl",
    getQuickstartWebUiServerUrl() || DEFAULT_SERVER_URL
  )
  const [legacyAuthMode] = useStorage("authMode", "single-user")
  const [legacyApiKey] = useStorage("apiKey", "")
  const [legacyAccessToken] = useStorage("accessToken", "")

  const fallbackConfig = React.useMemo<TldwConfig>(
    () => ({
      serverUrl:
        resolveCanonicalServerUrl(legacyServerUrl, DEFAULT_SERVER_URL),
      authMode: legacyAuthMode === "multi-user" ? "multi-user" : "single-user",
      apiKey: legacyApiKey || undefined,
      accessToken: legacyAccessToken || undefined
    }),
    [
      legacyAccessToken,
      legacyApiKey,
      legacyAuthMode,
      legacyServerUrl
    ]
  )

  const [config, setConfig] = React.useState<TldwConfig | null>(null)
  const [loading, setLoading] = React.useState(true)

  React.useEffect(() => {
    let cancelled = false

    const resolveConfig = async () => {
      setLoading(true)
      try {
        const canonicalConfig = await tldwClient.getConfig()
        if (!cancelled) {
          setConfig(
            normalizeConnectionConfig(canonicalConfig, fallbackConfig)
          )
        }
      } catch {
        if (!cancelled) {
          setConfig(fallbackConfig)
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void resolveConfig()

    return () => {
      cancelled = true
    }
  }, [fallbackConfig])

  return { config, loading }
}
