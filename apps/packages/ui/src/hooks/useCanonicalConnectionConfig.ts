import React from "react"
import { useStorage } from "@plasmohq/storage/hook"

import { tldwClient, type TldwConfig } from "@/services/tldw/TldwApiClient"
import {
  resolveWebUiQuickstartServerUrl,
  type BrowserSurface
} from "@/services/tldw/browser-networking"
import {
  applyRefreshRotation,
  REFRESH_ROTATION_KEY
} from "@/services/tldw/single-user-credential"
import { connectionAuthoritiesMatch } from "@/services/chat-surface-scope"
import { servicePromptTargetsMatch } from "@/services/tldw/service-prompt-scope-error"

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
  /** False during rehydration only when stored lineage preserves resolved authority. */
  authorityLoading: boolean
} => {
  const [legacyServerUrl] = useStorage(
    "serverUrl",
    getQuickstartWebUiServerUrl() || DEFAULT_SERVER_URL
  )
  const [legacyAuthMode] = useStorage("authMode", "single-user")
  const [legacyApiKey] = useStorage("apiKey", "")
  const [legacyAccessToken] = useStorage("accessToken", "")
  const [refreshRotation] = useStorage<unknown>(REFRESH_ROTATION_KEY)
  const [storedConfig] = useStorage<TldwConfig | null>("tldwConfig", null)

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
  const [resolvedAuthority, setResolvedAuthority] = React.useState<{
    config: TldwConfig
    fallbackConfig: TldwConfig
    storedConfig: TldwConfig | null
    refreshRotation: unknown
  } | null>(null)

  const sameInputs = resolvedAuthority?.fallbackConfig === fallbackConfig &&
    resolvedAuthority?.storedConfig === storedConfig &&
    resolvedAuthority?.refreshRotation === refreshRotation
  let authorityPreserved = false
  if (resolvedAuthority && storedConfig && typeof storedConfig === "object" &&
    connectionAuthoritiesMatch(fallbackConfig, resolvedAuthority.fallbackConfig)) {
    const previous = applyRefreshRotation(storedConfig, resolvedAuthority.refreshRotation)
    const next = applyRefreshRotation(storedConfig, refreshRotation)
    // Prove the source lineage against the last canonical credential pair;
    // matching claims in a replacement token alone cannot establish continuity.
    authorityPreserved =
      (refreshRotation === resolvedAuthority.refreshRotation || next !== storedConfig) &&
      servicePromptTargetsMatch(previous, resolvedAuthority.config) &&
      previous.accessToken === resolvedAuthority.config.accessToken &&
      previous.refreshToken === resolvedAuthority.config.refreshToken &&
      connectionAuthoritiesMatch(previous, resolvedAuthority.config) &&
      connectionAuthoritiesMatch(next, resolvedAuthority.config)
  }
  const authorityLoading = !resolvedAuthority || (!sameInputs && !authorityPreserved)

  React.useEffect(() => {
    let cancelled = false

    const resolveConfig = async () => {
      setLoading(true)
      try {
        const canonicalConfig = await tldwClient.getConfig()
        if (!cancelled) {
          const next = normalizeConnectionConfig(canonicalConfig, fallbackConfig)
          setConfig(next)
          setResolvedAuthority(canonicalConfig ? {
            config: next, fallbackConfig, storedConfig, refreshRotation
          } : null)
        }
      } catch {
        if (!cancelled) {
          setConfig(fallbackConfig)
          setResolvedAuthority(null)
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
  }, [fallbackConfig, refreshRotation, storedConfig])

  return { config, loading, authorityLoading }
}
