import { browser } from "./wxt-browser"
import { clearRuntimeAuth } from "@web/lib/authStorage"
import { createSafeStorage } from "@/utils/safe-storage"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"
import { FEATURE_FLAGS } from "@/hooks/useFeatureFlags"
import {
  DEFAULT_HEADER_SHORTCUT_SELECTION,
  DEFAULT_SIDEBAR_SHORTCUT_SELECTION,
  HEADER_SHORTCUTS_EXPANDED_SETTING,
  HEADER_SHORTCUT_SELECTION_SETTING,
  SIDEBAR_ACTIVE_TAB_SETTING,
  SIDEBAR_SHORTCUTS_COLLAPSED_SETTING,
  SIDEBAR_SHORTCUT_SELECTION_SETTING,
  THEME_SETTING,
  UI_MODE_SETTING
} from "@/services/settings/ui-settings"
import {
  COOKIE_SESSION_CONFIG_KEY,
  resolveWebUiQuickstartServerUrl,
  type BrowserSurface
} from "@/services/tldw/browser-networking"
import {
  activateCookieSessionConfig,
  clearRuntimeAuthOverride,
  invalidateCookieSessionConfig
} from "@/services/tldw/runtime-auth-override"

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === "object" && value !== null
}

const mergeMissingProperties = (
  target: Record<string, unknown>,
  source: Record<string, unknown>
) => {
  Object.entries(source).forEach(([key, sourceValue]) => {
    const targetValue = target[key]

    if (targetValue === undefined || targetValue === null) {
      try {
        target[key] = sourceValue
      } catch {
        // Ignore write failures on non-configurable host objects.
      }
      return
    }

    if (isRecord(targetValue) && isRecord(sourceValue)) {
      mergeMissingProperties(targetValue, sourceValue)
    }
  })
}

if (typeof globalThis !== "undefined") {
  const globalScope = globalThis as typeof globalThis & {
    browser?: typeof browser
    chrome?: typeof browser | Record<string, unknown>
  }

  if (!isRecord(globalScope.browser)) {
    globalScope.browser = browser
  } else {
    mergeMissingProperties(
      globalScope.browser as unknown as Record<string, unknown>,
      browser as unknown as Record<string, unknown>
    )
  }

  if (!isRecord(globalScope.chrome)) {
    ;(globalScope as any).chrome = browser
  } else {
    mergeMissingProperties(
      globalScope.chrome as Record<string, unknown>,
      browser as unknown as Record<string, unknown>
    )
  }
}

const normalizeBaseUrl = (value?: string | null): string | null => {
  const raw = (value || "").trim()
  if (!raw) return null
  return raw.replace(/\/$/, "")
}

const getCurrentBrowserOrigin = (): string | null => {
  if (typeof window === "undefined") return null
  try {
    return normalizeBaseUrl(window.location?.origin)
  } catch {
    return null
  }
}

const isCurrentBrowserOrigin = (value?: string | null): boolean => {
  const currentOrigin = getCurrentBrowserOrigin()
  const normalized = normalizeBaseUrl(value)
  return Boolean(currentOrigin && normalized && currentOrigin === normalized)
}

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
      pageOrigin: getCurrentBrowserOrigin(),
      apiOrigin: process.env.NEXT_PUBLIC_API_URL
    })
  } catch {
    return null
  }
}

const getCurrentBrowserHostname = (): string | null => {
  if (typeof window === "undefined") return null
  try {
    const hostname = String(window.location?.hostname || "").trim().toLowerCase()
    return hostname || null
  } catch {
    return null
  }
}

const isLocalhostLikeHostname = (value?: string | null): boolean => {
  const normalized = String(value || "").trim().toLowerCase()
  return (
    normalized === "localhost" ||
    normalized === "127.0.0.1" ||
    normalized === "::1" ||
    normalized === "[::1]"
  )
}

const parsePrivateIpv4Host = (value?: string | null): number[] | null => {
  const normalized = String(value || "").trim().toLowerCase()
  const match = normalized.match(/^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/)
  if (!match) return null

  const parts = match.slice(1).map((raw) => Number(raw))
  if (parts.some((part) => Number.isNaN(part) || part < 0 || part > 255)) {
    return null
  }

  const [a, b] = parts
  if (a === 10) return parts
  if (a === 192 && b === 168) return parts
  if (a === 172 && b >= 16 && b <= 31) return parts
  return null
}

const formatHostnameForUrl = (value: string): string => {
  return value.includes(":") && !value.startsWith("[") ? `[${value}]` : value
}

const deriveCurrentHostRecoveryServerUrl = (
  configuredServerUrl?: string | null
): string | null => {
  if (!configuredServerUrl) return null

  const browserHostname = getCurrentBrowserHostname()
  if (!browserHostname) return null

  try {
    const parsed = new URL(String(configuredServerUrl))
    const configuredHost = String(parsed.hostname || "").trim().toLowerCase()
    if (!configuredHost || configuredHost === browserHostname) return null

    const configuredPrivateIp = parsePrivateIpv4Host(configuredHost)
    const browserPrivateIp = parsePrivateIpv4Host(browserHostname)
    const configuredIsLocal = isLocalhostLikeHostname(configuredHost)
    const browserIsLocal = isLocalhostLikeHostname(browserHostname)

    const shouldRecover =
      (configuredPrivateIp && browserIsLocal) ||
      (configuredIsLocal && browserPrivateIp) ||
      (configuredPrivateIp && browserPrivateIp)
    if (!shouldRecover) return null

    const port = parsed.port || "8000"
    return `${parsed.protocol}//${formatHostnameForUrl(browserHostname)}:${port}`
  } catch {
    return null
  }
}

const DEFAULT_TLDW_SERVER_URL = "http://127.0.0.1:8000"
const RUNTIME_CONFIG_ENDPOINT = "/api/_tldw-webui/runtime-config"
const RUNTIME_SESSION_ENDPOINT = "/api/_tldw-webui/session"
const RUNTIME_PROFILE_PROBE_ENDPOINT = "/api/v1/users/me/profile"
const RUNTIME_REQUEST_TIMEOUT_MS = 8_000
const RUNTIME_AUTH_METADATA_KEY = "tldwRuntimeAuthMetadata"
const MANUAL_SESSION_API_KEY = "tldwManualSessionApiKey"
const LEGACY_RUNTIME_KEYS = [
  "apiKey",
  "tldwRuntimeSessionSingleUserApiKey",
  RUNTIME_AUTH_METADATA_KEY
] as const

type RuntimeConfigPayload = {
  runtimeAuth?: {
    available?: boolean
    authMode?: "single-user" | string
    transport?: "cookie-session" | string
  }
  networking?: {
    deploymentMode?: string
    serverUrl?: string
  }
}

const withRuntimeRequestTimeout = async <T>(
  request: (signal: AbortSignal) => Promise<T>
): Promise<T> => {
  const controller = new AbortController()
  let timeoutId: ReturnType<typeof setTimeout> | undefined
  const timeout = new Promise<never>((_resolve, reject) => {
    timeoutId = setTimeout(() => {
      controller.abort()
      const error = new Error("Runtime bootstrap request timed out")
      error.name = "TimeoutError"
      reject(error)
    }, RUNTIME_REQUEST_TIMEOUT_MS)
  })

  try {
    return await Promise.race([request(controller.signal), timeout])
  } finally {
    if (timeoutId !== undefined) clearTimeout(timeoutId)
  }
}

const isRuntimeConfigFetchAllowed = (): boolean => {
  if (typeof window === "undefined" || typeof fetch !== "function") {
    return false
  }

  try {
    const protocol = String(window.location?.protocol || "").toLowerCase()
    return protocol === "http:" || protocol === "https:"
  } catch {
    return false
  }
}

const fetchRuntimeConfig = async (): Promise<RuntimeConfigPayload | null> => {
  if (!isRuntimeConfigFetchAllowed()) return null

  try {
    return await withRuntimeRequestTimeout(async (signal) => {
      const response = await fetch(RUNTIME_CONFIG_ENDPOINT, {
        credentials: "same-origin",
        cache: "no-store",
        headers: {
          "Cache-Control": "no-cache",
          Pragma: "no-cache"
        },
        signal
      })
      if (!response.ok) return null
      const payload = await response.json()
      return isRecord(payload) ? (payload as RuntimeConfigPayload) : null
    })
  } catch {
    return null
  }
}

const bootstrapCookieSession = async (): Promise<boolean> => {
  try {
    const response = await withRuntimeRequestTimeout((signal) =>
      fetch(RUNTIME_SESSION_ENDPOINT, {
        method: "POST",
        credentials: "include",
        cache: "no-store",
        signal
      })
    )
    if (!response.ok) return false

    const probe = await withRuntimeRequestTimeout((signal) =>
      fetch(RUNTIME_PROFILE_PROBE_ENDPOINT, {
        method: "GET",
        credentials: "same-origin",
        cache: "no-store",
        signal
      })
    )
    return probe.ok
  } catch {
    return false
  }
}

const removeLegacyRuntimeSecrets = (): void => {
  for (const key of LEGACY_RUNTIME_KEYS) {
    try {
      window.localStorage.removeItem(key)
    } catch {
      // Best-effort cleanup for upgraded profiles.
    }
  }
}

const normalizeHttpOrigin = (value: unknown): string | null => {
  try {
    const parsed = new URL(String(value || ""))
    return parsed.protocol === "http:" || parsed.protocol === "https:"
      ? parsed.origin
      : null
  } catch {
    return null
  }
}

const hasCompleteManualDeviceKey = (config: TldwConfig | null): boolean =>
  Boolean(
    config?.credentialSource === "manual" &&
      config.apiKeyPersistence === "device" &&
      typeof config.apiKey === "string" &&
      config.apiKey.trim() &&
      normalizeHttpOrigin(config.serverUrl) === config.apiKeyServerOrigin
  )

const scrubMismatchedManualSessionKey = (
  config: TldwConfig | null,
  quickstartServerUrl: string
): void => {
  try {
    const raw = window.sessionStorage.getItem(MANUAL_SESSION_API_KEY)
    if (!raw) return
    const record = JSON.parse(raw) as Record<string, unknown>
    const serverOrigin = normalizeHttpOrigin(config?.serverUrl)
    const quickstartOrigin = normalizeHttpOrigin(quickstartServerUrl)
    const complete =
      config?.authMode === "single-user" &&
      config.credentialSource === "manual" &&
      config.apiKeyPersistence === "session" &&
      serverOrigin !== null &&
      serverOrigin !== quickstartOrigin &&
      config.apiKeyServerOrigin === serverOrigin &&
      record.credentialSource === "manual" &&
      record.apiKeyPersistence === "session" &&
      typeof record.apiKey === "string" &&
      record.apiKey.trim() &&
      record.apiKeyServerOrigin === serverOrigin
    if (!complete) window.sessionStorage.removeItem(MANUAL_SESSION_API_KEY)
  } catch {
    try {
      window.sessionStorage.removeItem(MANUAL_SESSION_API_KEY)
    } catch {
      // Best-effort cleanup for upgraded profiles.
    }
  }
}

const seedTldwConfigFromRuntime = async (): Promise<void> => {
  if (typeof window === "undefined") return

  const storage = createSafeStorage()
  invalidateCookieSessionConfig()
  await storage.remove(COOKIE_SESSION_CONFIG_KEY).catch(() => undefined)

  const payload = await fetchRuntimeConfig()
  if (
    payload?.runtimeAuth?.available !== true ||
    payload.runtimeAuth.authMode !== "single-user" ||
    payload.runtimeAuth.transport !== "cookie-session"
  ) {
    return
  }

  if (!(await bootstrapCookieSession())) return

  try {
    const existing =
      (await storage.get<TldwConfig>("tldwConfig").catch(() => null)) || null
    const quickstartWebUiServerUrl = getQuickstartWebUiServerUrl()
    if (!quickstartWebUiServerUrl) return

    const next: TldwConfig = {
      authMode: "single-user",
      authSource: "cookie-session",
      serverUrl: quickstartWebUiServerUrl
    }

    if (existing?.apiKey && !hasCompleteManualDeviceKey(existing)) {
      const { apiKey: _ambiguousSecret, ...safeExisting } = existing
      await storage.set("tldwConfig", safeExisting)
    }
    await storage.set(COOKIE_SESSION_CONFIG_KEY, next)
    activateCookieSessionConfig()
    clearRuntimeAuth()
    clearRuntimeAuthOverride()
    removeLegacyRuntimeSecrets()
    scrubMismatchedManualSessionKey(existing, quickstartWebUiServerUrl)
  } catch {
    // Leave existing manual configuration intact if client storage is unavailable.
  }
}

const seedTldwConfigFromEnv = async (): Promise<void> => {
  if (typeof window === "undefined") return
  if (getQuickstartWebUiServerUrl()) return

  const explicitWebHost = (() => {
    try {
      return normalizeBaseUrl(window.localStorage.getItem("tldw-api-host"))
    } catch {
      return null
    }
  })()
  const repairedExplicitWebHostCandidate =
    deriveCurrentHostRecoveryServerUrl(explicitWebHost) || explicitWebHost
  const envDefaultServerUrl =
    normalizeBaseUrl(process.env.NEXT_PUBLIC_API_URL) || DEFAULT_TLDW_SERVER_URL
  const repairedEnvDefaultServerUrl =
    deriveCurrentHostRecoveryServerUrl(envDefaultServerUrl) ||
    envDefaultServerUrl
  const initialQuickstartWebUiServerUrl = getQuickstartWebUiServerUrl()
  const repairedExplicitWebHost =
    !initialQuickstartWebUiServerUrl &&
    isCurrentBrowserOrigin(repairedExplicitWebHostCandidate)
      ? null
      : repairedExplicitWebHostCandidate
  const initialServerUrl =
    initialQuickstartWebUiServerUrl ||
    repairedExplicitWebHost ||
    repairedEnvDefaultServerUrl
  const apiKey = (process.env.NEXT_PUBLIC_X_API_KEY || "").trim() || null
  const apiBearer = (process.env.NEXT_PUBLIC_API_BEARER || "").trim() || null

  if (initialServerUrl && explicitWebHost !== initialServerUrl) {
    try {
      window.localStorage.setItem("tldw-api-host", initialServerUrl)
    } catch {
      // Best-effort only; ignore storage failures in web contexts.
    }
  }

  try {
    const storage = createSafeStorage()
    const existing = (await storage.get<TldwConfig>("tldwConfig").catch(() => null)) || null
    const storedServerUrl =
      (await storage.get<string>("tldwServerUrl").catch(() => null)) || null
    const quickstartWebUiServerUrl = getQuickstartWebUiServerUrl()
    const effectiveExplicitWebHost =
      !quickstartWebUiServerUrl && isCurrentBrowserOrigin(repairedExplicitWebHost)
        ? null
        : repairedExplicitWebHost
    const serverUrl =
      quickstartWebUiServerUrl ||
      effectiveExplicitWebHost ||
      repairedEnvDefaultServerUrl

    if (!serverUrl && !apiKey && !apiBearer) return

    if (serverUrl && initialServerUrl !== serverUrl) {
      try {
        window.localStorage.setItem("tldw-api-host", serverUrl)
      } catch {
        // Best-effort only; ignore storage failures in web contexts.
      }
    }

    const next: TldwConfig = {
      ...(existing || {}),
      authMode: existing?.authMode || "single-user",
      serverUrl: existing?.serverUrl || ""
    }

    let changed = false
    let shouldSyncStoredServerUrl = false

    if (serverUrl && next.serverUrl !== serverUrl) {
      next.serverUrl = serverUrl
      changed = true
    }
    if (next.serverUrl && storedServerUrl !== next.serverUrl) {
      shouldSyncStoredServerUrl = true
    }

    if (!next.apiKey && !next.accessToken) {
      if (apiKey) {
        next.authMode = "single-user"
        next.apiKey = apiKey
        changed = true
      } else if (apiBearer) {
        next.authMode = "multi-user"
        next.accessToken = apiBearer
        changed = true
      }
    }

    if (changed || shouldSyncStoredServerUrl) {
      await storage.set("tldwConfig", next)
      if (next.serverUrl) {
        await storage.set("tldwServerUrl", next.serverUrl)
      }
    }
  } catch {
    // Best-effort only; ignore storage failures in web contexts.
  }
}

export const runtimeBootstrapReady = (async () => {
  await seedTldwConfigFromRuntime().catch(() => undefined)
  await seedTldwConfigFromEnv().catch(() => undefined)
})()

const WEB_DEFAULTS_MIRRORED_KEY = "tldw:web-defaults:mirrored"
const WEB_HEADER_SHORTCUT_DOC_WORKSPACE_BACKFILL_KEY =
  "tldw:web-defaults:header-shortcuts-document-workspace:v1"
const WEB_HEADER_SHORTCUT_MCP_HUB_BACKFILL_KEY =
  "tldw:web-defaults:header-shortcuts-mcp-hub:v1"

const isWebRuntime = () => {
  if (typeof window === "undefined") return false
  const protocol = window.location.protocol
  return protocol !== "chrome-extension:" && protocol !== "moz-extension:"
}

const writeLocalStorageValue = (key: string, value: unknown) => {
  if (typeof window === "undefined") return
  try {
    const serialized =
      typeof value === "string" ? value : JSON.stringify(value)
    window.localStorage.setItem(key, serialized)
  } catch {
    // ignore storage failures
  }
}

const removeLocalStorageValue = (key: string) => {
  if (typeof window === "undefined") return
  try {
    window.localStorage.removeItem(key)
  } catch {
    // ignore storage failures
  }
}

const getLocalStorageValue = (key: string) => {
  if (typeof window === "undefined") return null
  try {
    return window.localStorage.getItem(key)
  } catch {
    return null
  }
}

const setDefault = (key: string, value: unknown, force = false) => {
  const existing = getLocalStorageValue(key)
  if (!force && existing !== null) return
  writeLocalStorageValue(key, value)
}

const mirrorWebDefaultsFromExtension = () => {
  if (!isWebRuntime()) return
  const shouldMirrorDefaults =
    getLocalStorageValue(WEB_DEFAULTS_MIRRORED_KEY) !== "true"

  const legacyTheme = getLocalStorageValue("tldw-theme")
  if (getLocalStorageValue(THEME_SETTING.key) === null && legacyTheme) {
    writeLocalStorageValue(THEME_SETTING.key, legacyTheme)
  }
  if (legacyTheme !== null) {
    removeLocalStorageValue("tldw-theme")
  }

  if (!shouldMirrorDefaults) return

  // Theme + UI mode defaults
  setDefault(THEME_SETTING.key, THEME_SETTING.defaultValue)
  setDefault(UI_MODE_SETTING.key, UI_MODE_SETTING.defaultValue)
  setDefault("tldw-ui-mode", "casual")

  // Feature flags (default true, compare mode default false)
  Object.values(FEATURE_FLAGS).forEach((flag) => {
    const isCompareMode = flag === FEATURE_FLAGS.COMPARE_MODE
    setDefault(flag, isCompareMode ? false : true)
  })

  // Sidebar + header shortcuts defaults
  setDefault(
    SIDEBAR_ACTIVE_TAB_SETTING.key,
    SIDEBAR_ACTIVE_TAB_SETTING.defaultValue
  )
  setDefault(
    SIDEBAR_SHORTCUTS_COLLAPSED_SETTING.key,
    SIDEBAR_SHORTCUTS_COLLAPSED_SETTING.defaultValue
  )
  setDefault(
    SIDEBAR_SHORTCUT_SELECTION_SETTING.key,
    DEFAULT_SIDEBAR_SHORTCUT_SELECTION
  )
  setDefault(
    HEADER_SHORTCUT_SELECTION_SETTING.key,
    DEFAULT_HEADER_SHORTCUT_SELECTION
  )
  setDefault(
    HEADER_SHORTCUTS_EXPANDED_SETTING.key,
    HEADER_SHORTCUTS_EXPANDED_SETTING.defaultValue
  )

  writeLocalStorageValue(WEB_DEFAULTS_MIRRORED_KEY, "true")
}

const backfillHeaderShortcutForWeb = (
  shortcutId: string,
  markerKey: string
) => {
  if (!isWebRuntime()) return
  if (getLocalStorageValue(markerKey) === "true") {
    return
  }

  const rawSelection = getLocalStorageValue(HEADER_SHORTCUT_SELECTION_SETTING.key)
  if (rawSelection === null) {
    writeLocalStorageValue(markerKey, "true")
    return
  }

  let parsedSelection: unknown = null
  try {
    parsedSelection = JSON.parse(rawSelection)
  } catch {
    writeLocalStorageValue(markerKey, "true")
    return
  }

  if (!Array.isArray(parsedSelection)) {
    writeLocalStorageValue(markerKey, "true")
    return
  }

  const selectedIds = new Set(
    parsedSelection.filter(
      (entry): entry is string => typeof entry === "string"
    )
  )
  selectedIds.add(shortcutId)

  const nextSelection = DEFAULT_HEADER_SHORTCUT_SELECTION.filter((id) =>
    selectedIds.has(id)
  )
  writeLocalStorageValue(HEADER_SHORTCUT_SELECTION_SETTING.key, nextSelection)
  writeLocalStorageValue(markerKey, "true")
}

mirrorWebDefaultsFromExtension()
backfillHeaderShortcutForWeb(
  "document-workspace",
  WEB_HEADER_SHORTCUT_DOC_WORKSPACE_BACKFILL_KEY
)
backfillHeaderShortcutForWeb("mcp-hub", WEB_HEADER_SHORTCUT_MCP_HUB_BACKFILL_KEY)
