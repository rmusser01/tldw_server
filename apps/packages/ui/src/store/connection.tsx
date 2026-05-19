import { createWithEqualityFn } from "zustand/traditional"

import { tldwClient, type TldwConfig } from "@/services/tldw/TldwApiClient"
import { getStoredTldwServerURL } from "@/services/tldw-server"
import { apiSend } from "@/services/api-send"
import { createSafeStorage } from "@/utils/safe-storage"
import {
  resolveWebUiQuickstartServerUrl,
  type BrowserSurface
} from "@/services/tldw/browser-networking"
import { resolveBrowserRequestTransport } from "@/services/tldw/request-core"
import {
  ConnectionPhase,
  type ConnectionState,
  type KnowledgeStatus,
  type UserPersona,
  deriveConnectionUxState
} from "@/types/connection"
import { CONNECTED_THROTTLE_MS } from "@/config/connection-timing"

// Shared timeout before treating the server as unreachable.
// See New-Views-PRD.md §5.1.x / §10.1 (20 seconds).
export const CONNECTION_TIMEOUT_MS = 20_000
const HEALTH_LIVENESS_PATH = "/api/v1/health/live"
const CONNECTED_FAILURE_THRESHOLD = 3
const KNOWLEDGE_RECHECK_INTERVAL_MS = 5 * 60_000

const TEST_BYPASS_KEY = "__tldw_allow_offline"
const FORCE_UNCONFIGURED_KEY = "__tldw_force_unconfigured"
const FIRST_RUN_COMPLETE_KEY = "__tldw_first_run_complete"
const USER_PERSONA_KEY = "__tldw_user_persona"

const coerceStorageFlag = (value: unknown): boolean | null => {
  if (typeof value === "boolean") return value
  if (typeof value === "string") return value === "true"
  return null
}

const readLocalStorageFlag = (key: string): boolean | null => {
  try {
    if (typeof localStorage !== "undefined") {
      const raw = localStorage.getItem(key)
      if (raw != null) {
        return raw === "true"
      }
    }
  } catch {
    // ignore localStorage availability
  }

  return null
}

const getStorageFlag = async (key: string): Promise<boolean> => {
  try {
    if (typeof chrome !== "undefined" && chrome?.storage?.local) {
      return await new Promise<boolean>((resolve) => {
        chrome.storage.local.get(key, (res) => {
          if (res && Object.prototype.hasOwnProperty.call(res, key)) {
            resolve(coerceStorageFlag(res[key]) ?? false)
            return
          }

          resolve(readLocalStorageFlag(key) ?? false)
        })
      })
    }
  } catch {
    // ignore storage read errors
  }

  return readLocalStorageFlag(key) ?? false
}

const getOfflineBypassFlag = async (): Promise<boolean> => {
  // Build-time flag for Playwright/CI: VITE_TLDW_E2E_ALLOW_OFFLINE=true
  const env = import.meta.env as
    | { VITE_TLDW_E2E_ALLOW_OFFLINE?: string }
    | undefined
  if (env?.VITE_TLDW_E2E_ALLOW_OFFLINE === "true") {
    return true
  }

  // Runtime toggle (settable by tests) via chrome.storage.local or localStorage.
  return getStorageFlag(TEST_BYPASS_KEY)
}

const setOfflineBypassFlag = async (enabled: boolean): Promise<void> => {
  try {
    if (typeof chrome !== "undefined" && chrome?.storage?.local) {
      await new Promise<void>((resolve) => {
        const storage = chrome.storage.local
        if (enabled) {
          storage.set({ [TEST_BYPASS_KEY]: true }, () => resolve())
        } else {
          storage.remove(TEST_BYPASS_KEY, () => resolve())
        }
      })
      return
    }
  } catch {
    // ignore storage write errors
  }

  try {
    if (typeof localStorage !== "undefined") {
      if (enabled) {
        localStorage.setItem(TEST_BYPASS_KEY, "true")
      } else {
        localStorage.removeItem(TEST_BYPASS_KEY)
      }
    }
  } catch {
    // ignore localStorage availability
  }
}

const getForceUnconfiguredFlag = async (): Promise<boolean> => {
  return getStorageFlag(FORCE_UNCONFIGURED_KEY)
}

const getFirstRunCompleteFlag = async (): Promise<boolean> => {
  return getStorageFlag(FIRST_RUN_COMPLETE_KEY)
}

const setFirstRunCompleteFlag = async (complete: boolean): Promise<void> => {
  try {
    if (typeof chrome !== "undefined" && chrome?.storage?.local) {
      await new Promise<void>((resolve) => {
        const storage = chrome.storage.local
        if (complete) {
          storage.set({ [FIRST_RUN_COMPLETE_KEY]: true }, () => resolve())
        } else {
          storage.remove(FIRST_RUN_COMPLETE_KEY, () => resolve())
        }
      })
      return
    }
  } catch {
    // ignore storage write errors
  }

  try {
    if (typeof localStorage !== "undefined") {
      if (complete) {
        localStorage.setItem(FIRST_RUN_COMPLETE_KEY, "true")
      } else {
        localStorage.removeItem(FIRST_RUN_COMPLETE_KEY)
      }
    }
  } catch {
    // ignore localStorage availability
  }
}

const VALID_PERSONAS = new Set<string>(["family", "researcher", "explorer"])

const readLocalStoragePersona = (key: string): UserPersona => {
  try {
    if (typeof localStorage !== "undefined") {
      const raw = localStorage.getItem(key)
      if (raw != null && VALID_PERSONAS.has(raw)) {
        return raw as UserPersona
      }
    }
  } catch {
    // ignore localStorage availability
  }

  return null
}

/** Read a single key from chrome.storage.local (extension context only). */
const chromeStorageGet = (key: string): Promise<string | undefined> =>
  new Promise((resolve, reject) => {
    chrome.storage.local.get(key, (res) => {
      if (chrome.runtime.lastError) {
        reject(chrome.runtime.lastError)
        return
      }
      resolve(
        res && Object.prototype.hasOwnProperty.call(res, key)
          ? (res[key] as string)
          : undefined
      )
    })
  })

const getUserPersonaFlag = async (): Promise<UserPersona> => {
  try {
    if (typeof chrome !== "undefined" && chrome?.storage?.local) {
      const value = await chromeStorageGet(USER_PERSONA_KEY)
      if (typeof value === "string" && VALID_PERSONAS.has(value)) {
        return value as UserPersona
      }
      return readLocalStoragePersona(USER_PERSONA_KEY)
    }
  } catch {
    // ignore storage read errors
  }

  return readLocalStoragePersona(USER_PERSONA_KEY)
}

const setUserPersonaFlag = async (persona: UserPersona): Promise<void> => {
  try {
    if (typeof chrome !== "undefined" && chrome?.storage?.local) {
      await new Promise<void>((resolve, reject) => {
        const storage = chrome.storage.local
        const cb = () => {
          if (chrome.runtime.lastError) { reject(chrome.runtime.lastError); return }
          resolve()
        }
        if (persona) {
          storage.set({ [USER_PERSONA_KEY]: persona }, cb)
        } else {
          storage.remove(USER_PERSONA_KEY, cb)
        }
      })
      return
    }
  } catch {
    // ignore storage write errors
  }

  try {
    if (typeof localStorage !== "undefined") {
      if (persona) {
        localStorage.setItem(USER_PERSONA_KEY, persona)
      } else {
        localStorage.removeItem(USER_PERSONA_KEY)
      }
    }
  } catch {
    // ignore localStorage availability
  }
}

const ensurePlaceholderConfig = async (): Promise<string | null> => {
  try {
    const cfg = await tldwClient.getConfig()
    if (cfg?.serverUrl) return cfg.serverUrl
  } catch {
    // ignore missing config
  }

  const placeholderUrl = "http://127.0.0.1:0"
  try {
    await tldwClient.updateConfig({
      serverUrl: placeholderUrl,
      authMode: "single-user",
      apiKey: "test-bypass"
    })
    return placeholderUrl
  } catch {
    return null
  }
}

const deriveKnowledgeStatusFromHealth = (raw: unknown): KnowledgeStatus => {
  try {
    if (!raw || typeof raw !== "object") {
      return "ready"
    }
    const obj = raw as Record<string, unknown>
    const components = obj.components
    if (components && typeof components === "object") {
      const comp = components as Record<string, unknown>
      const search = comp.search_index || comp.searchIndex
      if (search && typeof search === "object") {
        const s = search as Record<string, unknown>
        const status = String(s.status || "").toLowerCase()
        const message = String(s.message || "")
        const rawCount = s.fts_table_count
        const ftsCount =
          typeof rawCount === "number" && Number.isFinite(rawCount)
            ? rawCount
            : null

        const noIndexByCount = ftsCount !== null && ftsCount <= 0
        const noIndexByMessage = /no fts indexes found/i.test(message)

        if ((noIndexByCount || noIndexByMessage) && status !== "unhealthy") {
          return "empty"
        }
      }
    }
  } catch {
    // ignore parse errors and fall back to ready
  }
  return "ready"
}

const getNormalizedOrigin = (value: string | null | undefined): string | null => {
  if (!value) return null
  try {
    return new URL(String(value)).origin
  } catch {
    return null
  }
}

const getCurrentBrowserOrigin = (): string | null => {
  if (typeof window === "undefined") return null
  try {
    return window.location?.origin ?? null
  } catch {
    return null
  }
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
    const hostname = window.location?.hostname
    if (!hostname) return null
    return String(hostname).trim().toLowerCase() || null
  } catch {
    return null
  }
}

const parsePrivateIpv4Host = (value: string | null | undefined): number[] | null => {
  if (!value) return null
  const normalized = String(value).trim().toLowerCase()
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

const deriveCurrentHostRecoveryServerUrl = (
  configuredServerUrl: string | null | undefined
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
    if (!configuredPrivateIp || !browserPrivateIp) return null
    const port = parsed.port || "8000"
    return `${parsed.protocol}//${browserHostname}:${port}`
  } catch {
    return null
  }
}

const isNetworkTransportFailure = (value: string | null | undefined): boolean => {
  if (!value) return false
  const text = String(value)
  return NETWORK_BLOCK_PATTERNS.some((pattern) => pattern.test(text))
}

const probeServerLiveness = async (
  serverUrl: string,
  timeoutMs: number
): Promise<boolean> => {
  if (!serverUrl || typeof fetch === "undefined") return false
  const safeTimeoutMs = Number(timeoutMs) > 0 ? Number(timeoutMs) : 3000
  const controller = typeof AbortController !== "undefined" ? new AbortController() : null
  let timeoutId: ReturnType<typeof setTimeout> | null = null
  try {
    if (controller) {
      timeoutId = setTimeout(() => controller.abort(), safeTimeoutMs)
    }
    const response = await fetch(resolveBrowserRequestTransport({
      config: { serverUrl },
      path: HEALTH_LIVENESS_PATH
    }).url, {
      method: "GET",
      credentials: "omit",
      signal: controller?.signal
    })
    return Boolean(response?.ok)
  } catch {
    return false
  } finally {
    if (timeoutId) {
      clearTimeout(timeoutId)
    }
  }
}

const CORS_ERROR_PATTERNS = [
  /cors/i,
  /cross-origin/i,
  /disallowed origin/i
]

const NETWORK_BLOCK_PATTERNS = [
  /networkerror when attempting to fetch resource/i,
  /failed to fetch/i,
  /network request failed/i,
  /load failed/i,
  /the operation was aborted/i
]

const maybeAnnotateCorsMismatchError = ({
  error,
  status,
  serverUrl
}: {
  error: string | null
  status: number
  serverUrl: string | null
}): string | null => {
  if (!error) return error
  const trimmed = String(error).trim()
  if (!trimmed) return error
  const normalized = trimmed.toLowerCase()
  if (normalized.startsWith("likely cors mismatch:")) {
    return trimmed
  }

  const mentionsCors = CORS_ERROR_PATTERNS.some((pattern) =>
    pattern.test(trimmed)
  )
  const looksLikeNetworkBlock = NETWORK_BLOCK_PATTERNS.some((pattern) =>
    pattern.test(trimmed)
  )
  if (!mentionsCors && !looksLikeNetworkBlock) {
    return trimmed
  }

  const browserOrigin = getCurrentBrowserOrigin()
  const backendOrigin = getNormalizedOrigin(serverUrl)
  if (browserOrigin && backendOrigin && browserOrigin === backendOrigin && !mentionsCors) {
    return trimmed
  }

  if (status > 0 && status < 400 && !mentionsCors) {
    return trimmed
  }

  const browserLabel = browserOrigin || "current browser origin"
  const backendLabel = backendOrigin || (serverUrl ? String(serverUrl) : "configured server")
  return (
    `Likely CORS mismatch: ${browserLabel} is not allowed by ${backendLabel}. ` +
    `Set ALLOWED_ORIGINS to include ${browserLabel} (or disable CORS for local development). ` +
    `Original error: ${trimmed}`
  )
}

type ConnectionStore = {
  state: ConnectionState
  checkOnce: () => Promise<void>
  setServerUrl: (url: string) => Promise<void>
  enableOfflineBypass: () => Promise<void>
  disableOfflineBypass: () => Promise<void>
  beginOnboarding: () => Promise<void>
  restartOnboarding: () => Promise<void>
  setConfigPartial: (config: Partial<TldwConfig>) => Promise<void>
  testConnectionFromOnboarding: () => Promise<void>
  setDemoMode: () => void
  markFirstRunComplete: () => Promise<void>
  setUserPersona: (persona: UserPersona) => Promise<void>
}

const initialState: ConnectionState = {
  phase: ConnectionPhase.SEARCHING,
  serverUrl: null,
  lastCheckedAt: null,
  lastError: null,
  lastStatusCode: null,
  isConnected: false,
  isChecking: false,
  consecutiveFailures: 0,
  offlineBypass: false,
  knowledgeStatus: "unknown",
  knowledgeLastCheckedAt: null,
  knowledgeError: null,
  mode: "normal",
  configStep: "none",
  errorKind: "none",
  hasCompletedFirstRun: false,
  userPersona: null,
  lastConfigUpdatedAt: null,
  checksSinceConfigChange: 0
}

const getPersistedServerUrl = async (): Promise<string | null> => {
  try {
    const cfg = await tldwClient.getConfig()
    const quickstartWebUiServerUrl = getQuickstartWebUiServerUrl()
    if (quickstartWebUiServerUrl) {
      return quickstartWebUiServerUrl
    }
    if (cfg?.serverUrl) return cfg.serverUrl
  } catch {
    // ignore config read errors
  }

  try {
    const storage = createSafeStorage()
    const cfg = await storage.get<TldwConfig>("tldwConfig")
    const quickstartWebUiServerUrl = getQuickstartWebUiServerUrl()
    if (quickstartWebUiServerUrl) {
      return quickstartWebUiServerUrl
    }
    if (cfg?.serverUrl) return cfg.serverUrl
  } catch {
    // ignore storage read errors
  }

  return null
}

const hasRequiredAuthForConfig = (config: Partial<TldwConfig> | null | undefined): boolean => {
  const authMode = config?.authMode ?? "single-user"
  if (authMode === "multi-user") {
    return Boolean(String(config?.accessToken || "").trim())
  }

  return Boolean(String(config?.apiKey || "").trim())
}

const deriveOnboardingConfigStep = (
  config: Partial<TldwConfig> | null | undefined,
  fallbackServerUrl: string | null,
  currentState: ConnectionState
): ConnectionState["configStep"] => {
  if (!config) {
    if (
      currentState.phase === ConnectionPhase.CONNECTED ||
      currentState.errorKind === "partial"
    ) {
      return currentState.configStep === "none" ? "health" : currentState.configStep
    }
    if (currentState.errorKind === "auth") {
      return "auth"
    }
  }

  const serverUrl = String(config?.serverUrl || fallbackServerUrl || "").trim()
  if (!serverUrl) {
    return "url"
  }

  if (!hasRequiredAuthForConfig(config)) {
    return "auth"
  }

  return currentState.configStep === "none" ? "health" : currentState.configStep
}

export const useConnectionStore = createWithEqualityFn<ConnectionStore>((set, get) => ({
  state: initialState,

  async checkOnce() {
    const prev = get().state

    // Avoid overlapping checks
    if (prev.isChecking) {
      return
    }

    // Load all persisted flags upfront
    const persistedFirstRun = await getFirstRunCompleteFlag()
    const persistedUserPersona = await getUserPersonaFlag()
    const persistedServerUrl = await getPersistedServerUrl()
    const forceUnconfigured = await getForceUnconfiguredFlag()
    const bypass = await getOfflineBypassFlag()

    const needsFirstRunSync = !prev.hasCompletedFirstRun && persistedFirstRun
    const needsPersonaSync = prev.userPersona !== persistedUserPersona

    const currentState =
      needsFirstRunSync || needsPersonaSync
        ? {
            ...prev,
            ...(needsFirstRunSync ? { hasCompletedFirstRun: true } : {}),
            ...(needsPersonaSync ? { userPersona: persistedUserPersona } : {})
          }
        : prev

    // Apply persisted first-run flag if not already set
    if (currentState !== prev) {
      set({
        state: currentState
      })
    }

    // Test-only hook: force a missing/unconfigured state without network calls.
    if (forceUnconfigured) {
      set({
        state: {
          ...currentState,
          errorKind: "none",
          phase: ConnectionPhase.UNCONFIGURED,
          serverUrl: persistedServerUrl,
          isConnected: false,
          isChecking: false,
          consecutiveFailures: 0,
          offlineBypass: false,
          lastCheckedAt: Date.now(),
          lastError: null,
          lastStatusCode: null,
          knowledgeStatus: "unknown",
          knowledgeLastCheckedAt: null,
          knowledgeError: null
        }
      })
      return
    }

    // Optional test toggle: allow CI/Playwright to treat the app as "connected"
    // without hitting a live server. Controlled via env VITE_TLDW_E2E_ALLOW_OFFLINE
    // or chrome.storage.local[__tldw_allow_offline].
    if (bypass) {
      const serverUrl =
        persistedServerUrl ??
        (await ensurePlaceholderConfig()) ??
        currentState.serverUrl ??
        "offline://local"

      set({
        state: {
          ...currentState,
          phase: ConnectionPhase.CONNECTED,
          serverUrl,
          isConnected: true,
          isChecking: false,
          consecutiveFailures: 0,
          offlineBypass: true,
          errorKind: "none",
          lastCheckedAt: Date.now(),
          lastError: null,
          lastStatusCode: null,
          knowledgeStatus: "ready",
          knowledgeLastCheckedAt: Date.now(),
          knowledgeError: null
        }
      })
      return
    }

    // Throttle repeated checks when already connected recently.
    // This prevents the landing page/header from hammering the server.
    const now = Date.now()
    const nextChecksSinceConfigChange = currentState.checksSinceConfigChange + 1
    if (
      currentState.isConnected &&
      currentState.phase === ConnectionPhase.CONNECTED &&
      currentState.lastCheckedAt != null &&
      now - currentState.lastCheckedAt < CONNECTED_THROTTLE_MS
    ) {
      return
    }

    const isBackgroundRefresh =
      currentState.isConnected && currentState.phase === ConnectionPhase.CONNECTED
    set({
      state: {
        ...currentState,
        phase: isBackgroundRefresh
          ? ConnectionPhase.CONNECTED
          : ConnectionPhase.SEARCHING,
        serverUrl: persistedServerUrl ?? currentState.serverUrl,
        errorKind: isBackgroundRefresh ? currentState.errorKind : "none",
        isChecking: true,
        offlineBypass: false,
        lastError: isBackgroundRefresh ? currentState.lastError : null,
        checksSinceConfigChange: nextChecksSinceConfigChange
      }
    })

    try {
      let cfg = await tldwClient.getConfig()
      const quickstartWebUiServerUrl = getQuickstartWebUiServerUrl()
      const recoveryProbeSourceServerUrl = cfg?.serverUrl ?? currentState.serverUrl ?? null
      let serverUrl = quickstartWebUiServerUrl ?? cfg?.serverUrl ?? null

      if (
        quickstartWebUiServerUrl &&
        cfg?.serverUrl !== quickstartWebUiServerUrl
      ) {
        await tldwClient.updateConfig({ serverUrl: quickstartWebUiServerUrl })
        cfg = {
          ...(cfg || {}),
          serverUrl: quickstartWebUiServerUrl,
          authMode: cfg?.authMode || "single-user"
        } as TldwConfig
        serverUrl = quickstartWebUiServerUrl
      }

      if (!serverUrl) {
        try {
          // Only reuse a previously stored URL; do not implicitly
          // fall back to the hard-coded localhost default here.
          const storedUrl = await getStoredTldwServerURL()
          if (storedUrl) {
            await tldwClient.updateConfig({
              serverUrl: storedUrl
            })
            cfg = await tldwClient.getConfig()
            serverUrl = cfg?.serverUrl ?? storedUrl
          }
        } catch {
          // ignore fallback errors; we will treat as unconfigured below
        }
      }

      const missingSingleUserApiKey =
        Boolean(serverUrl) &&
        (cfg?.authMode ?? "single-user") === "single-user" &&
        !String(cfg?.apiKey || "").trim()

      // If we have a server URL but no single-user API key, treat as
      // unconfigured/unauthenticated instead of marking the app connected
      // off an unauthenticated liveness check.
      // Users must explicitly configure their own credentials in
      // Settings/Onboarding before authenticated pages can function.
      if (missingSingleUserApiKey) {
        set({
          state: {
            ...currentState,
            phase: ConnectionPhase.UNCONFIGURED,
            serverUrl,
            isConnected: false,
            isChecking: false,
            consecutiveFailures: 0,
            offlineBypass: false,
            errorKind: "none",
            configStep: "auth",
            lastCheckedAt: Date.now(),
            lastError: null,
            lastStatusCode: null,
            knowledgeStatus: "unknown",
            knowledgeLastCheckedAt: null,
            knowledgeError: null
          }
        })
        return
      }

      if (!serverUrl) {
        set({
          state: {
            ...currentState,
            phase: ConnectionPhase.UNCONFIGURED,
            serverUrl: null,
            isConnected: false,
            isChecking: false,
            consecutiveFailures: 0,
            offlineBypass: false,
            errorKind: "none",
            lastCheckedAt: Date.now(),
            lastError: null,
            lastStatusCode: null,
            knowledgeStatus: "unknown",
            knowledgeLastCheckedAt: null,
            knowledgeError: null
          }
        })
        return
      }

      await tldwClient.initialize()

      // Request health via background for detailed status codes.
      // Health endpoints may require auth; apiSend injects headers based
      // on tldwConfig (API key / access token).
      const noAuthForHealth = !cfg ||
        (!cfg.apiKey &&
          !cfg.accessToken &&
          cfg.authMode !== "multi-user")

      const healthPromise = (async () => {
        try {
          const resp = await apiSend({
            path: HEALTH_LIVENESS_PATH,
            method: 'GET',
            timeoutMs: CONNECTION_TIMEOUT_MS,
            // Allow unauthenticated health checks when no credentials have
            // been configured yet so first‑run onboarding can still detect a
            // reachable server URL. Once an API key or access token exists,
            // health should run with auth.
            noAuth: noAuthForHealth
          })
          return { ok: Boolean(resp?.ok), status: Number(resp?.status) || 0, error: resp?.ok ? null : (resp?.error || null) }
        } catch (e) {
          return { ok: false, status: 0, error: (e as Error)?.message || 'Network error' }
        }
      })()
      let healthResult = await Promise.race([
        healthPromise,
        new Promise<{ ok: boolean; status: number; error: string | null }>((resolve) =>
          setTimeout(() => resolve({ ok: false, status: 0, error: 'timeout' }), CONNECTION_TIMEOUT_MS)
        )
      ])

      const fallbackServerUrl = deriveCurrentHostRecoveryServerUrl(
        quickstartWebUiServerUrl ? recoveryProbeSourceServerUrl : serverUrl
      )
      if (
        !healthResult.ok &&
        healthResult.status === 0 &&
        isNetworkTransportFailure(healthResult.error) &&
        fallbackServerUrl
      ) {
        const probeOk = await probeServerLiveness(
          fallbackServerUrl,
          Math.min(5_000, CONNECTION_TIMEOUT_MS)
        )
        if (probeOk) {
          if (!quickstartWebUiServerUrl) {
            await tldwClient.updateConfig({ serverUrl: fallbackServerUrl })
            serverUrl = fallbackServerUrl
            cfg = {
              ...(cfg || {}),
              serverUrl: fallbackServerUrl
            } as TldwConfig
          }
          const fallbackNoAuth = !cfg ||
            (!cfg.apiKey &&
              !cfg.accessToken &&
              cfg.authMode !== "multi-user")
          const fallbackResp = await apiSend({
            path: HEALTH_LIVENESS_PATH,
            method: "GET",
            timeoutMs: CONNECTION_TIMEOUT_MS,
            noAuth: fallbackNoAuth
          })
          healthResult = {
            ok: Boolean(fallbackResp?.ok),
            status: Number(fallbackResp?.status) || 0,
            error: fallbackResp?.ok ? null : (fallbackResp?.error || null)
          }
        }
      }

      const ok = healthResult.ok
      const resolvedHealthError = maybeAnnotateCorsMismatchError({
        error: healthResult.error,
        status: healthResult.status,
        serverUrl
      })

      let knowledgeStatus: KnowledgeStatus = currentState.knowledgeStatus
      let knowledgeLastCheckedAt = currentState.knowledgeLastCheckedAt
      let knowledgeError = currentState.knowledgeError
      const shouldRefreshKnowledge =
        !currentState.knowledgeLastCheckedAt ||
        now - currentState.knowledgeLastCheckedAt >= KNOWLEDGE_RECHECK_INTERVAL_MS ||
        currentState.knowledgeStatus !== "ready"

      if (ok && shouldRefreshKnowledge) {
        try {
          // Add timeout to RAG health check to prevent hanging
          // Increased from 5s to 15s to avoid false "offline" status when RAG is slow but working
          const ragPromise = tldwClient.ragHealth()
          const ragTimeout = new Promise<null>((resolve) =>
            setTimeout(() => resolve(null), 15000)
          )
          const rag = await Promise.race([ragPromise, ragTimeout])
          if (rag !== null) {
            knowledgeStatus = deriveKnowledgeStatusFromHealth(rag)
          } else {
            knowledgeStatus = "offline"
            knowledgeError = "rag-timeout"
          }
          knowledgeLastCheckedAt = Date.now()
          if (knowledgeStatus === "empty") {
            knowledgeError = "no-index"
          }
        } catch (e) {
          knowledgeStatus = "offline"
          knowledgeLastCheckedAt = Date.now()
          knowledgeError = (e as Error)?.message ?? "unknown-error"
        }
      } else if (!ok) {
        knowledgeStatus = "offline"
        knowledgeLastCheckedAt = Date.now()
        knowledgeError = "core-offline"
      }

      let errorKind: ConnectionState["errorKind"] = "none"
      const nextConsecutiveFailures = ok ? 0 : currentState.consecutiveFailures + 1

      if (ok) {
        if (knowledgeStatus === "offline") {
          errorKind = "partial"
        } else {
          errorKind = "none"
        }
      } else {
        const status = healthResult.status
        if (status === 401 || status === 403) {
          errorKind = "auth"
        } else {
          errorKind = "unreachable"
        }
      }

      const holdConnectedOnTransientFailure =
        !ok &&
        errorKind === "unreachable" &&
        currentState.isConnected &&
        currentState.phase === ConnectionPhase.CONNECTED &&
        nextConsecutiveFailures < CONNECTED_FAILURE_THRESHOLD

      if (holdConnectedOnTransientFailure) {
        set({
          state: {
            ...currentState,
            phase: ConnectionPhase.CONNECTED,
            isConnected: true,
            isChecking: false,
            offlineBypass: false,
            lastCheckedAt: Date.now(),
            lastError: resolvedHealthError || "transient-health-check-failure",
            lastStatusCode: healthResult.status || 0,
            errorKind: "partial",
            consecutiveFailures: nextConsecutiveFailures
          }
        })
        return
      }

      set({
        state: {
          ...currentState,
          phase: ok ? ConnectionPhase.CONNECTED : ConnectionPhase.ERROR,
          serverUrl,
          isConnected: ok,
          isChecking: false,
          consecutiveFailures:
            ok
              ? 0
              : errorKind === "unreachable"
                ? nextConsecutiveFailures
                : 0,
          offlineBypass: false,
          lastCheckedAt: Date.now(),
          lastError: ok ? null : (resolvedHealthError || 'timeout-or-offline'),
          lastStatusCode: ok ? null : healthResult.status,
          knowledgeStatus,
          knowledgeLastCheckedAt,
          knowledgeError,
          errorKind,
          checksSinceConfigChange: nextChecksSinceConfigChange
        }
      })
    } catch (error) {
      const fallbackError =
        maybeAnnotateCorsMismatchError({
          error: (error as Error)?.message ?? "unknown-error",
          status: 0,
          serverUrl: currentState.serverUrl
        }) ?? "unknown-error"
      set({
        state: {
          ...currentState,
          phase: ConnectionPhase.ERROR,
          isConnected: false,
          isChecking: false,
          consecutiveFailures: currentState.consecutiveFailures + 1,
          offlineBypass: false,
          lastCheckedAt: Date.now(),
          lastError: fallbackError,
          lastStatusCode: 0,
          knowledgeStatus: "offline",
          knowledgeLastCheckedAt: Date.now(),
          knowledgeError: fallbackError,
          errorKind: "unreachable",
          checksSinceConfigChange: nextChecksSinceConfigChange
        }
      })
    }
  },

  async setServerUrl(url: string) {
    await tldwClient.updateConfig({ serverUrl: url })
    await get().checkOnce()
  },

  async enableOfflineBypass() {
    await setOfflineBypassFlag(true)
    await get().checkOnce()
  },

  async disableOfflineBypass() {
    await setOfflineBypassFlag(false)
    await get().checkOnce()
  },

  async beginOnboarding() {
    const prev = get().state
    const isSyntheticConnectedState =
      prev.mode === "demo" || prev.offlineBypass === true
    let config: TldwConfig | null = null
    try {
      config = await tldwClient.getConfig()
    } catch {
      // ignore config lookup failures and fall back to current state
    }

    const nextServerUrl =
      String(config?.serverUrl || prev.serverUrl || "").trim() || null
    const nextConfigStep = deriveOnboardingConfigStep(
      config,
      nextServerUrl,
      isSyntheticConnectedState
        ? {
            ...prev,
            phase: ConnectionPhase.UNCONFIGURED,
            isConnected: false,
            configStep: "none",
            errorKind: "none",
            mode: "normal",
            offlineBypass: false
          }
        : prev
    )

    set({
      state: {
        ...prev,
        ...(isSyntheticConnectedState
          ? {
              phase: ConnectionPhase.UNCONFIGURED,
              isConnected: false,
              consecutiveFailures: 0,
              errorKind: "none" as const,
              lastError: null,
              lastStatusCode: null,
              knowledgeStatus: "unknown" as const,
              knowledgeLastCheckedAt: null,
              knowledgeError: null
            }
          : {}),
        serverUrl: nextServerUrl,
        configStep: nextConfigStep,
        mode: "normal",
        isChecking: false,
        offlineBypass: false
      }
    })
  },

  async restartOnboarding() {
    const prev = get().state
    await setFirstRunCompleteFlag(false)
    set({
      state: {
        ...prev,
        phase: ConnectionPhase.UNCONFIGURED,
        configStep: "url",
        hasCompletedFirstRun: false,
        mode: "normal",
        isConnected: false,
        isChecking: false,
        consecutiveFailures: 0,
        offlineBypass: false,
        errorKind: "none",
        lastError: null,
        lastStatusCode: null,
        knowledgeStatus: "unknown",
        knowledgeLastCheckedAt: null,
        knowledgeError: null
      }
    })
  },

  async setConfigPartial(config: Partial<TldwConfig>) {
    await tldwClient.updateConfig(config)
    const prev = get().state

    let nextStep: ConnectionState["configStep"] = prev.configStep
    const now = Date.now()

    if (typeof config.serverUrl === "string" && config.serverUrl.trim()) {
      nextStep = "auth"
    }

    if (
      typeof config.authMode !== "undefined" ||
      typeof config.apiKey !== "undefined" ||
      typeof config.accessToken !== "undefined"
    ) {
      nextStep = "auth"
    }

    set({
      state: {
        ...prev,
        serverUrl:
          typeof config.serverUrl === "string"
            ? config.serverUrl
            : prev.serverUrl,
        configStep: nextStep,
        consecutiveFailures: 0,
        lastConfigUpdatedAt: now,
        checksSinceConfigChange: 0
      }
    })
  },

  async testConnectionFromOnboarding() {
    const prev = get().state
    const isTestBypass =
      prev.offlineBypass || (await getOfflineBypassFlag())

    // When offline bypass is enabled (Playwright/CI path), treat the
    // connection as healthy immediately so onboarding can progress
    // without waiting on real network checks. In normal production
    // runs offlineBypass is false and we fall back to checkOnce().
    if (isTestBypass) {
      set({
        state: {
          ...prev,
          configStep: "health",
          phase: ConnectionPhase.CONNECTED,
          isConnected: true,
          isChecking: false,
          consecutiveFailures: 0,
          offlineBypass: true,
          errorKind: "none",
          lastError: null,
          lastStatusCode: null,
          knowledgeStatus: "ready",
          knowledgeLastCheckedAt: Date.now(),
          knowledgeError: null
        }
      })
      return
    }

    set({
      state: {
        ...prev,
        configStep: "health"
      }
    })
    await get().checkOnce()
  },

  setDemoMode() {
    const prev = get().state
    set({
      state: {
        ...prev,
        mode: "demo",
        phase: ConnectionPhase.CONNECTED,
        isConnected: true,
        consecutiveFailures: 0,
        offlineBypass: false,
        errorKind: "none",
        lastError: null,
        lastStatusCode: null,
        hasCompletedFirstRun: true
      }
    })
  },

  async markFirstRunComplete() {
    const prev = get().state
    if (prev.hasCompletedFirstRun) {
      return
    }
    // Persist to chrome.storage so it survives browser data clears
    await setFirstRunCompleteFlag(true)
    set({
      state: {
        ...prev,
        hasCompletedFirstRun: true
      }
    })
  },

  async setUserPersona(persona: UserPersona) {
    await setUserPersonaFlag(persona)
    set({
      state: {
        ...get().state,
        userPersona: persona
      }
    })
  }
}))

if (typeof window !== "undefined") {
  // Expose for Playwright tests and debugging only.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(window as any).__tldw_useConnectionStore = useConnectionStore

  // Optional helper so tests can derive the UX state from a raw
  // ConnectionState snapshot without re‑implementing the logic.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(window as any).__tldw_deriveUx = (state: any) => {
    try {
      return deriveConnectionUxState(state as ConnectionState)
    } catch {
      return "unknown"
    }
  }

  // Allow tests to flip the offline bypass without rebuilding the extension.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(window as any).__tldw_enableOfflineBypass = async () => {
    try {
      await useConnectionStore.getState().enableOfflineBypass()
      return true
    } catch {
      return false
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(window as any).__tldw_disableOfflineBypass = async () => {
    try {
      await useConnectionStore.getState().disableOfflineBypass()
      return true
    } catch {
      return false
    }
  }

  // Allow tests to force the unconfigured/waiting state without network calls.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(window as any).__tldw_forceUnconfigured = async () => {
    try {
      if (typeof chrome !== "undefined" && chrome?.storage?.local) {
        await new Promise<void>((resolve) =>
          chrome.storage.local.set({ [FORCE_UNCONFIGURED_KEY]: true }, () =>
            resolve()
          )
        )
      } else if (typeof localStorage !== "undefined") {
        localStorage.setItem(FORCE_UNCONFIGURED_KEY, "true")
      }
      await useConnectionStore.getState().checkOnce()
      return true
    } catch {
      return false
    }
  }
}
