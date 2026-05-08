import { test as base, expect, Page } from "@playwright/test"

/**
 * Diagnostics data collected during page visits
 */
export interface DiagnosticsData {
  console: Array<{ type: string; text: string; location?: { url: string; lineNumber: number } }>
  pageErrors: Array<{ message: string; stack: string }>
  requestFailures: Array<{ url: string; errorText: string }>
}

export type SmokeAllowlistScope = "console" | "request"

export interface SmokeHardGateAllowlistRule {
  id: string
  scope: SmokeAllowlistScope
  pattern: RegExp
  routes?: string[]
  rationale: string
  owner: string
  expiresOn: string
}

type ConsoleIssue = { type: string; text: string }
type RequestIssue = { url: string; errorText: string }

export interface ClassifiedSmokeIssues {
  pageErrors: Array<{ message: string; stack: string }>
  allowlistedConsoleErrors: Array<{ entry: ConsoleIssue; rule: SmokeHardGateAllowlistRule }>
  unexpectedConsoleErrors: ConsoleIssue[]
  allowlistedRequestFailures: Array<{
    entry: RequestIssue
    rule: SmokeHardGateAllowlistRule
  }>
  unexpectedRequestFailures: RequestIssue[]
}

/**
 * Extended test fixture that automatically collects diagnostics
 */
export const test = base.extend<{ diagnostics: DiagnosticsData }>({
  diagnostics: async ({ page }, use) => {
    const data: DiagnosticsData = {
      console: [],
      pageErrors: [],
      requestFailures: []
    }

    // Collect console messages
    page.on("console", (msg) => {
      const location = msg.location()
      data.console.push({
        type: msg.type(),
        text: msg.text(),
        location: location.url ? { url: location.url, lineNumber: location.lineNumber } : undefined
      })
    })

    // Collect page errors (uncaught exceptions)
    page.on("pageerror", (err) => {
      data.pageErrors.push({
        message: err.message,
        stack: err.stack || ""
      })
    })

    // Collect failed network requests
    page.on("requestfailed", (req) => {
      data.requestFailures.push({
        url: req.url(),
        errorText: req.failure()?.errorText || ""
      })
    })

    await use(data)
  }
})

export { expect }

const DEFAULT_SMOKE_LOAD_TIMEOUT_MS = 30_000
const MIN_SMOKE_LOAD_TIMEOUT_MS = 5_000

const resolveSmokeLoadTimeoutMs = (): number => {
  const raw = process.env.TLDW_SMOKE_LOAD_TIMEOUT_MS
  if (!raw || !raw.trim()) return DEFAULT_SMOKE_LOAD_TIMEOUT_MS
  const parsed = Number(raw)
  if (!Number.isFinite(parsed)) return DEFAULT_SMOKE_LOAD_TIMEOUT_MS
  const normalized = Math.floor(parsed)
  if (normalized < MIN_SMOKE_LOAD_TIMEOUT_MS) return DEFAULT_SMOKE_LOAD_TIMEOUT_MS
  return normalized
}

export const SMOKE_LOAD_TIMEOUT = resolveSmokeLoadTimeoutMs()

/**
 * Auth configuration for smoke tests
 */
export const AUTH_CONFIG = {
  serverUrl:
    process.env.TLDW_SERVER_URL ||
    process.env.TLDW_E2E_SERVER_URL ||
    process.env.E2E_TEST_BASE_URL ||
    "http://127.0.0.1:8000",
  apiKey:
    process.env.TLDW_API_KEY ||
    process.env.TLDW_E2E_API_KEY ||
    process.env.SINGLE_USER_API_KEY ||
    "THIS-IS-A-SECURE-KEY-123-FAKE-KEY",
  allowOffline: process.env.TLDW_E2E_ALLOW_OFFLINE !== "0"
}

type SeedAuthOverrides = {
  serverUrl?: string
  authMode?: "single-user" | "multi-user"
  apiKey?: string
  accessToken?: string
  allowOffline?: boolean
}

/**
 * Seed authentication config in localStorage before page loads
 * Pattern from login.spec.ts:35-44 and playwright-login.mjs:118-137
 */
export async function seedAuth(
  page: Page,
  overrides: SeedAuthOverrides = {}
): Promise<void> {
  const cfg = {
    serverUrl: overrides.serverUrl || AUTH_CONFIG.serverUrl,
    authMode: overrides.authMode || "single-user",
    apiKey: overrides.apiKey || AUTH_CONFIG.apiKey,
    accessToken: overrides.accessToken || "",
    allowOffline:
      typeof overrides.allowOffline === "boolean"
        ? overrides.allowOffline
        : AUTH_CONFIG.allowOffline
  }

  await page.addInitScript(
    (cfg) => {
      const readStorageValue = (key: string) => {
        try {
          const raw = localStorage.getItem(key)
          if (raw == null) return undefined
          return JSON.parse(raw)
        } catch {
          return localStorage.getItem(key) ?? undefined
        }
      }

      const writeStorageValue = (key: string, value: unknown) => {
        try {
          localStorage.setItem(key, JSON.stringify(value))
        } catch {}
      }

      const installChromeStorageShim = () => {
        const globalWindow = window as unknown as {
          chrome?: Record<string, unknown>
          browser?: Record<string, unknown>
        }

        const listeners = new Set<
          (changes: Record<string, { oldValue: unknown; newValue: unknown }>, area: string) => void
        >()

        const emitChanges = (
          changes: Record<string, { oldValue: unknown; newValue: unknown }>,
          area: string
        ) => {
          for (const listener of listeners) {
            try {
              listener(changes, area)
            } catch {}
          }
        }

        const areaApi = {
          get: async (
            keys?: string | string[] | Record<string, unknown> | null,
            callback?: (result: Record<string, unknown>) => void
          ) => {
            let result: Record<string, unknown>
            if (keys == null) {
              const out: Record<string, unknown> = {}
              for (let idx = 0; idx < localStorage.length; idx += 1) {
                const key = localStorage.key(idx)
                if (!key) continue
                out[key] = readStorageValue(key)
              }
              result = out
            } else if (typeof keys === "string") {
              result = { [keys]: readStorageValue(keys) }
            } else if (Array.isArray(keys)) {
              result = keys.reduce<Record<string, unknown>>((acc, key) => {
                acc[key] = readStorageValue(key)
                return acc
              }, {})
            } else {
              result = Object.entries(keys).reduce<Record<string, unknown>>(
                (acc, [key, fallback]) => {
                  const current = readStorageValue(key)
                  acc[key] = typeof current === "undefined" ? fallback : current
                  return acc
                },
                {}
              )
            }
            if (typeof callback === "function") {
              try {
                callback(result)
              } catch {}
            }
            return result
          },
          set: async (items: Record<string, unknown>, callback?: () => void) => {
            const changes: Record<string, { oldValue: unknown; newValue: unknown }> = {}
            for (const [key, value] of Object.entries(items || {})) {
              const oldValue = readStorageValue(key)
              writeStorageValue(key, value)
              changes[key] = { oldValue, newValue: value }
            }
            if (Object.keys(changes).length > 0) {
              emitChanges(changes, "sync")
            }
            if (typeof callback === "function") {
              try {
                callback()
              } catch {}
            }
          },
          remove: async (keys: string | string[], callback?: () => void) => {
            const values = Array.isArray(keys) ? keys : [keys]
            const changes: Record<string, { oldValue: unknown; newValue: unknown }> = {}
            for (const key of values) {
              const oldValue = readStorageValue(key)
              try {
                localStorage.removeItem(key)
              } catch {}
              changes[key] = { oldValue, newValue: undefined }
            }
            if (Object.keys(changes).length > 0) {
              emitChanges(changes, "sync")
            }
            if (typeof callback === "function") {
              try {
                callback()
              } catch {}
            }
          },
          clear: async (callback?: () => void) => {
            const changes: Record<string, { oldValue: unknown; newValue: unknown }> = {}
            for (let idx = 0; idx < localStorage.length; idx += 1) {
              const key = localStorage.key(idx)
              if (!key) continue
              changes[key] = { oldValue: readStorageValue(key), newValue: undefined }
            }
            try {
              localStorage.clear()
            } catch {}
            if (Object.keys(changes).length > 0) {
              emitChanges(changes, "sync")
            }
            if (typeof callback === "function") {
              try {
                callback()
              } catch {}
            }
          },
          getBytesInUse: async (_keys?: unknown, callback?: (bytes: number) => void) => {
            if (typeof callback === "function") {
              try {
                callback(0)
              } catch {}
            }
            return 0
          }
        }

        if (!globalWindow.chrome) {
          globalWindow.chrome = {}
        }
        const chromeLike = globalWindow.chrome as Record<string, unknown>
        if (!chromeLike.runtime) {
          chromeLike.runtime = { id: "mock-runtime-id" }
        } else if (
          typeof (chromeLike.runtime as { id?: unknown }).id === "undefined"
        ) {
          ;(chromeLike.runtime as { id?: string }).id = "mock-runtime-id"
        }
        const storageShim = {
          sync: areaApi,
          local: areaApi,
          managed: areaApi,
          onChanged: {
            addListener: (fn: (changes: Record<string, { oldValue: unknown; newValue: unknown }>, area: string) => void) =>
              listeners.add(fn),
            removeListener: (fn: (changes: Record<string, { oldValue: unknown; newValue: unknown }>, area: string) => void) =>
              listeners.delete(fn)
          }
        }
        chromeLike.storage = storageShim

        if (!globalWindow.browser) {
          globalWindow.browser = {}
        }
        const browserLike = globalWindow.browser as Record<string, unknown>
        browserLike.storage = storageShim as Record<string, unknown>
      }

      installChromeStorageShim()

      const authConfig = {
        serverUrl: cfg.serverUrl,
        authMode: cfg.authMode,
        apiKey: cfg.apiKey,
        accessToken: cfg.accessToken
      }

      try {
        localStorage.setItem(
          "tldwConfig",
          JSON.stringify(authConfig)
        )
      } catch {}
      try {
        const chromeLike = (window as unknown as { chrome?: any }).chrome
        chromeLike?.storage?.sync?.set?.({ tldwConfig: authConfig })
        chromeLike?.storage?.local?.set?.({ tldwConfig: authConfig })
        chromeLike?.storage?.sync?.set?.({ isMigrated: true })
        chromeLike?.storage?.local?.set?.({ isMigrated: true })
      } catch {}
      try {
        localStorage.setItem("isMigrated", "true")
      } catch {}
      // Backward-compat for routes still reading legacy top-level keys.
      try {
        localStorage.setItem("serverUrl", cfg.serverUrl)
      } catch {}
      try {
        localStorage.setItem("tldwServerUrl", cfg.serverUrl)
      } catch {}
      try {
        localStorage.setItem("authMode", cfg.authMode)
      } catch {}
      try {
        localStorage.setItem("apiKey", cfg.apiKey)
      } catch {}
      try {
        localStorage.setItem("accessToken", cfg.accessToken)
      } catch {}
      try {
        localStorage.setItem("__tldw_first_run_complete", "true")
      } catch {}
      try {
        localStorage.setItem("assistant_setup_dismissed", "true")
      } catch {}
      try {
        if (cfg.allowOffline) {
          localStorage.setItem("__tldw_allow_offline", "true")
        } else {
          localStorage.removeItem("__tldw_allow_offline")
        }
      } catch {}
    },
    cfg
  )
}

/**
 * Seed a deterministic admin fixture profile for smoke tests:
 * - keeps auth in single-user mode
 * - points serverUrl at the running WebUI base URL so Playwright route
 *   intercepts can fully own admin API responses.
 */
export async function seedAdminFixtureProfile(
  page: Page,
  baseURL?: string
): Promise<void> {
  const targetServerUrl =
    (typeof baseURL === "string" && baseURL.length > 0
      ? baseURL
      : process.env.TLDW_WEB_URL) || AUTH_CONFIG.serverUrl

  await seedAuth(page, {
    serverUrl: targetServerUrl,
    authMode: "single-user",
    apiKey: AUTH_CONFIG.apiKey,
    allowOffline: false
  })
}

/**
 * Patterns for console/error messages that are benign and should be ignored
 */
export const BENIGN_PATTERNS = [
  // ResizeObserver warnings are common and harmless
  /ResizeObserver loop/,
  // Non-Error promise rejections (often from cancelled requests)
  /Non-Error promise rejection/,
  // Aborted requests (navigation away, etc.)
  /net::ERR_ABORTED/,
  // Chrome extension errors
  /chrome-extension/,
  // React DevTools
  /Download the React DevTools/,
  // Hot reload messages
  /Fast Refresh/,
  /\[HMR\]/,
  // Favicon not found (common in dev)
  /favicon\.ico.*404/,
  // Source map warnings
  /Failed to load source map/,
  // Ant Design deprecation warnings
  /Warning.*findDOMNode is deprecated/,
  // Next.js hydration warnings that are often false positives
  /Hydration failed/,
  /There was an error while hydrating/
]

/**
 * Temporary allowlist for non-fatal console/request noise observed in full all-pages smoke.
 * These entries are intentionally narrow and route-scoped where possible.
 */
export const SMOKE_HARD_GATE_ALLOWLIST: SmokeHardGateAllowlistRule[] = [
  {
    id: "m5-chat-history-rate-limit",
    scope: "console",
    pattern: /rate_limited\s+\(GET\s+\/api\/v1\/chats\/\?limit=\d+&offset=\d+&ordering=-updated_at\)/i,
    rationale: "Chat history request bursts can hit server-side 429 in dense all-pages sweeps.",
    owner: "WebUI",
    expiresOn: "2026-03-31"
  },
  {
    id: "m5-http-429-resource",
    scope: "console",
    pattern: /Failed to load resource: the server responded with a status of 429/i,
    rationale: "Known rate-limit noise while traversing all routes in parallel; triaged separately.",
    owner: "Platform",
    expiresOn: "2026-03-31"
  },
  {
    id: "m5-react-key-prop-spread-warning",
    scope: "console",
    pattern: /A props object containing a "key" prop is being spread into JSX/i,
    rationale: "Known React warning in connectors/settings surfaces; no runtime crash.",
    owner: "WebUI",
    expiresOn: "2026-03-31",
    routes: ["/connectors", "/connectors/browse", "/connectors/jobs", "/connectors/sources", "/settings", "/config", "/profile", "/privileges"]
  },
  {
    id: "m5-react-non-boolean-attribute-warning",
    scope: "console",
    pattern: /Received `%s` for a non-boolean attribute `%s`/i,
    rationale: "Known non-breaking attribute warning in flashcards render path.",
    owner: "WebUI",
    expiresOn: "2026-03-31",
    routes: ["/flashcards"]
  },
  {
    id: "m5-media-max-update-depth-warning",
    scope: "console",
    pattern: /Maximum update depth exceeded/i,
    rationale:
      "Known media route warning remains scoped to legacy review/media surfaces; Stage 3 critical audited routes are enforced separately.",
    owner: "WebUI",
    expiresOn: "2026-03-31",
    routes: ["/media", "/media/*", "/media-multi"]
  },
  {
    id: "m5-optional-resource-404-noise",
    scope: "console",
    pattern: /Failed to load resource: the server responded with a status of 404/i,
    rationale: "Known optional static/resource fetch misses in selected routes during dev runtime.",
    owner: "WebUI",
    expiresOn: "2026-03-31",
    routes: [
      "/review",
      "/media",
      "/media/*",
      "/media-multi",
      "/prompt-studio",
      "/settings/prompt-studio",
      "/settings/family-guardrails",
      "/settings/about",
      "/settings/speech",
      "/chatbooks",
      "/watchlists",
      "/prompts",
      "/reading",
      "/collections",
      "/admin",
      "/admin/server",
      "/notes",
      "/moderation-playground",
      "/chunking-playground",
      "/workspace-playground",
      "/stt",
      "/speech",
      "/tts",
      "/audio",
      "/404",
      "/__wayfinding-missing-route__"
    ]
  },
  {
    id: "m5-drawer-width-deprecation-noise",
    scope: "console",
    pattern: /Warning:\s+\[antd:\s*Drawer\]\s+`width` is deprecated\. Please use `size` instead\./i,
    rationale:
      "Known Ant Design Drawer deprecation warning in selected routes; no functional regression in smoke path.",
    owner: "WebUI",
    expiresOn: "2026-03-31",
    routes: ["/media-multi", "/kanban", "/review"]
  },
  {
    id: "m5-model-oauth-status-403-noise",
    scope: "console",
    pattern: /Failed to load resource: the server responded with a status of 403 \(Forbidden\)/i,
    rationale:
      "Model settings probes optional OAuth status endpoint that can return 403 in minimal smoke backend profile.",
    owner: "Platform",
    expiresOn: "2026-03-31",
    routes: ["/settings/model"]
  },
  {
    id: "m5-notes-title-settings-cors-noise",
    scope: "console",
    pattern:
      /Access to fetch at 'http:\/\/127\.0\.0\.1:\d+\/api\/v1\/admin\/notes\/title-settings'.*blocked by CORS policy/i,
    rationale:
      "Notes title settings probe may be CORS-blocked in isolated smoke backend mode while page remains recoverable.",
    owner: "Platform",
    expiresOn: "2026-03-31",
    routes: ["/notes"]
  },
  {
    id: "m5-notes-title-settings-net-failed-noise",
    scope: "console",
    pattern: /Failed to load resource: net::ERR_FAILED/i,
    rationale:
      "Companion browser error after expected CORS rejection for notes title settings probe in smoke mode.",
    owner: "Platform",
    expiresOn: "2026-03-31",
    routes: ["/notes"]
  },
  {
    id: "m5-notes-title-settings-request-failure-noise",
    scope: "request",
    pattern: /\/api\/v1\/admin\/notes\/title-settings\s+\(net::ERR_FAILED\)/i,
    rationale:
      "Request-failure companion signal for expected notes title settings CORS rejection in isolated smoke mode.",
    owner: "Platform",
    expiresOn: "2026-03-31",
    routes: ["/notes"]
  },
  {
    id: "m5-media-not-found-search-console-noise",
    scope: "console",
    pattern:
      /Media search error:\s+Error:\s+Not Found \(GET \/api\/v1\/media\/\?page=1&results_per_page=20&include_keywords=true\)/i,
    rationale:
      "Media pages surface a handled Not Found search message when media endpoints are absent in minimal smoke backend profile.",
    owner: "WebUI",
    expiresOn: "2026-03-31",
    routes: ["/media", "/media/*"]
  },
  {
    id: "m5-quiz-tabs-deprecation-warning",
    scope: "console",
    pattern:
      /Warning:\s+\[antd:\s*Tabs\]\s+`destroyInactiveTabPane` is deprecated/i,
    rationale:
      "Known Ant Design deprecation warning in quiz route; tracked separately from functional regressions.",
    owner: "WebUI",
    expiresOn: "2026-03-31",
    routes: ["/quiz"]
  },
  {
    id: "m5-quiz-list-deprecation-warning",
    scope: "console",
    pattern:
      /Warning:\s+\[antd:\s*List\]\s+The `List` component is deprecated/i,
    rationale:
      "Known Ant Design List deprecation warning in quiz route; no user-impacting runtime break.",
    owner: "WebUI",
    expiresOn: "2026-03-31",
    routes: ["/quiz"]
  },
  {
    id: "m5-quiz-attempts-422-noise",
    scope: "console",
    pattern:
      /Failed to load resource: the server responded with a status of 422 \(Unprocessable Entity\)/i,
    rationale:
      "Quiz attempts list probes may return 422 in minimal smoke backend profile while route UI remains recoverable.",
    owner: "Platform",
    expiresOn: "2026-03-31",
    routes: ["/quiz"]
  },
  {
    id: "m5-prompt-studio-422-noise",
    scope: "console",
    pattern:
      /Failed to load resource: the server responded with a status of 422 \(Unprocessable Entity\)/i,
    rationale:
      "Prompt Studio settings probes can return 422 in minimal smoke backend profile.",
    owner: "Platform",
    expiresOn: "2026-03-31",
    routes: ["/prompt-studio"]
  },
  {
    id: "m5-dictionaries-optional-endpoint-500",
    scope: "console",
    pattern: /Failed to load resource: the server responded with a status of 500/i,
    rationale:
      "Dictionaries route can hit optional backend handlers unavailable in minimal smoke profile.",
    owner: "Platform",
    expiresOn: "2026-03-31",
    routes: ["/dictionaries"]
  },
  {
    id: "m5-collections-antd-message-context-warning",
    scope: "console",
    pattern:
      /Warning:\s+\[antd:\s*message\]\s+Static function can not consume context like dynamic theme/i,
    rationale:
      "Known Ant Design message context warning in collections route; no functional regression.",
    owner: "WebUI",
    expiresOn: "2026-03-31",
    routes: ["/collections", "/reading"]
  },
  {
    id: "m5-characters-useform-context-warning",
    scope: "console",
    pattern:
      /Warning:\s+Instance created by `useForm` is not connected to any Form element\.\s+Forget to pass `form` prop\?/i,
    rationale:
      "Known Ant Design form instance warning in characters route under minimal smoke backend profile; route remains functional.",
    owner: "WebUI",
    expiresOn: "2026-03-31",
    routes: ["/characters"]
  },
  {
    id: "m5-model-metadata-rate-limit-log-noise",
    scope: "console",
    pattern:
      /Failed to fetch models from tldw:\s+Error:\s+rate_limited \(GET \/api\/v1\/llm\/models\/metadata\)/i,
    rationale:
      "Dense smoke sweeps can rate-limit model metadata probes; treated as environment noise for these routes.",
    owner: "Platform",
    expiresOn: "2026-03-31",
    routes: ["/content-review", "/claims-review", "/workspace-playground"]
  },
  {
    id: "m5-model-metadata-abort-noise",
    scope: "console",
    pattern:
      /Failed to fetch models from tldw:\s+AbortError:\s+signal is aborted without reason/i,
    rationale:
      "Workspace Playground can abort in-flight model metadata fetches during route hydration without user-impacting breakage.",
    owner: "WebUI",
    expiresOn: "2026-03-31",
    routes: ["/workspace-playground"]
  },
  {
    id: "m5-chatbooks-evaluations-cors-noise",
    scope: "console",
    pattern:
      /Access to fetch at 'http:\/\/127\.0\.0\.1:\d+\/api\/v1\/evaluations\/\?limit=100'.*blocked by CORS policy/i,
    rationale:
      "Chatbooks route issues a best-effort evaluations probe that may be CORS-blocked in isolated smoke backend mode.",
    owner: "Platform",
    expiresOn: "2026-03-31",
    routes: ["/chatbooks"]
  },
  {
    id: "m5-chatbooks-evaluations-net-failed-noise",
    scope: "console",
    pattern: /Failed to load resource: net::ERR_FAILED/i,
    rationale:
      "Companion browser error emitted after expected CORS rejection for optional evaluations probe in chatbooks.",
    owner: "Platform",
    expiresOn: "2026-03-31",
    routes: ["/chatbooks"]
  },
  {
    id: "m5-chatbooks-evaluations-request-failure-noise",
    scope: "request",
    pattern: /\/api\/v1\/evaluations\/\?limit=100\s+\(net::ERR_FAILED\)/i,
    rationale:
      "Request-failure companion signal for expected chatbooks evaluations CORS rejection in isolated smoke mode.",
    owner: "Platform",
    expiresOn: "2026-03-31",
    routes: ["/chatbooks"]
  },
  {
    id: "m5-chatbooks-optional-dictionaries-500",
    scope: "console",
    pattern: /Failed to load resource: the server responded with a status of 500/i,
    rationale:
      "Chatbooks route performs optional dictionaries checks that can return 500 in minimal smoke backend profiles.",
    owner: "Platform",
    expiresOn: "2026-03-31",
    routes: ["/chatbooks"]
  },
  {
    id: "m5-route-boundary-forced-react-overlay-warning",
    scope: "console",
    pattern: /The above error occurred in the <ForcedRouteErrorProbe> component/i,
    rationale: "Expected React error-overlay emission when route boundary fixture intentionally throws.",
    owner: "WebUI",
    expiresOn: "2026-03-31",
    routes: [
      "/admin/server",
      "/admin/llamacpp",
      "/admin/mlx",
      "/content-review",
      "/data-tables",
      "/kanban",
      "/chunking-playground",
      "/moderation-playground",
      "/collections",
      "/world-books",
      "/dictionaries",
      "/characters",
      "/items",
      "/document-workspace",
      "/speech"
    ]
  },
  {
    id: "m5-route-boundary-forced-error-log",
    scope: "console",
    pattern: /\[RouteErrorBoundary:[^\]]+\]\s+Error:\s+Forced route boundary error/i,
    rationale: "Route boundary fixture emits deterministic forced-error log to confirm recovery branch.",
    owner: "WebUI",
    expiresOn: "2026-03-31",
    routes: [
      "/admin/server",
      "/admin/llamacpp",
      "/admin/mlx",
      "/content-review",
      "/data-tables",
      "/kanban",
      "/chunking-playground",
      "/moderation-playground",
      "/collections",
      "/world-books",
      "/dictionaries",
      "/characters",
      "/items",
      "/document-workspace",
      "/speech"
    ]
  },
  {
    id: "m5-admin-optional-endpoint-500",
    scope: "console",
    pattern: /Failed to load resource: the server responded with a status of 500/i,
    rationale: "Admin pages hit optional backend endpoints unavailable in minimal smoke profile.",
    owner: "Platform",
    expiresOn: "2026-03-31",
    routes: ["/admin", "/admin/server", "/admin/orgs", "/admin/data-ops", "/admin/watchlists-items", "/admin/watchlists-runs", "/admin/maintenance"]
  },
  {
    id: "m5-llamacpp-unavailable-503",
    scope: "console",
    pattern: /Failed to load resource: the server responded with a status of 503/i,
    rationale: "Expected when llama.cpp backend is not configured in smoke environment.",
    owner: "Platform",
    expiresOn: "2026-03-31",
    routes: ["/admin/llamacpp"]
  }
]

/**
 * Check if an error/warning message is benign
 */
export function isBenign(text: string): boolean {
  return BENIGN_PATTERNS.some((p) => p.test(text))
}

/**
 * Filter diagnostics to only critical issues
 */
export function getCriticalIssues(diagnostics: DiagnosticsData): {
  pageErrors: Array<{ message: string; stack: string }>
  consoleErrors: Array<{ type: string; text: string }>
  requestFailures: Array<{ url: string; errorText: string }>
} {
  return {
    pageErrors: diagnostics.pageErrors.filter((e) => !isBenign(e.message)),
    consoleErrors: diagnostics.console.filter(
      (c) => c.type === "error" && !isBenign(c.text)
    ),
    requestFailures: diagnostics.requestFailures.filter(
      (r) => !isBenign(r.url) && !isBenign(r.errorText)
    )
  }
}

function findAllowlistRule(
  scope: SmokeAllowlistScope,
  text: string,
  routePath: string
): SmokeHardGateAllowlistRule | null {
  const normalizedRoutePath = normalizeRoutePath(routePath)
  for (const rule of SMOKE_HARD_GATE_ALLOWLIST) {
    if (rule.scope !== scope) continue
    if (
      rule.routes &&
      !rule.routes.some((candidate) => routePatternMatches(candidate, normalizedRoutePath))
    ) {
      continue
    }
    if (rule.pattern.test(text)) {
      return rule
    }
  }
  return null
}

function normalizeRoutePath(routePath: string): string {
  try {
    if (routePath.startsWith("http://") || routePath.startsWith("https://")) {
      return new URL(routePath).pathname
    }
  } catch {}
  return routePath.split("?")[0]?.split("#")[0] || routePath
}

function routePatternMatches(pattern: string, routePath: string): boolean {
  if (pattern.endsWith("*")) {
    const prefix = pattern.slice(0, -1)
    return routePath.startsWith(prefix)
  }
  return pattern === routePath
}

export function classifySmokeIssues(
  routePath: string,
  issues: ReturnType<typeof getCriticalIssues>
): ClassifiedSmokeIssues {
  const classified: ClassifiedSmokeIssues = {
    pageErrors: issues.pageErrors,
    allowlistedConsoleErrors: [],
    unexpectedConsoleErrors: [],
    allowlistedRequestFailures: [],
    unexpectedRequestFailures: []
  }

  for (const entry of issues.consoleErrors) {
    const match = findAllowlistRule("console", entry.text, routePath)
    if (match) {
      classified.allowlistedConsoleErrors.push({ entry, rule: match })
    } else {
      classified.unexpectedConsoleErrors.push(entry)
    }
  }

  for (const entry of issues.requestFailures) {
    const requestText = `${entry.url} (${entry.errorText})`
    const match = findAllowlistRule("request", requestText, routePath)
    if (match) {
      classified.allowlistedRequestFailures.push({ entry, rule: match })
    } else {
      classified.unexpectedRequestFailures.push(entry)
    }
  }

  return classified
}
