import { test as base, expect, Page, type Route } from "@playwright/test"
import { resolveE2eApiKey } from "../utils/e2e-auth"

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

type ConsoleIssue = DiagnosticsData["console"][number]
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
const DEFAULT_SMOKE_SERVER_URL = "http://127.0.0.1:8000"

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

const resolveSmokeServerUrl = (): string =>
  process.env.TLDW_SERVER_URL ||
  process.env.TLDW_E2E_SERVER_URL ||
  process.env.E2E_TEST_BASE_URL ||
  DEFAULT_SMOKE_SERVER_URL

const SMOKE_SERVER_URL = resolveSmokeServerUrl()

/**
 * Auth configuration for smoke tests
 */
export const AUTH_CONFIG = {
  serverUrl: SMOKE_SERVER_URL,
  apiKey: resolveE2eApiKey({ serverUrl: SMOKE_SERVER_URL }),
  allowOffline: process.env.TLDW_E2E_ALLOW_OFFLINE !== "0"
}

export const shouldInstallSmokeApiStubs = (
  env: Record<string, string | undefined> = process.env
): boolean => env.TLDW_LIVE_TIER_UAT !== "1"

const fulfillSmokeJson = async (
  route: Route,
  status: number,
  body: unknown
): Promise<void> => {
  await route.fulfill({
    status,
    contentType: "application/json",
    headers: {
      "access-control-allow-origin": "*",
      "access-control-allow-headers": "*",
      "access-control-allow-methods": "GET,POST,OPTIONS"
    },
    body: JSON.stringify(body)
  })
}

async function stubCompletedFirstRunSetup(page: Page): Promise<void> {
  await page.route(/\/api\/v1\/setup\/first-run\/(?:state|metadata)(?:\?.*)?$/, async (route) => {
    const request = route.request()
    const method = request.method().toUpperCase()
    const path = new URL(request.url()).pathname

    if (method === "OPTIONS") {
      await route.fulfill({
        status: 204,
        headers: {
          "access-control-allow-origin": "*",
          "access-control-allow-headers": "*",
          "access-control-allow-methods": "GET,POST,OPTIONS"
        }
      })
      return
    }

    if (method !== "GET") {
      await route.fallback()
      return
    }

    if (path === "/api/v1/setup/first-run/state") {
      await fulfillSmokeJson(route, 200, {
        status: "completed",
        current_step: null,
        completed_steps: ["first_chat"],
        skipped_steps: [],
        step_data: {},
        acknowledged_steps: ["first_chat"],
        first_chat: {
          completed: true,
          provider: "openai",
          model: "gpt-4.1-mini",
          response_id: "smoke-first-chat",
          completed_at: "2026-06-01T12:00:00Z"
        },
        skip_reason: null,
        created_at: "2026-06-01T12:00:00Z",
        updated_at: "2026-06-01T12:00:00Z",
        completed_at: "2026-06-01T12:00:00Z"
      })
      return
    }

    await fulfillSmokeJson(route, 200, {
      auth_mode: "single_user",
      bundled_single_user_auth_available: true,
      manual_auth_required: false,
      setup_required: false,
      setup_completed: true,
      remote_setup_enabled: false,
      connection: {
        frontend_origin: "http://localhost:3000",
        api_origin: SMOKE_SERVER_URL,
        browser_access: "local"
      },
      setup_paths: [],
      multi_user_exit: { guide_path: "/Docs/AuthNZ/Multi_User_Setup.md" }
    })
  })
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

  if (cfg.authMode === "single-user" && shouldInstallSmokeApiStubs()) {
    await page.route("**/api/_tldw-webui/runtime-config", async (route) => {
      await fulfillSmokeJson(route, 200, {
        runtimeAuth: { available: false },
        networking: {
          deploymentMode: "advanced",
          serverUrl: cfg.serverUrl
        }
      })
    })
  }

  await page.addInitScript(
    (cfg) => {
      const readStorageValue = (key: string) => {
        const raw = localStorage.getItem(key)
        if (raw == null) return undefined
        try {
          return JSON.parse(raw)
        } catch {
          return raw
        }
      }

      const writeStorageValue = (key: string, value: unknown) => {
        localStorage.setItem(key, JSON.stringify(value))
      }

      const reportShimError = (label: string, error: unknown) => {
        console.warn(`[tldw smoke storage shim] ${label}`, error)
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
            } catch (error) {
              reportShimError("onChanged listener failed", error)
            }
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
              } catch (error) {
                reportShimError("get callback failed", error)
              }
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
              } catch (error) {
                reportShimError("set callback failed", error)
              }
            }
          },
          remove: async (keys: string | string[], callback?: () => void) => {
            const values = Array.isArray(keys) ? keys : [keys]
            const changes: Record<string, { oldValue: unknown; newValue: unknown }> = {}
            for (const key of values) {
              const oldValue = readStorageValue(key)
              localStorage.removeItem(key)
              changes[key] = { oldValue, newValue: undefined }
            }
            if (Object.keys(changes).length > 0) {
              emitChanges(changes, "sync")
            }
            if (typeof callback === "function") {
              try {
                callback()
              } catch (error) {
                reportShimError("remove callback failed", error)
              }
            }
          },
          clear: async (callback?: () => void) => {
            const changes: Record<string, { oldValue: unknown; newValue: unknown }> = {}
            for (let idx = 0; idx < localStorage.length; idx += 1) {
              const key = localStorage.key(idx)
              if (!key) continue
              changes[key] = { oldValue: readStorageValue(key), newValue: undefined }
            }
            localStorage.clear()
            if (Object.keys(changes).length > 0) {
              emitChanges(changes, "sync")
            }
            if (typeof callback === "function") {
              try {
                callback()
              } catch (error) {
                reportShimError("clear callback failed", error)
              }
            }
          },
          getBytesInUse: async (_keys?: unknown, callback?: (bytes: number) => void) => {
            if (typeof callback === "function") {
              try {
                callback(0)
              } catch (error) {
                reportShimError("getBytesInUse callback failed", error)
              }
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

      const authConfig =
        cfg.authMode === "single-user"
          ? {
              serverUrl: cfg.serverUrl,
              authMode: cfg.authMode,
              authSource: "manual",
              credentialSource: "manual",
              apiKeyPersistence: "device",
              apiKeyServerOrigin: cfg.serverUrl,
              apiKey: cfg.apiKey,
              accessToken: cfg.accessToken
            }
          : {
              serverUrl: cfg.serverUrl,
              authMode: cfg.authMode,
              accessToken: cfg.accessToken
            }

      // lgtm[js/clear-text-storage-of-sensitive-data]: synthetic E2E auth seed only.
      localStorage.setItem("tldwConfig", JSON.stringify(authConfig))
      localStorage.setItem("isMigrated", "true")
      // Backward-compat for routes still reading legacy top-level keys.
      localStorage.setItem("serverUrl", cfg.serverUrl)
      localStorage.setItem("tldwServerUrl", cfg.serverUrl)
      localStorage.setItem("authMode", cfg.authMode)
      // lgtm[js/clear-text-storage-of-sensitive-data]: synthetic E2E auth seed only.
      localStorage.setItem("accessToken", cfg.accessToken)
      localStorage.setItem("__tldw_first_run_complete", "true")
      localStorage.setItem("assistant_setup_dismissed", "true")
      localStorage.setItem("__tldw_test_bypass", "true")
      localStorage.setItem("dw-tips-tour-completed", "true")
      localStorage.setItem("document-workspace-onboarding-dismissed", "true")
      localStorage.setItem("tldw:first-source-milestone-dismissed", "1")
      if (cfg.allowOffline) {
        localStorage.setItem("__tldw_allow_offline", "true")
      } else {
        localStorage.removeItem("__tldw_allow_offline")
      }
    },
    cfg
  )
  if (shouldInstallSmokeApiStubs()) {
    await stubCompletedFirstRunSetup(page)
  }
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
  /There was an error while hydrating/,
  /Failed to execute 'removeChild' on 'Node': The node to be removed is not a child of this node\./
]

/**
 * Temporary allowlist for non-fatal console/request noise observed in full all-pages smoke.
 * These entries are intentionally narrow and route-scoped where possible.
 */
export const SMOKE_HARD_GATE_ALLOWLIST: SmokeHardGateAllowlistRule[] = [
  {
    id: "m5-wayfinding-missing-route-document-404",
    scope: "console",
    pattern:
      /^https?:\/\/[^/\s]+\/__wayfinding-missing-route__\s+Failed to load resource: the server responded with a status of 404\b/i,
    rationale:
      "The Wayfinding 404-recovery test requests a deliberately missing route, so its own document 404 is expected. TASK-13414 re-observed it; TASK-13406 owns the policy for deliberate fixture emissions.",
    owner: "WebUI",
    expiresOn: "2026-10-31",
    routes: ["/__wayfinding-missing-route__"]
  },
  {
    id: "m5-optional-resource-404-noise",
    scope: "console",
    pattern: /\/api\/v1\/moderation\/review\/items(?:\?[^ ]*)?\s+Failed to load resource: the server responded with a status of 404\b/i,
    rationale:
      "The minimal smoke backend omits moderation review items. /moderation requests it on mount, so a 404 that lands before the console snapshot is expected and the route stays recoverable. TASK-13406 owns adding the endpoint or retiring this rule.",
    owner: "WebUI",
    expiresOn: "2026-10-31",
    routes: ["/moderation"]
  },
  {
    id: "m5-route-boundary-forced-react-overlay-warning",
    scope: "console",
    pattern: /The above error occurred in the <ForcedRouteErrorProbe> component/i,
    rationale:
      "Expected React error-overlay emission from deliberate route-boundary fixtures (development runtime only). TASK-13414 re-observed it; TASK-13406 owns the policy for deliberate fixture emissions.",
    owner: "WebUI",
    expiresOn: "2026-10-31",
    routes: [
      "/admin/server",
      "/admin/llamacpp",
      "/admin/mlx",
      "/content-review",
      "/data-tables",
      "/kanban",
      "/chunking-playground",
      "/moderation",
      "/moderation/rules",
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
    rationale:
      "Deliberate route-boundary fixture logs confirm the recovery branch (development runtime only). TASK-13414 re-observed it; TASK-13406 owns the policy for deliberate fixture emissions.",
    owner: "WebUI",
    expiresOn: "2026-10-31",
    routes: [
      "/admin/server",
      "/admin/llamacpp",
      "/admin/mlx",
      "/content-review",
      "/data-tables",
      "/kanban",
      "/chunking-playground",
      "/moderation",
      "/moderation/rules",
      "/collections",
      "/world-books",
      "/dictionaries",
      "/characters",
      "/items",
      "/document-workspace",
      "/speech"
    ]
  }
]

const ALLOWLIST_GLOBAL_RATIONALE_PATTERN =
  /\b(all-pages|all routes|cross-route|dense|dev runtime|global|parallel|route boundary|runtime)\b/i
const ALLOWLIST_OWNER_PATTERN = /\S/
const ISO_DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/

function parseAllowlistExpiry(expiresOn: string): number | null {
  if (!ISO_DATE_PATTERN.test(expiresOn)) return null
  const [year, month, day] = expiresOn.split("-").map(Number)
  if (!Number.isInteger(year) || !Number.isInteger(month) || !Number.isInteger(day)) {
    return null
  }
  if (month < 1 || month > 12) return null
  const daysInMonth = new Date(Date.UTC(year, month, 0)).getUTCDate()
  if (day < 1 || day > daysInMonth) return null

  const endOfDayUtc = Date.UTC(year, month - 1, day, 23, 59, 59, 999)
  if (Number.isNaN(endOfDayUtc)) return null

  const reconstructedDate = new Date(endOfDayUtc)
  if (
    reconstructedDate.getUTCFullYear() !== year ||
    reconstructedDate.getUTCMonth() + 1 !== month ||
    reconstructedDate.getUTCDate() !== day
  ) {
    return null
  }
  return endOfDayUtc
}

export function validateSmokeHardGateAllowlist(
  rules: SmokeHardGateAllowlistRule[] = SMOKE_HARD_GATE_ALLOWLIST,
  now: Date = new Date()
): string[] {
  const errors: string[] = []
  const seenIds = new Set<string>()
  const nowMs = now.getTime()

  rules.forEach((rule, index) => {
    const label = rule.id || `entry-${index}`

    if (!rule.id?.trim()) {
      errors.push(`${label}: missing id`)
    } else if (seenIds.has(rule.id)) {
      errors.push(`${label}: duplicate id`)
    } else {
      seenIds.add(rule.id)
    }

    if (rule.scope !== "console" && rule.scope !== "request") {
      errors.push(`${label}: scope must be console or request`)
    }

    if (!(rule.pattern instanceof RegExp)) {
      errors.push(`${label}: pattern must be a RegExp`)
    } else if (rule.pattern.global || rule.pattern.sticky) {
      errors.push(`${label}: pattern must not use global or sticky flags`)
    }

    if (!rule.rationale?.trim()) {
      errors.push(`${label}: missing rationale`)
    }

    if (!ALLOWLIST_OWNER_PATTERN.test(rule.owner || "")) {
      errors.push(`${label}: missing owner`)
    }

    const expiry = parseAllowlistExpiry(rule.expiresOn || "")
    if (expiry === null) {
      errors.push(`${label}: expiresOn must be a valid YYYY-MM-DD date`)
    } else if (expiry < nowMs) {
      errors.push(`${label}: expired on ${rule.expiresOn}`)
    }

    const routes = Array.isArray(rule.routes) ? rule.routes : []
    if (routes.length > 0) {
      for (const route of routes) {
        if (!route.startsWith("/")) {
          errors.push(`${label}: route scope must start with / (${route})`)
        }
      }
    } else if (!ALLOWLIST_GLOBAL_RATIONALE_PATTERN.test(rule.rationale || "")) {
      errors.push(
        `${label}: unscoped allowlist entries need a global/cross-route rationale`
      )
    }
  })

  return errors
}

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
  consoleErrors: DiagnosticsData["console"]
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
  } catch {
    // Invalid absolute URLs fall back to simple route-path stripping below.
  }
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
    const match = findAllowlistRule("console", `${entry.location?.url || ""} ${entry.text}`, routePath)
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
