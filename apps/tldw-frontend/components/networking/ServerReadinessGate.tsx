import React from "react"

import { resolvePublicApiOrigin, type DeploymentEnv } from "@web/lib/api-base"
import { useConnectionStore } from "@tldw/ui/store/connection"
import {
  StatePanel,
  type StatePanelDiagnostic,
} from "@tldw/ui/components/ui/state"
import { ServerHealthWarningBanner } from "./ServerHealthWarningBanner"
import { ServerReconnectBanner } from "./ServerReconnectBanner"

const _env: DeploymentEnv = {
  NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE,
  NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL
}

const MAX_WAIT_MS = 15_000
const RETRY_INTERVAL_MS = 2_000
const OFFLINE_BYPASS_KEYS = ["__tldw_allow_offline", "__tldw_test_bypass"] as const

type GateState = "checking" | "ready" | "waiting" | "timeout" | "degraded"
type ServerReadinessPublishedState = {
  state: "ready" | "degraded" | "blocked"
  degradedChecks: string[]
  healthUrl: string
  httpStatus?: number
  healthStatus?: string
  errorMessage?: string
  checkedAt: string
}
type ReadinessResult =
  | { state: "ready"; diagnostics: ServerReadinessPublishedState }
  | { state: "degraded"; degradedChecks: string[]; diagnostics: ServerReadinessPublishedState }
  | { state: "blocked"; diagnostics: ServerReadinessPublishedState }

declare global {
  interface Window {
    __tldwServerReadinessState?: ServerReadinessPublishedState
  }
}

const ENTERABLE_HTTP_STATUSES = new Set([200, 206])
const READY_HEALTH_STATUSES = new Set(["healthy", "ok"])
const HEALTHY_CHECK_STATUSES = new Set(["healthy", "ok"])
const SERVER_READINESS_STATE_EVENT = "tldw:server-readiness-state"
const HEALTH_PATH = "/health"

const trimTrailingSlash = (value: string): string => value.replace(/\/+$/, "")
const buildHealthUrl = (origin: string): string =>
  origin ? `${trimTrailingSlash(origin)}${HEALTH_PATH}` : HEALTH_PATH

const warnReadinessUrlIssue = (
  message: string,
  metadata?: Record<string, string>
): void => {
  if (metadata) {
    console.warn(message, metadata)
    return
  }
  console.warn(message)
}

export function resolveReadinessHealthUrl({
  configuredServerUrl,
  env,
  pageOrigin
}: {
  configuredServerUrl?: string | null
  env: DeploymentEnv
  pageOrigin?: string
}): string {
  const configured = String(configuredServerUrl || "").trim()
  if (configured) {
    try {
      const configuredUrl = new URL(configured)
      if (
        configuredUrl.protocol === "http:" ||
        configuredUrl.protocol === "https:"
      ) {
        return buildHealthUrl(configuredUrl.origin)
      }
      warnReadinessUrlIssue(
        "Ignoring unsupported tldw server URL protocol for readiness health check.",
        { protocol: configuredUrl.protocol }
      )
    } catch {
      warnReadinessUrlIssue(
        "Ignoring invalid tldw server URL for readiness health check."
      )
    }
  }

  try {
    const fallbackOrigin =
      typeof pageOrigin === "string"
        ? resolvePublicApiOrigin(env, pageOrigin)
        : resolvePublicApiOrigin(env)
    return buildHealthUrl(fallbackOrigin)
  } catch {
    warnReadinessUrlIssue(
      "Falling back to relative readiness health check after API origin resolution failed."
    )
    return HEALTH_PATH
  }
}

function readStorageFlag(key: string): boolean {
  try {
    return window.localStorage.getItem(key) === "true"
  } catch {
    return false
  }
}

function shouldBypassReadinessForOffline(): boolean {
  if (typeof window === "undefined") return false
  return OFFLINE_BYPASS_KEYS.some(readStorageFlag)
}

function extractDegradedChecks(body: unknown): string[] {
  if (!body || typeof body !== "object") return []
  const checks = (body as { checks?: unknown }).checks
  if (!checks || typeof checks !== "object") return []

  return Object.entries(checks as Record<string, unknown>)
    .filter(([, value]) => {
      if (!value || typeof value !== "object") return true
      const status = (value as { status?: unknown }).status
      if (typeof status !== "string") return true
      return !HEALTHY_CHECK_STATUSES.has(status.toLowerCase())
    })
    .map(([name]) => name)
}

function extractHealthStatus(body: unknown): string | undefined {
  if (!body || typeof body !== "object") return undefined
  const status = (body as { status?: unknown }).status
  return typeof status === "string" ? status.toLowerCase() : undefined
}

function buildReadinessDiagnostics({
  state,
  healthUrl,
  degradedChecks = [],
  httpStatus,
  healthStatus,
  errorMessage
}: {
  state: "ready" | "degraded" | "blocked"
  healthUrl: string
  degradedChecks?: string[]
  httpStatus?: number
  healthStatus?: string
  errorMessage?: string
}): ServerReadinessPublishedState {
  return {
    state,
    degradedChecks,
    healthUrl,
    httpStatus,
    healthStatus,
    errorMessage,
    checkedAt: new Date().toISOString()
  }
}

async function checkHealth(healthUrl: string): Promise<ReadinessResult> {
  try {
    const res = await fetch(healthUrl, {
      method: "GET",
      signal: AbortSignal.timeout(3000)
    })
    const isEnterable = ENTERABLE_HTTP_STATUSES.has(res.status)
    let body: unknown
    let healthStatus: string | undefined
    try {
      body = await res.json()
      healthStatus = extractHealthStatus(body)
    } catch (err) {
      if (isEnterable) {
        return {
          state: "blocked",
          diagnostics: buildReadinessDiagnostics({
            state: "blocked",
            healthUrl,
            httpStatus: res.status,
            errorMessage: err instanceof Error ? err.message : "Could not parse health response."
          })
        }
      }
    }
    if (!isEnterable) {
      return {
        state: "blocked",
        diagnostics: buildReadinessDiagnostics({
          state: "blocked",
          healthUrl,
          httpStatus: res.status,
          healthStatus
        })
      }
    }
    const status = healthStatus ?? ""
    if (READY_HEALTH_STATUSES.has(status)) {
      return {
        state: "ready",
        diagnostics: buildReadinessDiagnostics({
          state: "ready",
          healthUrl,
          httpStatus: res.status,
          healthStatus: status
        })
      }
    }
    if (status === "degraded") {
      const degradedChecks = extractDegradedChecks(body)
      return {
        state: "degraded",
        degradedChecks,
        diagnostics: buildReadinessDiagnostics({
          state: "degraded",
          healthUrl,
          degradedChecks,
          httpStatus: res.status,
          healthStatus: status
        })
      }
    }
    return {
      state: "blocked",
      diagnostics: buildReadinessDiagnostics({
        state: "blocked",
        healthUrl,
        httpStatus: res.status,
        // Empty status means the health JSON did not include a usable status field.
        healthStatus: status !== "" ? status : undefined
      })
    }
  } catch (err) {
    return {
      state: "blocked",
      diagnostics: buildReadinessDiagnostics({
        state: "blocked",
        healthUrl,
        errorMessage: err instanceof Error ? err.message : "Health request failed."
      })
    }
  }
}

// Module-level readiness probe cache. It exists so the app shell can warm the
// /health round trip in parallel with auth resolution (perf remediation W1:
// single gate phase instead of auth → health serial phases) and so a freshly
// mounted gate reuses the warmed in-flight/ready probe instead of issuing a
// duplicate request. Only the first attempt of a gate effect run consults the
// cache; retries always hit the network. Each effect run resets the probe so
// sequential gate sessions (remounts, URL changes) re-check as before.
type ReadinessProbeEntry = {
  url: string
  startedAt: number
  promise: Promise<ReadinessResult>
  settled: boolean
  result: ReadinessResult | null
}

const READINESS_PROBE_REUSE_TTL_MS = 15_000
let readinessProbeEntry: ReadinessProbeEntry | null = null

const resetReadinessProbe = (): void => {
  readinessProbeEntry = null
}

const probeReadinessHealth = (
  healthUrl: string,
  options?: { force?: boolean; warmOnly?: boolean }
): Promise<ReadinessResult> => {
  const cached = readinessProbeEntry
  if (cached && cached.url === healthUrl) {
    // Warm-up calls only join or seed a probe; they never refresh an existing
    // one, so repeated auth refreshes cannot spam /health requests.
    if (options?.warmOnly) return cached.promise
    if (!options?.force) {
      if (!cached.settled) return cached.promise
      if (
        cached.result?.state === "ready" &&
        Date.now() - cached.startedAt <= READINESS_PROBE_REUSE_TTL_MS
      ) {
        return cached.promise
      }
    }
  }

  const entry: ReadinessProbeEntry = {
    url: healthUrl,
    startedAt: Date.now(),
    promise: checkHealth(healthUrl),
    settled: false,
    result: null
  }
  entry.promise = entry.promise.then((result) => {
    entry.settled = true
    entry.result = result
    return result
  })
  readinessProbeEntry = entry
  return entry.promise
}

/**
 * Start (or join) the readiness /health probe before the gate mounts.
 *
 * Called by the app shell as soon as the configured auth state is read so the
 * health round trip overlaps the multi-user auth/me validation instead of
 * running serially after it. Results are memoized per URL; see
 * `probeReadinessHealth`.
 */
export function warmServerReadinessHealth(
  configuredServerUrl?: string | null
): void {
  if (typeof window === "undefined") return
  // Read the (possibly still hydrating) connection store without requiring the
  // zustand API shape in test doubles.
  const getState = (
    useConnectionStore as unknown as {
      getState?: () => { state: { serverUrl: string | null } }
    }
  ).getState
  let storedServerUrl: string | null = null
  try {
    storedServerUrl = getState?.().state.serverUrl ?? null
  } catch {
    storedServerUrl = null
  }
  const healthUrl = resolveReadinessHealthUrl({
    configuredServerUrl: storedServerUrl || configuredServerUrl,
    env: _env,
    pageOrigin: window.location.origin
  })
  void probeReadinessHealth(healthUrl, { warmOnly: true })
}

function emitServerReadinessState(detail: ServerReadinessPublishedState) {
  if (typeof window === "undefined") return
  window.__tldwServerReadinessState = detail
  window.dispatchEvent(
    new CustomEvent(SERVER_READINESS_STATE_EVENT, {
      detail
    })
  )
}

function navigateTo(path: string): void {
  if (typeof window === "undefined") return
  window.location.assign(path)
}

function ReadinessRecoveryPanel({
  diagnostics,
  healthUrl,
  onRetry,
}: {
  diagnostics: ServerReadinessPublishedState | null
  healthUrl: string
  onRetry: () => void
}) {
  const panelDiagnostics: StatePanelDiagnostic[] = [
    {
      label: "Health endpoint",
      value: healthUrl,
      code: true,
    },
    {
      label: "Waited",
      value: `${Math.round(MAX_WAIT_MS / 1000)} seconds`,
    },
  ]

  if (diagnostics?.httpStatus != null) {
    panelDiagnostics.push({
      label: "HTTP status",
      value: String(diagnostics.httpStatus),
    })
  }
  if (diagnostics?.healthStatus) {
    panelDiagnostics.push({
      label: "Health status",
      value: diagnostics.healthStatus,
    })
  }
  if (diagnostics?.degradedChecks.length) {
    panelDiagnostics.push({
      label: "Degraded checks",
      value: diagnostics.degradedChecks.join(", "),
    })
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-bg px-4 py-10 text-text">
      <StatePanel
        state="unavailable"
        title="Backend readiness check failed"
        message="The WebUI could not confirm that the tldw server is ready. You can retry the health check, inspect diagnostics, or update server settings before continuing."
        diagnostics={panelDiagnostics}
        primaryAction={{ label: "Retry", onClick: onRetry }}
        secondaryActions={[
          {
            label: "Health & diagnostics",
            onClick: () => navigateTo("/settings/health"),
          },
          {
            label: "Server settings",
            onClick: () => navigateTo("/settings/tldw"),
          },
        ]}
        role="alert"
        aria-live="assertive"
        className="w-full max-w-3xl"
        data-testid="server-readiness-recovery"
      />
    </main>
  )
}

export const ServerReadinessGate: React.FC<{
  children: React.ReactNode
  allowDegraded?: boolean
  bypass?: boolean
  configuredServerUrl?: string | null
  /**
   * Render children behind the gate instead of blocking on health (perf
   * remediation W1). Intended for established sessions where a successful
   * auth/me round trip already proved server reachability: health failures
   * surface as a non-blocking reconnect banner rather than a full-screen
   * spinner/recovery panel. Cold starts without a session keep the fully
   * blocking behavior (default).
   */
  nonBlocking?: boolean
}> = ({
  children,
  allowDegraded = false,
  bypass = false,
  configuredServerUrl = null,
  nonBlocking = false
}) => {
  const storedServerUrl = useConnectionStore((s) => s.state.serverUrl)
  const effectiveServerUrl = storedServerUrl || configuredServerUrl
  const offlineBypassEnabled = shouldBypassReadinessForOffline()
  const [gate, setGate] = React.useState<GateState>(() =>
    offlineBypassEnabled ? "ready" : "checking"
  )
  const [degradedChecks, setDegradedChecks] = React.useState<string[]>([])
  const [lastReadinessState, setLastReadinessState] =
    React.useState<ServerReadinessPublishedState | null>(null)
  const [retryVersion, setRetryVersion] = React.useState(0)
  const pageOrigin =
    typeof window !== "undefined" ? window.location.origin : undefined
  const healthUrl = React.useMemo(
    () => {
      if (bypass || offlineBypassEnabled) return HEALTH_PATH
      return resolveReadinessHealthUrl({
        configuredServerUrl: effectiveServerUrl,
        env: _env,
        pageOrigin
      })
    },
    [bypass, effectiveServerUrl, offlineBypassEnabled, pageOrigin]
  )

  const retryNow = React.useCallback(() => {
    setRetryVersion((version) => version + 1)
  }, [])

  React.useEffect(() => {
    if (typeof window === "undefined") return
    if (bypass) {
      setGate((current) => (current === "ready" ? current : "checking"))
      return
    }
    if (offlineBypassEnabled) {
      setGate("ready")
      return
    }

    setGate((current) => (current === "ready" ? current : "checking"))
    setDegradedChecks([])
    setLastReadinessState(null)

    let cancelled = false
    let retryTimer: ReturnType<typeof setTimeout> | undefined
    let deadlineTimer: ReturnType<typeof setTimeout> | undefined
    const deadline = Date.now() + MAX_WAIT_MS
    deadlineTimer = setTimeout(() => {
      if (!cancelled) {
        setGate("timeout")
      }
    }, MAX_WAIT_MS)

    // Only the first attempt of a run may reuse a warmed probe (app-shell
    // preflight or a still-settling previous request); retries are always live.
    let isFirstAttempt = true
    const attempt = async () => {
      const result = await probeReadinessHealth(healthUrl, {
        force: !isFirstAttempt
      })
      isFirstAttempt = false
      if (cancelled) return
      setLastReadinessState(result.diagnostics)

      if (result.state === "ready") {
        if (deadlineTimer) clearTimeout(deadlineTimer)
        setGate("ready")
        return
      }

      if (result.state === "degraded" && (allowDegraded || nonBlocking)) {
        if (deadlineTimer) clearTimeout(deadlineTimer)
        setDegradedChecks(result.degradedChecks)
        setGate("degraded")
        return
      }

      if (Date.now() >= deadline) {
        setGate("timeout")
        return
      }

      setGate("waiting")
      retryTimer = setTimeout(() => {
        if (!cancelled) void attempt()
      }, RETRY_INTERVAL_MS)
    }

    void attempt()

    return () => {
      cancelled = true
      if (retryTimer) clearTimeout(retryTimer)
      if (deadlineTimer) clearTimeout(deadlineTimer)
      resetReadinessProbe()
    }
  }, [allowDegraded, bypass, healthUrl, nonBlocking, offlineBypassEnabled, retryVersion])

  // Non-blocking sessions stop retrying at the deadline; give them one free
  // automatic retry when connectivity returns so the banner can clear without a
  // manual action. This deliberately adds no poller of its own.
  React.useEffect(() => {
    if (!nonBlocking || bypass || gate !== "timeout") return
    if (typeof window === "undefined") return
    const handleOnline = () => retryNow()
    window.addEventListener("online", handleOnline)
    return () => window.removeEventListener("online", handleOnline)
  }, [bypass, gate, nonBlocking, retryNow])

  React.useEffect(() => {
    if (typeof window === "undefined" || bypass) return
    const state =
      gate === "ready"
        ? "ready"
        : gate === "degraded"
          ? "degraded"
          : gate === "timeout"
            ? "blocked"
            : null
    if (!state) return

    const emitTimer = window.setTimeout(() => {
      const emittedDegradedChecks =
        state === "degraded"
          ? degradedChecks
          : state === "blocked"
            ? (lastReadinessState?.degradedChecks ?? [])
            : []
      emitServerReadinessState({
        ...(lastReadinessState ??
          buildReadinessDiagnostics({
            state,
            healthUrl,
            degradedChecks: emittedDegradedChecks
          })),
        state,
        degradedChecks: emittedDegradedChecks
      })
    }, 0)

    return () => {
      window.clearTimeout(emitTimer)
    }
  }, [bypass, degradedChecks, gate, healthUrl, lastReadinessState])

  if (bypass || gate === "ready") {
    return <>{children}</>
  }

  if (nonBlocking) {
    // Established session: the auth round trip already proved reachability, so
    // health problems surface as banners around live content rather than a
    // blocking spinner/recovery panel.
    if (gate === "degraded") {
      return (
        <div
          data-testid="server-readiness-degraded-shell"
          className="server-readiness-degraded-shell"
        >
          <ServerHealthWarningBanner degradedChecks={degradedChecks} />
          <div className="server-readiness-degraded-content">
            {children}
          </div>
        </div>
      )
    }

    if (gate === "waiting" || gate === "timeout") {
      return (
        <div
          data-testid="server-readiness-nonblocking-shell"
          className="server-readiness-degraded-shell"
        >
          <ServerReconnectBanner
            exhausted={gate === "timeout"}
            onRetry={retryNow}
          />
          <div className="server-readiness-degraded-content">
            {children}
          </div>
        </div>
      )
    }

    // Initial check still in flight: render children with no banner.
    return <>{children}</>
  }

  if (gate === "degraded") {
    return (
      <div
        data-testid="server-readiness-degraded-shell"
        className="server-readiness-degraded-shell"
      >
        <ServerHealthWarningBanner degradedChecks={degradedChecks} />
        <div className="server-readiness-degraded-content">
          {children}
        </div>
      </div>
    )
  }

  if (gate === "timeout") {
    return (
      <ReadinessRecoveryPanel
        diagnostics={lastReadinessState}
        healthUrl={healthUrl}
        onRetry={retryNow}
      />
    )
  }

  const isRetrying = gate === "waiting"
  const state = isRetrying ? "retrying" : "loading"
  const title = isRetrying ? "Retrying server readiness" : "Checking server readiness"
  const message = isRetrying
    ? "The WebUI is retrying the health check before opening the app."
    : "The WebUI is checking the API health endpoint before opening the app."

  return (
    <main
      className="flex min-h-screen items-center justify-center bg-bg px-4 py-10 text-text"
      role="status"
      aria-live="polite"
    >
      <StatePanel
        state={state}
        title={title}
        message={message}
        primaryAction={{ label: "Waiting", disabled: true }}
        className="w-full max-w-lg"
      />
    </main>
  )
}

export default ServerReadinessGate
