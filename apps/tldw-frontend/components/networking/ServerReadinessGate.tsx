import React from "react"

import { resolvePublicApiOrigin, type DeploymentEnv } from "@web/lib/api-base"
import { StatePanel } from "@tldw/ui/components/ui/state"
import { ServerHealthWarningBanner } from "./ServerHealthWarningBanner"

const _env: DeploymentEnv = {
  NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE,
  NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL
}

const _origin =
  typeof window !== "undefined"
    ? resolvePublicApiOrigin(_env, window.location.origin)
    : resolvePublicApiOrigin(_env)

const HEALTH_URL = `${_origin}/api/v1/health`
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
  degradedChecks = [],
  httpStatus,
  healthStatus,
  errorMessage
}: {
  state: "ready" | "degraded" | "blocked"
  degradedChecks?: string[]
  httpStatus?: number
  healthStatus?: string
  errorMessage?: string
}): ServerReadinessPublishedState {
  return {
    state,
    degradedChecks,
    healthUrl: HEALTH_URL,
    httpStatus,
    healthStatus,
    errorMessage,
    checkedAt: new Date().toISOString()
  }
}

async function checkHealth(): Promise<ReadinessResult> {
  try {
    const res = await fetch(HEALTH_URL, {
      method: "GET",
      signal: AbortSignal.timeout(3000)
    })
    let body: unknown
    let healthStatus: string | undefined
    try {
      body = await res.json()
      healthStatus = extractHealthStatus(body)
    } catch (err) {
      return {
        state: "blocked",
        diagnostics: buildReadinessDiagnostics({
          state: "blocked",
          httpStatus: res.status,
          errorMessage: err instanceof Error ? err.message : "Could not parse health response."
        })
      }
    }
    if (!ENTERABLE_HTTP_STATUSES.has(res.status)) {
      return {
        state: "blocked",
        diagnostics: buildReadinessDiagnostics({
          state: "blocked",
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
        httpStatus: res.status,
        healthStatus: status || undefined
      })
    }
  } catch (err) {
    return {
      state: "blocked",
      diagnostics: buildReadinessDiagnostics({
        state: "blocked",
        errorMessage: err instanceof Error ? err.message : "Health request failed."
      })
    }
  }
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

export const ServerReadinessGate: React.FC<{
  children: React.ReactNode
  allowDegraded?: boolean
  bypass?: boolean
}> = ({ children, allowDegraded = false, bypass = false }) => {
  const [gate, setGate] = React.useState<GateState>(() =>
    shouldBypassReadinessForOffline() ? "ready" : "checking"
  )
  const [degradedChecks, setDegradedChecks] = React.useState<string[]>([])
  const [lastReadinessState, setLastReadinessState] =
    React.useState<ServerReadinessPublishedState | null>(null)

  React.useEffect(() => {
    if (typeof window === "undefined") return
    if (bypass) {
      setGate((current) => (current === "ready" ? current : "checking"))
      return
    }
    if (shouldBypassReadinessForOffline()) {
      setGate("ready")
      return
    }

    setGate((current) => (current === "ready" ? current : "checking"))
    setDegradedChecks([])
    setLastReadinessState(null)

    let cancelled = false
    let retryTimer: number | undefined
    const deadline = Date.now() + MAX_WAIT_MS

    const attempt = async () => {
      const result = await checkHealth()
      if (cancelled) return
      setLastReadinessState(result.diagnostics)

      if (result.state === "ready") {
        setGate("ready")
        return
      }

      if (result.state === "degraded" && allowDegraded) {
        setDegradedChecks(result.degradedChecks)
        setGate("degraded")
        return
      }

      if (Date.now() >= deadline) {
        setGate("timeout")
        return
      }

      setGate("waiting")
      retryTimer = window.setTimeout(() => {
        if (!cancelled) void attempt()
      }, RETRY_INTERVAL_MS)
    }

    void attempt()

    return () => {
      cancelled = true
      if (retryTimer) window.clearTimeout(retryTimer)
    }
  }, [allowDegraded, bypass])

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
      emitServerReadinessState({
        ...(lastReadinessState ??
          buildReadinessDiagnostics({
            state,
            degradedChecks: state === "degraded" ? degradedChecks : []
          })),
        state,
        degradedChecks: state === "degraded" ? degradedChecks : []
      })
    }, 0)

    return () => {
      window.clearTimeout(emitTimer)
    }
  }, [bypass, degradedChecks, gate, lastReadinessState])

  if (bypass || gate === "ready" || gate === "timeout") {
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
