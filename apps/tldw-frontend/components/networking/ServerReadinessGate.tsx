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
type ReadinessResult =
  | { state: "ready" }
  | { state: "degraded"; degradedChecks: string[] }
  | { state: "blocked" }

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

async function checkHealth(): Promise<boolean> {
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

async function checkHealth(): Promise<ReadinessResult> {
  try {
    const res = await fetch(HEALTH_URL, {
      method: "GET",
      signal: AbortSignal.timeout(3000)
    })
    if (!ENTERABLE_HTTP_STATUSES.has(res.status)) {
      return { state: "blocked" }
    }
    const body = await res.json()
    const status =
      typeof body?.status === "string" ? body.status.toLowerCase() : ""
    if (READY_HEALTH_STATUSES.has(status)) {
      return { state: "ready" }
    }
    if (status === "degraded") {
      return {
        state: "degraded",
        degradedChecks: extractDegradedChecks(body)
      }
    }
    return { state: "blocked" }
  } catch {
    return { state: "blocked" }
  }
}

function emitServerReadinessState(
  state: "ready" | "degraded" | "blocked",
  degradedChecks: string[] = []
) {
  if (typeof window === "undefined") return
  window.dispatchEvent(
    new CustomEvent(SERVER_READINESS_STATE_EVENT, {
      detail: { state, degradedChecks }
    })
  )
}

export const ServerReadinessGate: React.FC<{
  children: React.ReactNode
  allowDegraded?: boolean
  bypass?: boolean
}> = ({ children, bypass = false }) => {
  const [gate, setGate] = React.useState<GateState>(() =>
    shouldBypassReadinessForOffline() ? "ready" : "checking"
  )

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

    let cancelled = false
    let retryTimer: number | undefined
    const deadline = Date.now() + MAX_WAIT_MS

    const attempt = async () => {
      const result = await checkHealth()
      if (cancelled) return

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
      emitServerReadinessState(
        state,
        state === "degraded" ? degradedChecks : []
      )
    }, 0)

    return () => {
      window.clearTimeout(emitTimer)
    }
  }, [bypass, degradedChecks, gate])

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
