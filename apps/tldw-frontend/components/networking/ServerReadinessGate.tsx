import React from "react"

import { resolvePublicApiOrigin, type DeploymentEnv } from "@web/lib/api-base"
import { useConnectionStore } from "@tldw/ui/store/connection"
import {
  StatePanel,
  type StatePanelDiagnostic
} from "@tldw/ui/components/ui/state"

const _env: DeploymentEnv = {
  NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE,
  NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL
}

const MAX_WAIT_MS = 15_000
const HEALTH_CHECK_TIMEOUT_MS = 3_000
const RETRY_INTERVAL_MS = 2_000
const OFFLINE_BYPASS_KEYS = ["__tldw_allow_offline", "__tldw_test_bypass"] as const

type GateState = "checking" | "ready" | "waiting" | "timeout"
type HealthCheckResult =
  | { ok: true }
  | { ok: false; reason: "failed" | "stalled" }

const trimTrailingSlash = (value: string): string => value.replace(/\/+$/, "")

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
    return `${trimTrailingSlash(configured)}/api/v1/health`
  }

  try {
    const fallbackOrigin =
      typeof pageOrigin === "string"
        ? resolvePublicApiOrigin(env, pageOrigin)
        : resolvePublicApiOrigin(env)
    return `${trimTrailingSlash(fallbackOrigin)}/api/v1/health`
  } catch {
    return "/api/v1/health"
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

function navigateTo(path: string): void {
  if (typeof window === "undefined") return
  window.location.assign(path)
}

async function checkHealth(healthUrl: string): Promise<HealthCheckResult> {
  const controller = new AbortController()
  let timeoutId: ReturnType<typeof setTimeout> | undefined
  const stalled = new Promise<HealthCheckResult>((resolve) => {
    timeoutId = setTimeout(() => {
      controller.abort()
      resolve({ ok: false, reason: "stalled" })
    }, HEALTH_CHECK_TIMEOUT_MS)
  })
  const request = (async (): Promise<HealthCheckResult> => {
    try {
      const res = await fetch(healthUrl, {
        method: "GET",
        signal: controller.signal
      })
      if (!res.ok) return { ok: false, reason: "failed" }
      const body = await res.json()
      return body.status === "ok" || body.status === "healthy"
        ? { ok: true }
        : { ok: false, reason: "failed" }
    } catch {
      return controller.signal.aborted
        ? { ok: false, reason: "stalled" }
        : { ok: false, reason: "failed" }
    }
  })()

  try {
    return await Promise.race([request, stalled])
  } finally {
    if (timeoutId) {
      clearTimeout(timeoutId)
    }
  }
}

function ReadinessRecoveryPanel({
  healthUrl,
  onRetry
}: {
  healthUrl: string
  onRetry: () => void
}) {
  const diagnostics: StatePanelDiagnostic[] = [
    {
      label: "Health endpoint",
      value: healthUrl,
      code: true
    },
    {
      label: "Waited",
      value: `${Math.round(MAX_WAIT_MS / 1000)} seconds`
    }
  ]

  return (
    <main className="flex min-h-screen items-center justify-center bg-bg px-4 py-10 text-text">
      <StatePanel
        state="unavailable"
        title="Backend readiness check failed"
        message="The WebUI could not confirm that the tldw server is ready. You can retry the health check, inspect diagnostics, or update server settings before continuing."
        diagnostics={diagnostics}
        primaryAction={{ label: "Retry", onClick: onRetry }}
        secondaryActions={[
          {
            label: "Health & diagnostics",
            onClick: () => navigateTo("/settings/health")
          },
          {
            label: "Server settings",
            onClick: () => navigateTo("/settings/tldw")
          }
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
  bypass?: boolean
}> = ({ children, bypass = false }) => {
  const configuredServerUrl = useConnectionStore((s) => s.state.serverUrl)
  const lastConfigUpdatedAt = useConnectionStore(
    (s) => s.state.lastConfigUpdatedAt
  )
  const [gate, setGate] = React.useState<GateState>(() =>
    shouldBypassReadinessForOffline() ? "ready" : "checking"
  )
  const [retryVersion, setRetryVersion] = React.useState(0)
  const pageOrigin =
    typeof window !== "undefined" ? window.location.origin : undefined
  const healthUrl = React.useMemo(
    () =>
      resolveReadinessHealthUrl({
        configuredServerUrl,
        env: _env,
        pageOrigin
      }),
    [configuredServerUrl, pageOrigin]
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
    if (shouldBypassReadinessForOffline()) {
      setGate("ready")
      return
    }

    setGate((current) => (current === "ready" ? current : "checking"))

    let cancelled = false
    let retryTimer: number | undefined
    let deadlineTimer: number | undefined
    const deadline = Date.now() + MAX_WAIT_MS
    deadlineTimer = window.setTimeout(() => {
      if (!cancelled) {
        setGate("timeout")
      }
    }, MAX_WAIT_MS)

    const attempt = async () => {
      const health = await checkHealth(healthUrl)
      if (cancelled) return

      if (health.ok) {
        if (deadlineTimer) {
          window.clearTimeout(deadlineTimer)
        }
        setGate("ready")
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
      if (deadlineTimer) window.clearTimeout(deadlineTimer)
    }
  }, [bypass, healthUrl, lastConfigUpdatedAt, retryVersion])

  if (bypass || gate === "ready") {
    return <>{children}</>
  }

  if (gate === "timeout") {
    return <ReadinessRecoveryPanel healthUrl={healthUrl} onRetry={retryNow} />
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
