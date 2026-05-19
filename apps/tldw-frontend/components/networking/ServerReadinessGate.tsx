import React from "react"

import { resolvePublicApiOrigin, type DeploymentEnv } from "@web/lib/api-base"
import { StatePanel } from "@tldw/ui/components/ui/state"

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

type GateState = "checking" | "ready" | "waiting" | "timeout"

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
    const res = await fetch(HEALTH_URL, {
      method: "GET",
      signal: AbortSignal.timeout(3000)
    })
    if (!res.ok) return false
    const body = await res.json()
    return body.status === "ok" || body.status === "healthy"
  } catch {
    return false
  }
}

export const ServerReadinessGate: React.FC<{
  children: React.ReactNode
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

    let cancelled = false
    let retryTimer: number | undefined
    const deadline = Date.now() + MAX_WAIT_MS

    const attempt = async () => {
      const ok = await checkHealth()
      if (cancelled) return

      if (ok) {
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
    }
  }, [bypass])

  if (bypass || gate === "ready" || gate === "timeout") {
    return <>{children}</>
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
