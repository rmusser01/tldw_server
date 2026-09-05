import { tldwClient } from "@/services/tldw/TldwApiClient"

/**
 * Optional live signals for the Admin Operations overview (#2876): one cheap
 * request per module with a hard timeout, so the overview can answer "is
 * anything wrong?" at a glance while degrading to the static map whenever a
 * signal is unavailable.
 */
export type AdminModuleSignalState = "healthy" | "attention" | "unavailable"

export interface AdminModuleSignal {
  state: AdminModuleSignalState
  detail: string
}

const SIGNAL_TIMEOUT_MS = 4000

const withTimeout = async <T>(work: Promise<T>): Promise<T> => {
  let timer: ReturnType<typeof setTimeout> | undefined
  try {
    return await Promise.race([
      work,
      new Promise<never>((_, reject) => {
        timer = setTimeout(
          () => reject(new Error("signal timeout")),
          SIGNAL_TIMEOUT_MS
        )
      })
    ])
  } finally {
    if (timer) clearTimeout(timer)
  }
}

const plural = (count: number, noun: string): string =>
  `${count} ${noun}${count === 1 ? "" : "s"}`

const SIGNAL_FETCHERS: Record<string, () => Promise<AdminModuleSignal>> = {
  "/admin/server": async () => {
    const stats = await withTimeout(tldwClient.getSystemStats())
    const total = stats?.users?.total
    return {
      state: "healthy",
      detail:
        typeof total === "number" ? plural(total, "user") : "Server reachable"
    }
  },
  "/admin/monitoring": async () => {
    // The alerts/history endpoint is an audit log of alert actions, not a
    // list of open alerts, so the aggregated security alert health is the
    // honest signal here ("ok" | "degraded" | "errors").
    const status = await withTimeout(tldwClient.getSecurityAlertStatus())
    const health = String(status?.health ?? "").toLowerCase()
    if (health === "ok") {
      return { state: "healthy", detail: "Alerting healthy" }
    }
    if (health === "degraded" || health === "errors") {
      return { state: "attention", detail: `Alerting ${health}` }
    }
    return { state: "healthy", detail: "Monitoring reachable" }
  },
  "/admin/data-ops": async () => {
    const result = await withTimeout(tldwClient.listBackups())
    const backups = Array.isArray(result)
      ? result
      : (result?.backups ?? result?.items ?? [])
    if (!Array.isArray(backups) || backups.length === 0) {
      return { state: "attention", detail: "No backups yet" }
    }
    return { state: "healthy", detail: plural(backups.length, "backup") }
  },
  "/admin/llamacpp": async () => {
    const status = await withTimeout(tldwClient.getLlamacppStatus())
    const state = String(status?.state ?? status?.status ?? "").toLowerCase()
    return state === "running"
      ? { state: "healthy", detail: "Runtime running" }
      : { state: "attention", detail: "Runtime stopped" }
  },
  "/admin/mlx": async () => {
    const status = await withTimeout(tldwClient.getMlxStatus())
    return status?.active
      ? { state: "healthy", detail: "Model loaded" }
      : { state: "attention", detail: "No model loaded" }
  },
  "/admin/rate-limiting": async () => {
    const coverage = await withTimeout(tldwClient.getGovernorCoverage())
    const pct = coverage?.coverage_pct
    return typeof pct === "number"
      ? {
          state: pct >= 50 ? "healthy" : "attention",
          detail: `${pct}% endpoint coverage`
        }
      : { state: "healthy", detail: "Governor reachable" }
  }
}

export const loadAdminModuleSignals = async (): Promise<
  Record<string, AdminModuleSignal>
> => {
  const routes = Object.keys(SIGNAL_FETCHERS)
  const results = await Promise.allSettled(
    routes.map((route) => SIGNAL_FETCHERS[route]())
  )
  const signals: Record<string, AdminModuleSignal> = {}
  routes.forEach((route, index) => {
    const result = results[index]
    if (result.status === "fulfilled") {
      signals[route] = result.value
      return
    }
    console.warn(
      `[admin-signals] signal for ${route} unavailable:`,
      result.reason
    )
    signals[route] = { state: "unavailable", detail: "Status unavailable" }
  })
  return signals
}
