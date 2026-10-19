import { tldwClient } from "@/services/tldw/TldwApiClient"
import { adminKeys } from "@/services/tldw/adminQueries"
import type { AdminQuerySession } from "./AdminQueryProvider"

/**
 * Optional live signals for the Admin Operations overview (#2876): one cheap
 * request per module with a hard timeout, so the overview can answer "is
 * anything wrong?" at a glance while degrading to the static map whenever a
 * signal is unavailable.
 *
 * Since B-S4/F11 the probes run through the shared admin query cache with
 * the same keys the dashboards use: revisiting the overview inside the
 * stale window is served from cache instead of re-probing every module on
 * every visit.
 */
export type AdminModuleSignalState =
  | "healthy"
  | "attention"
  | "unavailable"
  | "off"

export interface AdminModuleSignal {
  state: AdminModuleSignalState
  detail: string
}

const SIGNAL_TIMEOUT_MS = 4000

/** Race a signal request against the shared signal timeout. Exported for the
 *  first-steps checklist, which probes the same cheap admin endpoints. */
export const withSignalTimeout = async <T>(work: Promise<T>): Promise<T> => {
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

/**
 * Read a signal endpoint through the shared admin query cache. Concurrent
 * and repeated probes inside the stale window reuse one request; the signal
 * timeout still races each individual await so a hung probe cannot block
 * the overview.
 */
const adminSignalFetch = <T>(
  session: AdminQuerySession,
  queryKey: readonly unknown[],
  fetch: () => Promise<T>
): Promise<T> =>
  session.client.fetchQuery({
    queryKey,
    queryFn: fetch
  })

const plural = (count: number, noun: string): string =>
  `${count} ${noun}${count === 1 ? "" : "s"}`

const signalFetchers = (session: AdminQuerySession): Record<string, () => Promise<AdminModuleSignal>> => ({
  "/admin/server": async () => {
    const stats = await withSignalTimeout(
      adminSignalFetch(session, adminKeys.systemStats, () => tldwClient.getSystemStats(session.requestOptions()))
    )
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
    const status = await withSignalTimeout(
      adminSignalFetch(session, adminKeys.securityAlertStatus, () =>
        tldwClient.getSecurityAlertStatus(session.requestOptions())
      )
    )
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
    const result = await withSignalTimeout(
      adminSignalFetch(session, adminKeys.backups, () => tldwClient.listBackups(undefined, session.requestOptions()))
    )
    const backups = Array.isArray(result)
      ? result
      : (result?.backups ?? result?.items ?? [])
    if (!Array.isArray(backups) || backups.length === 0) {
      return { state: "attention", detail: "No backups yet" }
    }
    return { state: "healthy", detail: plural(backups.length, "backup") }
  },
  "/admin/llamacpp": async () => {
    const status = await withSignalTimeout(
      adminSignalFetch(session, adminKeys.llamacppStatus, () =>
        tldwClient.getLlamacppStatus(session.requestOptions())
      )
    )
    const state = String(status?.state ?? status?.status ?? "").toLowerCase()
    return state === "running"
      ? { state: "healthy", detail: "Runtime running" }
      : { state: "attention", detail: "Runtime stopped" }
  },
  "/admin/mlx": async () => {
    const status = await withSignalTimeout(
      adminSignalFetch(session, adminKeys.mlxStatus, () => tldwClient.getMlxStatus(session.requestOptions()))
    )
    return status?.active
      ? { state: "healthy", detail: "Model loaded" }
      : { state: "attention", detail: "No model loaded" }
  },
  "/admin/rate-limiting": async () => {
    const coverage = await withSignalTimeout(
      adminSignalFetch(session, adminKeys.governorCoverage, () =>
        tldwClient.getGovernorCoverage(undefined, session.requestOptions())
      )
    )
    const pct = coverage?.coverage_pct
    return typeof pct === "number"
      ? {
          state: pct >= 50 ? "healthy" : "attention",
          detail: `${pct}% endpoint coverage`
        }
      : { state: "healthy", detail: "Governor reachable" }
  }
})

/**
 * A backend that answers "this module is not configured/enabled" is off on
 * purpose - render it as a neutral "off" signal, not as an outage (#2894).
 */
const isNotConfiguredError = (reason: unknown): boolean => {
  const message =
    reason instanceof Error ? reason.message : String(reason ?? "")
  return /not configured|not enabled|is disabled/i.test(message)
}

// One log line per route per session: unavailable signals are expected on
// servers that leave optional modules off, and the overview reloads on every
// visit (#2896).
const loggedSignalFailures = new Set<string>()

export const loadAdminModuleSignals = async (session: AdminQuerySession): Promise<
  Record<string, AdminModuleSignal>
> => {
  if (!session.scope.isCurrent()) return {}
  const fetchers = signalFetchers(session)
  const routes = Object.keys(fetchers)
  const results = await Promise.allSettled(
    routes.map((route) => fetchers[route]())
  )
  if (!session.scope.isCurrent()) return {}
  const signals: Record<string, AdminModuleSignal> = {}
  routes.forEach((route, index) => {
    const result = results[index]
    if (result.status === "fulfilled") {
      signals[route] = result.value
      return
    }
    if (isNotConfiguredError(result.reason)) {
      signals[route] = { state: "off", detail: "Not configured" }
      return
    }
    if (!loggedSignalFailures.has(route)) {
      loggedSignalFailures.add(route)
      console.warn(
        `[admin-signals] signal for ${route} unavailable:`,
        result.reason
      )
    }
    signals[route] = { state: "unavailable", detail: "Status unavailable" }
  })
  return signals
}
