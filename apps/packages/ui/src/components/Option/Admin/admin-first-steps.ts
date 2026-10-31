import { tldwClient } from "@/services/tldw/TldwApiClient"
import { withSignalTimeout } from "./admin-module-signals"

/**
 * First-session checklist for the Admin Operations overview (#2899 I6):
 * three cheap probes answering "has this server been set up like an
 * operator's server yet?". Each item names the task, whether it is done,
 * and where to do it. A probe that fails resolves to null and its item is
 * simply not shown - the checklist never blocks or degrades the overview.
 */
export interface AdminFirstStep {
  key: string
  label: string
  done: boolean
  route: string
}

const asList = (value: unknown, ...keys: string[]): unknown[] => {
  if (Array.isArray(value)) return value
  if (value && typeof value === "object") {
    for (const key of keys) {
      const inner = (value as Record<string, unknown>)[key]
      if (Array.isArray(inner)) return inner
    }
  }
  return []
}

const STEP_PROBES: Array<() => Promise<AdminFirstStep>> = [
  async () => {
    const result = await withSignalTimeout(tldwClient.listBackupSchedules())
    const schedules = asList(result, "schedules", "items")
    return {
      key: "backup-schedule",
      label: "Create a backup schedule",
      done: schedules.length > 0,
      route: "/admin/data-ops"
    }
  },
  async () => {
    const result = await withSignalTimeout(tldwClient.listAlertRules())
    const rules = asList(result, "rules", "items")
    return {
      key: "alert-rule",
      label: "Add an alert rule",
      done: rules.length > 0,
      route: "/admin/monitoring"
    }
  },
  async () => {
    const coverage = await withSignalTimeout(
      tldwClient.getGovernorCoverage()
    )
    const pct = coverage?.coverage_pct
    return {
      key: "coverage-review",
      label: "Review unprotected endpoints",
      // There is no completion event for "reviewed"; treat strong coverage
      // as the done state so the item retires once the surface is governed.
      done: typeof pct === "number" ? pct >= 80 : true,
      route: "/admin/rate-limiting"
    }
  }
]

export const loadAdminFirstSteps = async (): Promise<AdminFirstStep[]> => {
  const results = await Promise.allSettled(STEP_PROBES.map((probe) => probe()))
  return results
    .filter(
      (result): result is PromiseFulfilledResult<AdminFirstStep> =>
        result.status === "fulfilled"
    )
    .map((result) => result.value)
}
