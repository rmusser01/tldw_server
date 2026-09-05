import React from "react"
import { PageShell } from "@/components/Common/PageShell"
import { useConnectionState } from "@/hooks/useConnectionState"
import {
  ADMIN_MODULE_GROUPS,
  ADMIN_MODULES,
  type AdminModuleGroup
} from "./admin-modules"
import {
  loadAdminModuleSignals,
  type AdminModuleSignal
} from "./admin-module-signals"
import {
  loadAdminFirstSteps,
  type AdminFirstStep
} from "./admin-first-steps"

const groupedModules = ADMIN_MODULE_GROUPS.map((group: AdminModuleGroup) => ({
  group,
  modules: ADMIN_MODULES.filter((module) => module.group === group)
})).filter((entry) => entry.modules.length > 0)

const SIGNAL_DOT_COLOR: Record<AdminModuleSignal["state"], string> = {
  healthy: "var(--state-ready, #2f9e6e)",
  attention: "var(--state-degraded, #d98324)",
  unavailable: "var(--state-unavailable, #8a8fa3)",
  // "off" is a deliberate state (module not configured), quieter than an
  // outage so it never reads as something to fix (#2894).
  off: "var(--state-off, #5c6170)"
}

const ModuleSignalBadge: React.FC<{
  signal: AdminModuleSignal | undefined
  route: string
}> = ({ signal, route }) => {
  if (!signal) return null
  return (
    <p
      className="mt-2 flex items-center gap-1.5 text-xs text-text-muted"
      data-testid="admin-module-signal"
    >
      <span
        aria-hidden="true"
        className="inline-block h-2 w-2 shrink-0 rounded-full"
        style={{ backgroundColor: SIGNAL_DOT_COLOR[signal.state] }}
      />
      {/* The status is the reason to visit the module - make it the link. */}
      <a className="hover:text-text hover:underline" href={route}>
        {signal.detail}
      </a>
    </p>
  )
}

export const AdminOperationsOverviewPage: React.FC = () => {
  const [signals, setSignals] = React.useState<
    Record<string, AdminModuleSignal>
  >({})
  const { serverUrl } = useConnectionState()
  const [firstSteps, setFirstSteps] = React.useState<AdminFirstStep[]>([])
  // Dismissal is scoped per server, like the resume-setup banner: a new
  // connection gets its own first-session checklist.
  const firstStepsDismissKey = `__tldw_admin_first_steps_dismissed::${
    serverUrl || "unconfigured"
  }`
  const [dismissedKeys, setDismissedKeys] = React.useState<ReadonlySet<string>>(
    () => new Set()
  )
  const firstStepsDismissed = React.useMemo(() => {
    if (dismissedKeys.has(firstStepsDismissKey)) return true
    if (typeof window === "undefined") return false
    try {
      return window.localStorage.getItem(firstStepsDismissKey) === "1"
    } catch {
      return false
    }
  }, [firstStepsDismissKey, dismissedKeys])
  const dismissFirstSteps = () => {
    if (typeof window !== "undefined") {
      try {
        window.localStorage.setItem(firstStepsDismissKey, "1")
      } catch {
        // Dismissal is best-effort frontend-only state.
      }
    }
    setDismissedKeys((prev) => new Set(prev).add(firstStepsDismissKey))
  }

  React.useEffect(() => {
    let cancelled = false
    void loadAdminModuleSignals().then((loaded) => {
      if (!cancelled) setSignals(loaded)
    })
    void loadAdminFirstSteps().then((steps) => {
      if (!cancelled) setFirstSteps(steps)
    })
    return () => {
      cancelled = true
    }
  }, [])

  // The card earns its place only while something is left to do; a finished
  // (or dismissed, or unconnected) checklist renders nothing.
  const showFirstSteps =
    Boolean(serverUrl) &&
    !firstStepsDismissed &&
    firstSteps.some((step) => !step.done)

  return (
    <PageShell maxWidthClassName="max-w-6xl" className="py-8">
      {!serverUrl ? (
        <div
          role="status"
          data-testid="admin-not-connected-banner"
          className="mb-6 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border bg-surface px-4 py-3"
        >
          <p className="m-0 text-sm text-text">
            Not connected to a tldw server, so module signals are unavailable.
            Connect to a server to administer it.
          </p>
          <a
            className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-white hover:bg-primaryStrong"
            href="/setup"
          >
            Connect
          </a>
        </div>
      ) : null}
      <header className="space-y-3">
        <p className="text-sm font-semibold uppercase tracking-wide text-text-muted">
          Admin
        </p>
        <div>
          <h1 className="text-3xl font-semibold text-text">Admin Operations</h1>
          <p className="mt-2 max-w-3xl text-sm text-text-muted">
            Everything you can administer on this server, in one place. Open a
            module to load its live data.
          </p>
        </div>
      </header>

      {showFirstSteps ? (
        <section
          aria-labelledby="admin-first-steps-title"
          data-testid="admin-first-steps"
          className="mt-6 rounded-lg border border-border bg-surface p-4 shadow-sm"
        >
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h2
                id="admin-first-steps-title"
                className="text-sm font-semibold uppercase tracking-wide text-text-muted"
              >
                First steps
              </h2>
              <p className="mt-1 text-sm text-text-muted">
                A few one-time setups most servers want before day-to-day
                operation.
              </p>
            </div>
            <button
              type="button"
              className="rounded-md px-2.5 py-1 text-sm text-text-muted hover:bg-surface2"
              onClick={dismissFirstSteps}
            >
              Dismiss
            </button>
          </div>
          <ul className="mt-3 space-y-1.5">
            {firstSteps.map((step) => (
              <li
                key={step.key}
                className="flex items-center gap-2 text-sm"
                data-testid={`admin-first-step-${step.key}`}
              >
                <span
                  aria-hidden="true"
                  className={
                    step.done
                      ? "text-[color:var(--state-ready,#2f9e6e)]"
                      : "text-text-muted"
                  }
                >
                  {step.done ? "☑" : "☐"}
                </span>
                {step.done ? (
                  <span className="text-text-muted line-through">
                    {step.label}
                  </span>
                ) : (
                  <a className="text-text hover:underline" href={step.route}>
                    {step.label}
                  </a>
                )}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <div className="mt-8 space-y-8" data-testid="admin-operations-modules">
        {groupedModules.map(({ group, modules }) => (
          <section key={group} aria-labelledby={`admin-group-${group}`}>
            <h2
              id={`admin-group-${group}`}
              className="text-sm font-semibold uppercase tracking-wide text-text-muted"
            >
              {group}
            </h2>
            <div className="mt-3 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {modules.map((module) => (
                <article
                  key={module.route}
                  className="rounded-lg border border-border bg-surface p-4 shadow-sm"
                  data-testid={`admin-module-${module.route}`}
                >
                  <h3 className="text-base font-semibold text-text">
                    <a className="hover:text-primary" href={module.route}>
                      {module.label}
                    </a>
                    {module.comingSoon ? (
                      <span className="ml-2 rounded-full border border-border px-2 py-0.5 align-middle text-[10px] font-medium uppercase tracking-wide text-text-muted">
                        Coming soon
                      </span>
                    ) : null}
                  </h3>
                  <p className="mt-1.5 text-sm text-text-muted">
                    {module.description}
                  </p>
                  <ModuleSignalBadge
                    signal={signals[module.route]}
                    route={module.route}
                  />
                </article>
              ))}
            </div>
          </section>
        ))}
      </div>
    </PageShell>
  )
}

export default AdminOperationsOverviewPage
