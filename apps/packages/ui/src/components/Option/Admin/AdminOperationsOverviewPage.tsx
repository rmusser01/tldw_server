import React from "react"
import {
  getOperationsRouteJob,
  type OperationsRouteJob
} from "@/routes/operations-route-jobs"
import { PageShell } from "@/components/Common/PageShell"

type AdminOverviewModule = {
  route: string
  fallbackLabel: string
  summary: string
  status: "Route ready" | "Needs module configuration"
  statusDescription: string
}

const ADMIN_MODULES: AdminOverviewModule[] = [
  {
    route: "/admin/server",
    fallbackLabel: "Server Admin",
    summary: "Server health, users, roles, storage, sessions, and media budget diagnostics.",
    status: "Route ready",
    statusDescription: "Open the module to load live server health and user data."
  },
  {
    route: "/admin/integrations",
    fallbackLabel: "Workspace Integrations",
    summary: "Workspace integration policy for Slack, Discord, Telegram, and linked actors.",
    status: "Route ready",
    statusDescription: "Open the module to load workspace policy and provider state."
  },
  {
    route: "/admin/sources",
    fallbackLabel: "Admin Sources",
    summary: "Administrative view of ingestion source availability, sync state, and setup.",
    status: "Route ready",
    statusDescription: "Open the module to load source capability and sync state."
  },
  {
    route: "/admin/monitoring",
    fallbackLabel: "Monitoring",
    summary: "Monitoring metrics, alerts, runtime diagnostics, and operations telemetry.",
    status: "Route ready",
    statusDescription: "Open the module to load metrics, alerts, and diagnostic state."
  }
]

const labelForJob = (
  job: OperationsRouteJob | undefined,
  fallbackLabel: string
): string => job?.label ?? fallbackLabel

const ModuleDiagnostics: React.FC<{
  job: OperationsRouteJob | undefined
  module: AdminOverviewModule
}> = ({ job, module }) => {
  return (
    <details className="mt-4 rounded-md border border-border bg-surface2 px-3 py-2 text-xs text-text-muted">
      <summary className="cursor-pointer font-medium text-text">Diagnostics</summary>
      <dl className="mt-3 grid gap-2">
        <div>
          <dt className="font-semibold text-text-muted">Route</dt>
          <dd>
            <code className="rounded bg-surface px-1 py-0.5">{module.route}</code>
          </dd>
        </div>
        <div>
          <dt className="font-semibold text-text-muted">Capability source</dt>
          <dd>{job?.capabilityMode ?? "frontend_state"}</dd>
        </div>
        <div>
          <dt className="font-semibold text-text-muted">Owner</dt>
          <dd>{job?.implementationOwner ?? "shared_route"}</dd>
        </div>
      </dl>
    </details>
  )
}

export const AdminOperationsOverviewPage: React.FC = () => {
  const adminJob = getOperationsRouteJob("/admin")

  return (
    <PageShell maxWidthClassName="max-w-6xl" className="py-8">
      <header className="space-y-3">
        <p className="text-sm font-semibold uppercase tracking-wide text-text-muted">
          Admin
        </p>
        <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <h1 className="text-3xl font-semibold text-text">Admin Operations</h1>
            <p className="mt-2 max-w-3xl text-sm text-text-muted">
              Review the available admin modules before opening a drill-down page.
              Live health, users, policies, sources, and monitoring data load inside
              each module.
            </p>
          </div>
          <a
            href="/admin/server"
            className="inline-flex w-fit rounded-md bg-primary px-3 py-2 text-sm font-medium text-white hover:bg-primaryStrong"
          >
            Open server admin
          </a>
        </div>
      </header>

      <section
        className="mt-8 grid gap-4 md:grid-cols-2"
        data-testid="admin-operations-modules"
      >
        {ADMIN_MODULES.map((module) => {
          const job = getOperationsRouteJob(module.route)
          const label = labelForJob(job, module.fallbackLabel)

          return (
            <article
              key={module.route}
              className="rounded-lg border border-border bg-surface p-5 shadow-sm"
              data-testid={`admin-module-${module.route}`}
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h2 className="text-lg font-semibold text-text">
                    <a className="hover:text-primary" href={module.route}>
                      {label}
                    </a>
                  </h2>
                  <p className="mt-2 text-sm text-text-muted">{module.summary}</p>
                </div>
                <span className="shrink-0 rounded-full border border-border bg-surface2 px-2 py-1 text-xs font-medium text-text">
                  {module.status}
                </span>
              </div>

              <p className="mt-4 text-sm text-text-muted">
                {module.statusDescription}
              </p>

              <ModuleDiagnostics job={job} module={module} />
            </article>
          )
        })}
      </section>

      <section className="mt-6 rounded-lg border border-border bg-surface p-4 text-sm text-text-muted">
        <p>
          {adminJob?.primaryJob ??
            "Review operations status and choose an admin module."}
        </p>
      </section>
    </PageShell>
  )
}

export default AdminOperationsOverviewPage
