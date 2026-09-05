import React from "react"
import { PageShell } from "@/components/Common/PageShell"
import {
  ADMIN_MODULE_GROUPS,
  ADMIN_MODULES,
  type AdminModuleGroup
} from "./admin-modules"

const groupedModules = ADMIN_MODULE_GROUPS.map((group: AdminModuleGroup) => ({
  group,
  modules: ADMIN_MODULES.filter((module) => module.group === group)
})).filter((entry) => entry.modules.length > 0)

export const AdminOperationsOverviewPage: React.FC = () => {
  return (
    <PageShell maxWidthClassName="max-w-6xl" className="py-8">
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
                  </h3>
                  <p className="mt-1.5 text-sm text-text-muted">
                    {module.description}
                  </p>
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
