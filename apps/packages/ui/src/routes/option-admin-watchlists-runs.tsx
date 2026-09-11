import React from "react"
import { Link } from "react-router-dom"
import OptionLayout from "~/components/Layouts/Layout"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { AdminRouteShell } from "@/components/Option/Admin/AdminRouteShell"

/**
 * Shared coming-soon surface for /admin/watchlists-runs (#2897): the module
 * is advertised in the admin nav and overview, so every platform that can
 * reach the link must render an honest placeholder instead of a router
 * fallback. The web build serves its own Next-side RoutePlaceholder page;
 * this route covers the shared registry (extension and friends).
 */
const OptionAdminWatchlistsRuns = () => {
  return (
    <RouteErrorBoundary
      routeId="admin-watchlists-runs"
      routeLabel="Watchlist Runs"
    >
      <OptionLayout>
        <AdminRouteShell path="/admin/watchlists-runs">
          <div className="flex min-h-[60vh] w-full items-center justify-center px-6 py-12">
            <div className="w-full max-w-xl rounded-xl border border-border bg-surface p-8 shadow-sm">
              <p className="text-xs font-semibold uppercase tracking-wide text-text-muted">
                Coming Soon
              </p>
              <h1 className="mt-2 text-2xl font-semibold text-text">
                Watchlist Runs Admin Is Coming Soon
              </h1>
              <p className="mt-3 text-sm text-text-muted">
                Administrative run inspection for watchlists will land on this
                route. Use Watchlists for current run history and job
                management.
              </p>
              <div className="mt-6 flex flex-wrap gap-2">
                <Link
                  to="/watchlists"
                  className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-white hover:bg-primaryStrong"
                >
                  Open Watchlists
                </Link>
                <Link
                  to="/admin"
                  className="rounded-md border border-border px-3 py-1.5 text-sm text-text hover:bg-surface2"
                >
                  Back to Admin Operations
                </Link>
              </div>
            </div>
          </div>
        </AdminRouteShell>
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionAdminWatchlistsRuns
