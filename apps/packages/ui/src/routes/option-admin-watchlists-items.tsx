import OptionLayout from "~/components/Layouts/Layout"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import WatchlistsOversightPage from "@/components/Option/Admin/WatchlistsOversightPage"
import { AdminRouteShell } from "@/components/Option/Admin/AdminRouteShell"

/**
 * Fleet oversight route (#2922): this admin surface inspects ANY user's
 * watchlists via an explicit user selector. It used to embed the personal
 * ItemsTab triage tool, which silently showed the operator's own (usually
 * empty) data - the personal tool lives on /watchlists.
 */
const OptionAdminWatchlistsItems = () => {
  return (
    <RouteErrorBoundary routeId="admin-watchlists-items" routeLabel="Watchlists Oversight">
      <OptionLayout>
        <AdminRouteShell path="/admin/watchlists-items">
          <WatchlistsOversightPage />
        </AdminRouteShell>
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionAdminWatchlistsItems
