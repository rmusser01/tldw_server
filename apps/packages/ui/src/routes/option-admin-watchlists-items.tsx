import OptionLayout from "~/components/Layouts/Layout"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { ItemsTab } from "@/components/Option/Watchlists/ItemsTab"
import { AdminRouteShell } from "@/components/Option/Admin/AdminRouteShell"

const OptionAdminWatchlistsItems = () => {
  return (
    <RouteErrorBoundary routeId="admin-watchlists-items" routeLabel="Watchlists Items">
      <OptionLayout>
        <AdminRouteShell path="/admin/watchlists-items">
          <div style={{ padding: "24px", maxWidth: "100%" }}>
            <h1 style={{ marginBottom: 4, fontSize: "1.5rem", fontWeight: 600 }}>
              Watchlists Items
            </h1>
            <p style={{ marginBottom: 16, color: "var(--color-text-secondary, #888)" }}>
              Review collected updates, alert matches, and briefing candidates
              across your watchlists. Create and configure watchlists on the
              Watchlists page.
            </p>
            <ItemsTab />
          </div>
        </AdminRouteShell>
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionAdminWatchlistsItems
