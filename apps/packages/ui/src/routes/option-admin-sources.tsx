import OptionLayout from "@/components/Layouts/Layout"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { SourcesWorkspacePage } from "@/components/Option/Sources/SourcesWorkspacePage"
import { AdminRouteShell } from "@/components/Option/Admin/AdminRouteShell"

export default function OptionAdminSources() {
  return (
    <RouteErrorBoundary routeId="admin-sources" routeLabel="Sources">
      <OptionLayout>
        <AdminRouteShell path="/admin/sources">
          <SourcesWorkspacePage mode="admin" />
        </AdminRouteShell>
      </OptionLayout>
    </RouteErrorBoundary>
  )
}
