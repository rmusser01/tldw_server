import OptionLayout from "~/components/Layouts/Layout"
import UsageAnalyticsPage from "@/components/Option/Admin/UsageAnalyticsPage"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { AdminRouteShell } from "@/components/Option/Admin/AdminRouteShell"

const OptionAdminUsage = () => {
  return (
    <RouteErrorBoundary routeId="admin-usage" routeLabel="Usage Analytics">
      <OptionLayout>
        <AdminRouteShell path="/admin/usage">
          <UsageAnalyticsPage />
        </AdminRouteShell>
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionAdminUsage
