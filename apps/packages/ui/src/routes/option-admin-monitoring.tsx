import OptionLayout from "~/components/Layouts/Layout"
import MonitoringDashboardPage from "@/components/Option/Admin/MonitoringDashboardPage"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { AdminRouteShell } from "@/components/Option/Admin/AdminRouteShell"

const OptionAdminMonitoring = () => {
  return (
    <RouteErrorBoundary routeId="admin-monitoring" routeLabel="Monitoring">
      <OptionLayout>
        <AdminRouteShell path="/admin/monitoring">
          <MonitoringDashboardPage />
        </AdminRouteShell>
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionAdminMonitoring
