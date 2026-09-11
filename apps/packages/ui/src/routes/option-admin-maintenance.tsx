import OptionLayout from "~/components/Layouts/Layout"
import MaintenancePage from "@/components/Option/Admin/MaintenancePage"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { AdminRouteShell } from "@/components/Option/Admin/AdminRouteShell"

const OptionAdminMaintenance = () => {
  return (
    <RouteErrorBoundary routeId="admin-maintenance" routeLabel="Maintenance">
      <OptionLayout>
        <AdminRouteShell path="/admin/maintenance">
          <MaintenancePage />
        </AdminRouteShell>
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionAdminMaintenance
