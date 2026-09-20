import OptionLayout from "~/components/Layouts/Layout"
import DataOpsPage from "@/components/Option/Admin/DataOpsPage"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { AdminRouteShell } from "@/components/Option/Admin/AdminRouteShell"

const OptionAdminDataOps = () => {
  return (
    <RouteErrorBoundary routeId="admin-data-ops" routeLabel="Data Operations">
      <OptionLayout>
        <AdminRouteShell path="/admin/data-ops">
          <DataOpsPage />
        </AdminRouteShell>
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionAdminDataOps
