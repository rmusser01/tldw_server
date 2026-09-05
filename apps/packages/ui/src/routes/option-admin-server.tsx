import OptionLayout from "~/components/Layouts/Layout"
import ServerAdminPage from "@/components/Option/Admin/ServerAdminPage"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { AdminRouteShell } from "@/components/Option/Admin/AdminRouteShell"

const OptionAdminServer = () => {
  return (
    <RouteErrorBoundary routeId="admin-server" routeLabel="Server Admin">
      <OptionLayout>
        <AdminRouteShell path="/admin/server">
          <ServerAdminPage />
        </AdminRouteShell>
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionAdminServer
