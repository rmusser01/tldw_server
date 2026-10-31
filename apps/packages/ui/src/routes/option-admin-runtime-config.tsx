import OptionLayout from "~/components/Layouts/Layout"
import RuntimeConfigPage from "@/components/Option/Admin/RuntimeConfigPage"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { AdminRouteShell } from "@/components/Option/Admin/AdminRouteShell"

const OptionAdminRuntimeConfig = () => {
  return (
    <RouteErrorBoundary routeId="admin-runtime-config" routeLabel="Runtime Config">
      <OptionLayout>
        <AdminRouteShell path="/admin/runtime-config">
          <RuntimeConfigPage />
        </AdminRouteShell>
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionAdminRuntimeConfig
