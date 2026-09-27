import OptionLayout from "@/components/Layouts/Layout"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { IntegrationManagementPage } from "@/components/Option/Integrations/IntegrationManagementPage"
import { AdminRouteShell } from "@/components/Option/Admin/AdminRouteShell"

const OptionAdminIntegrations = () => {
  return (
    <RouteErrorBoundary routeId="admin-integrations" routeLabel="Integrations">
      <OptionLayout>
        <AdminRouteShell path="/admin/integrations">
          <IntegrationManagementPage scope="workspace" />
        </AdminRouteShell>
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionAdminIntegrations
