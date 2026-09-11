import OptionLayout from "~/components/Layouts/Layout"
import ApiKeyManagementPage from "@/components/Option/Admin/ApiKeyManagementPage"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { AdminRouteShell } from "@/components/Option/Admin/AdminRouteShell"

const OptionAdminApiKeys = () => {
  return (
    <RouteErrorBoundary routeId="admin-api-keys" routeLabel="API Key Management">
      <OptionLayout>
        <AdminRouteShell path="/admin/api-keys">
          <ApiKeyManagementPage />
        </AdminRouteShell>
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionAdminApiKeys
