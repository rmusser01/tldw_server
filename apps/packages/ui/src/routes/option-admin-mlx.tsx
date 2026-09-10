import OptionLayout from "~/components/Layouts/Layout"
import MlxAdminPage from "@/components/Option/Admin/MlxAdminPage"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { AdminRouteShell } from "@/components/Option/Admin/AdminRouteShell"

const OptionAdminMlx = () => {
  return (
    <RouteErrorBoundary routeId="admin-mlx" routeLabel="MLX Admin">
      <OptionLayout>
        <AdminRouteShell path="/admin/mlx">
          <MlxAdminPage />
        </AdminRouteShell>
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionAdminMlx
