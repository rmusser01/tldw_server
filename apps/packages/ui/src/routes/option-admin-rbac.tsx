import OptionLayout from "~/components/Layouts/Layout"
import RbacEditorPage from "@/components/Option/Admin/RbacEditorPage"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { AdminRouteShell } from "@/components/Option/Admin/AdminRouteShell"

const OptionAdminRbac = () => {
  return (
    <RouteErrorBoundary routeId="admin-rbac" routeLabel="RBAC / Permissions">
      <OptionLayout>
        <AdminRouteShell path="/admin/rbac">
          <RbacEditorPage />
        </AdminRouteShell>
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionAdminRbac
