import OptionLayout from "~/components/Layouts/Layout"
import OrgsTeamsPage from "@/components/Option/Admin/OrgsTeamsPage"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { AdminRouteShell } from "@/components/Option/Admin/AdminRouteShell"

const OptionAdminOrgs = () => {
  return (
    <RouteErrorBoundary routeId="admin-orgs" routeLabel="Organizations & Teams">
      <OptionLayout>
        <AdminRouteShell path="/admin/orgs">
          <OrgsTeamsPage />
        </AdminRouteShell>
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionAdminOrgs
