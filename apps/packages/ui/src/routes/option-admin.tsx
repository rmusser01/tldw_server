import OptionLayout from "~/components/Layouts/Layout"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { AdminOperationsOverviewPage } from "@/components/Option/Admin/AdminOperationsOverviewPage"
import { AdminRouteShell } from "@/components/Option/Admin/AdminRouteShell"

const OptionAdmin = () => {
  return (
    <RouteErrorBoundary routeId="admin" routeLabel="Admin Operations">
      <OptionLayout>
        <AdminRouteShell path="/admin">
          <AdminOperationsOverviewPage />
        </AdminRouteShell>
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionAdmin
