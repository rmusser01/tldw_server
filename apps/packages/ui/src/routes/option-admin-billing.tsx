import OptionLayout from "~/components/Layouts/Layout"
import BillingDashboardPage from "@/components/Option/Admin/BillingDashboardPage"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { AdminRouteShell } from "@/components/Option/Admin/AdminRouteShell"

const OptionAdminBilling = () => {
  return (
    <RouteErrorBoundary routeId="admin-billing" routeLabel="Billing">
      <OptionLayout>
        <AdminRouteShell path="/admin/billing">
          <BillingDashboardPage />
        </AdminRouteShell>
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionAdminBilling
