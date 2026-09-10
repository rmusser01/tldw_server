import OptionLayout from "~/components/Layouts/Layout"
import RateLimitingPage from "@/components/Option/Admin/RateLimitingPage"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { AdminRouteShell } from "@/components/Option/Admin/AdminRouteShell"

const OptionAdminRateLimiting = () => {
  return (
    <RouteErrorBoundary routeId="admin-rate-limiting" routeLabel="Rate Limiting">
      <OptionLayout>
        <AdminRouteShell path="/admin/rate-limiting">
          <RateLimitingPage />
        </AdminRouteShell>
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionAdminRateLimiting
