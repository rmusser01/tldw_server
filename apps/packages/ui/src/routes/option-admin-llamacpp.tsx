import OptionLayout from "~/components/Layouts/Layout"
import LlamacppAdminPage from "@/components/Option/Admin/LlamacppAdminPage"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { AdminRouteShell } from "@/components/Option/Admin/AdminRouteShell"

const OptionAdminLlamacpp = () => {
  return (
    <RouteErrorBoundary routeId="admin-llamacpp" routeLabel="Llama.cpp Admin">
      <OptionLayout>
        <AdminRouteShell path="/admin/llamacpp">
          <LlamacppAdminPage />
        </AdminRouteShell>
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionAdminLlamacpp
