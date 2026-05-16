import OptionLayout from "~/components/Layouts/Layout"
import VllmAdminPage from "@/components/Option/Admin/VllmAdminPage"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"

const OptionAdminVllm = () => {
  return (
    <RouteErrorBoundary routeId="admin-vllm" routeLabel="vLLM Admin">
      <OptionLayout>
        <VllmAdminPage />
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionAdminVllm
