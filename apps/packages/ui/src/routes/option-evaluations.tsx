import OptionLayout from "~/components/Layouts/Layout"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { EvaluationsPlaygroundPage } from "@/components/Option/Evaluations/EvaluationsPlaygroundPage"

const OptionEvaluations = () => {
  return (
    <RouteErrorBoundary routeId="evaluations" routeLabel="Evaluations">
      <OptionLayout>
        <EvaluationsPlaygroundPage />
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionEvaluations
