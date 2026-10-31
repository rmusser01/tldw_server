import OptionLayout from "@/components/Layouts/Layout"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { PresentationStudioNew } from "@/components/Option/PresentationStudio/PresentationStudioNew"

export default function OptionPresentationStudioNew() {
  return (
    <RouteErrorBoundary routeId="presentation-studio-new" routeLabel="Presentation Studio">
      <OptionLayout>
        <PresentationStudioNew />
      </OptionLayout>
    </RouteErrorBoundary>
  )
}
